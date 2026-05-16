import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.core.paginator import Paginator
from .models import BoardGame, Escape, CrimeScene

CAT_EMOJI  = {"escape": "🔐", "boardgame": "🎲", "crimescene": "🕵️"}
CAT_LABEL  = {"escape": "방탈출", "boardgame": "보드게임", "crimescene": "크라임씬"}


# =====================================================================
# 카테고리 탐색 메인 페이지
# =====================================================================

class ExploreView(View):
    """GET /contents/explore/"""

    def get(self, request):
        category   = request.GET.get("category", "all")
        search     = request.GET.get("search", "").strip().lower()
        difficulty = request.GET.get("difficulty", "")

        activities = []

        # ── 보드게임 ──────────────────────────────────────
        if category in ("all", "boardgame"):
            qs = BoardGame.objects.all()
            if search:
                qs = qs.filter(name__icontains=search)
            if difficulty:
                qs = qs.filter(difficulty=difficulty)
            for g in qs[:50]:
                activities.append({
                    "id":         g.pk,
                    "title":      g.name,
                    "category":   "boardgame",
                    "emoji":      "🎲",
                    "cat_label":  "보드게임",
                    "rating":     g.rating or 0,
                    "players":    g.players_display,
                    "time":       g.play_time_display,
                    "difficulty": g.difficulty,
                    "horror":     None,
                    "tags":       g.tags[:4] if isinstance(g.tags, list) else [],
                    "description": (g.description or "")[:80],
                })

        # ── 방탈출 ────────────────────────────────────────
        if category in ("all", "escape"):
            qs = Escape.objects.all()
            if search:
                qs = qs.filter(name__icontains=search)
            if difficulty:
                qs = qs.filter(difficulty=difficulty)
            for g in qs[:50]:
                # 공포도 텍스트 변환
                if g.fear_level is None or g.fear_level == 0:
                    horror = "공포 없음"
                elif g.fear_level <= 2:
                    horror = "공포 약함"
                elif g.fear_level <= 3:
                    horror = "공포 있음"
                else:
                    horror = "공포 강함"

                activities.append({
                    "id":         g.pk,
                    "title":      g.name,
                    "category":   "escape",
                    "emoji":      "🔐",
                    "cat_label":  "방탈출",
                    "rating":     g.rating or 0,
                    "players":    g.players_display,
                    "time":       g.play_time_display,
                    "difficulty": g.difficulty,
                    "horror":     horror,
                    "tags":       g.tags[:4] if isinstance(g.tags, list) else [],
                    "description": (g.description or "")[:80],
                })

        # ── 머더미스터리 (크라임씬) ───────────────────────
        if category in ("all", "crimescene"):
            qs = CrimeScene.objects.all()
            if search:
                qs = qs.filter(name__icontains=search)
            if difficulty:
                qs = qs.filter(difficulty=difficulty)
            for g in qs[:50]:
                activities.append({
                    "id":          g.pk,
                    "title":       g.name,
                    "category":    "crimescene",
                    "emoji":       "🕵️",
                    "cat_label":   "크라임씬",
                    "rating":      g.rating or 0,
                    "players":     g.players_display,
                    "time":        g.play_time_display,
                    "difficulty":  g.difficulty,
                    "horror":      None,
                    "tags":        g.tags[:4] if isinstance(g.tags, list) else [],
                    "description": (g.description or "")[:80],
                })

        # 태그 검색 (이름 검색 미포함 태그 대상)
        if search:
            activities = [
                a for a in activities
                if search in a["title"].lower()
                or any(search in t.lower() for t in a["tags"])
                or search in (a["description"] or "").lower()
            ]

        return render(request, "contents/explore.html", {
            "current_page":   "explore",
            "activities":     activities,
            "total":          len(activities),
            "sel_category":   category,
            "sel_search":     request.GET.get("search", ""),
            "sel_difficulty": difficulty,
        })


# =====================================================================
# 보드게임
# =====================================================================

class BoardGameListView(View):
    def get(self, request):
        qs = BoardGame.objects.all()
        if q := request.GET.get("q", ""):
            qs = qs.filter(name__icontains=q)
        if d := request.GET.get("difficulty", ""):
            qs = qs.filter(difficulty=d)
        if p := request.GET.get("min_players", ""):
            qs = qs.filter(players_max__gte=int(p))
        if t := request.GET.get("max_time", ""):
            qs = qs.filter(play_time__lte=int(t))

        paginator = Paginator(qs, 12)
        page      = paginator.get_page(int(request.GET.get("page", 1)))
        games     = [_serialize_boardgame(g) for g in page.object_list]

        if request.headers.get("Accept") == "application/json":
            return JsonResponse({"results": games, "total_pages": paginator.num_pages, "count": paginator.count})
        return render(request, "contents/boardgame/list.html", {"games": page, "q": q})


class BoardGameDetailView(View):
    def get(self, request, pk):
        try:
            game = BoardGame.objects.get(pk=pk)
        except BoardGame.DoesNotExist:
            return JsonResponse({"error": "게임을 찾을 수 없습니다."}, status=404)
        data = _serialize_boardgame(game, detail=True)
        if request.headers.get("Accept") == "application/json":
            return JsonResponse(data)
        return render(request, "contents/boardgame/detail.html", {"game": data})


# =====================================================================
# 방탈출
# =====================================================================

class EscapeListView(View):
    def get(self, request):
        qs = Escape.objects.all()
        if q := request.GET.get("q", ""):
            qs = qs.filter(name__icontains=q)
        if r := request.GET.get("region", ""):
            qs = qs.filter(region=r)
        if d := request.GET.get("difficulty", ""):
            qs = qs.filter(difficulty=d)
        if h := request.GET.get("max_horror", ""):
            qs = qs.filter(fear_level__lte=int(h))

        paginator = Paginator(qs, 12)
        page      = paginator.get_page(int(request.GET.get("page", 1)))
        games     = [_serialize_escape(g) for g in page.object_list]

        if request.headers.get("Accept") == "application/json":
            return JsonResponse({"results": games, "total_pages": paginator.num_pages, "count": paginator.count})
        return render(request, "contents/escape/list.html", {"games": page})


class EscapeDetailView(View):
    def get(self, request, pk):
        try:
            game = Escape.objects.get(pk=pk)
        except Escape.DoesNotExist:
            return JsonResponse({"error": "게임을 찾을 수 없습니다."}, status=404)
        data = _serialize_escape(game, detail=True)
        if request.headers.get("Accept") == "application/json":
            return JsonResponse(data)
        return render(request, "contents/escape/detail.html", {"game": data})


# =====================================================================
# 머더미스터리
# =====================================================================

class CrimeSceneListView(View):
    def get(self, request):
        qs = CrimeScene.objects.all()
        if q := request.GET.get("q", ""):
            qs = qs.filter(name__icontains=q)
        if d := request.GET.get("difficulty", ""):
            qs = qs.filter(difficulty=d)
        if p := request.GET.get("min_players", ""):
            qs = qs.filter(players_max__gte=int(p))
        if t := request.GET.get("max_time", ""):
            qs = qs.filter(play_time__lte=int(t))

        paginator = Paginator(qs, 12)
        page      = paginator.get_page(int(request.GET.get("page", 1)))
        games     = [_serialize_crimescene(g) for g in page.object_list]

        if request.headers.get("Accept") == "application/json":
            return JsonResponse({"results": games, "total_pages": paginator.num_pages, "count": paginator.count})
        return render(request, "contents/crimescene/list.html", {"games": page})


class CrimeSceneDetailView(View):
    def get(self, request, pk):
        try:
            game = CrimeScene.objects.get(pk=pk)
        except CrimeScene.DoesNotExist:
            return JsonResponse({"error": "게임을 찾을 수 없습니다."}, status=404)
        data = _serialize_crimescene(game, detail=True)
        if request.headers.get("Accept") == "application/json":
            return JsonResponse(data)
        return render(request, "contents/crimescene/detail.html", {"game": data})


# =====================================================================
# 직렬화 헬퍼
# =====================================================================

def _serialize_boardgame(g, detail=False):
    data = {
        "id": g.pk, "category": "boardgame",
        "name": g.name, "rating": g.rating,
        "players": g.players_display, "play_time": g.play_time_display,
        "difficulty": g.difficulty, "image_url": g.image_url,
        "tags": g.tags if isinstance(g.tags, list) else [],
        "publisher": g.publisher, "designer": g.designer,
        "bgg_rank": g.bgg_rank, "mechanism": g.mechanism,
    }
    if detail:
        reviews = g.reviews if isinstance(g.reviews, list) else []
        data["reviews"]     = reviews[:3]
        data["description"] = g.description
    return data


def _serialize_escape(g, detail=False):
    if g.fear_level is None or g.fear_level == 0:
        horror_text = "공포 없음"
    elif g.fear_level <= 2:
        horror_text = "약함"
    elif g.fear_level <= 3:
        horror_text = "중간"
    else:
        horror_text = "강함"

    data = {
        "id": g.pk, "category": "escape",
        "name": g.name, "rating": g.rating,
        "players": g.players_display, "play_time": g.play_time_display,
        "difficulty": g.difficulty, "image_url": g.image_url,
        "tags": g.tags if isinstance(g.tags, list) else [],
        "region": g.region, "brand": g.brand,
        "theme": g.theme, "horror_level": horror_text, "fear_level": g.fear_level,
    }
    if detail:
        reviews = g.reviews if isinstance(g.reviews, list) else []
        data["reviews"]     = reviews[:3]
        data["description"] = g.description
    return data


def _serialize_crimescene(g, detail=False):
    data = {
        "id": g.pk, "category": "crimescene",
        "name": g.name, "rating": g.rating,
        "players": g.players_display, "play_time": g.play_time_display,
        "difficulty": g.difficulty, "image_url": g.image_url,
        "tags": g.tags if isinstance(g.tags, list) else [],
        "series": g.series, "maker": g.maker,
        "publisher": g.publisher, "publisher_kr": g.publisher_kr,
    }
    if detail:
        reviews = g.reviews if isinstance(g.reviews, list) else []
        if reviews and isinstance(reviews[0], str) and "||" in reviews[0]:
            reviews = [r.strip() for r in reviews[0].split("||") if r.strip()]
        data["reviews"]     = reviews[:3]
        data["description"] = g.description
    return data