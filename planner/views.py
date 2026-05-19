import json
from django.shortcuts import render
from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse                          # ✅ networkx.reverse → django.urls.reverse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator


DEFAULT_PERSONA = {
    "groupSize": "4", "relationship": "friends",
    "horrorTolerance": "low", "activityLevel": "moderate", "budget": "20000",
}


# @method_decorator(csrf_exempt, name="dispatch")
# class PlannerCalculateView(View):
#     """
#     플레이타임 계산 API
#     POST /planner/calculate/
#     body: {
#         "players": 4,
#         "games": [
#             {"name": "카탄", "play_time": 90, "players": "3~4"},
#             {"name": "웬디", "play_time": 120, "players": "4~5"}
#         ]
#     }
#     """

#     def post(self, request):
#         try:
#             body    = json.loads(request.body)
#             games   = body.get("games", [])
#             players = int(body.get("players", 0))
#         except (json.JSONDecodeError, ValueError):
#             return JsonResponse({"error": "잘못된 요청 형식입니다."}, status=400)

#         if not games:
#             return JsonResponse({"error": "게임을 선택해주세요."}, status=400)

#         time_result    = calculate_total_time(games)
#         compat_result  = check_player_compatibility(games, players) if players else []

#         return JsonResponse({
#             "time":          time_result,
#             "compatibility": compat_result,
#         })
    
@method_decorator(csrf_exempt, name="dispatch")
class PersonaView(View):

    def get(self, request):                              # ✅ def persona → def get (GET 요청 처리)
        return render(request, "planner/persona.html", { # ✅ 템플릿 경로: planner 앱에 맞게 수정
            "current_page": "persona",
        })
 
    def post(self, request):                             # ✅ POST도 별도 메서드로 분리
        # 현재 persona.html은 localStorage 기반이라 서버 저장 불필요
        # 추후 서버 저장이 필요하면 여기에 로직 추가
        return JsonResponse({"status": "ok"})
 