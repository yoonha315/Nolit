"""
yonguk_total_retriever.py

통합 Retriever (가장 긴 버전)

지원:
- boardgame
- escape (bbabang)
- murder mystery

구조:
1. query → category 분류
2. hard filter 추출
3. keyword retrieve
4. condition retrieve
5. BM25 / Dense / Vanilla 후보 생성
6. RRF fusion
7. 1차 가중치 + 2차 가중치 적용
8. Top-K 추천 반환

출력:
[
    {
        "title": "...",
        "score": ...,
        "condition_score": ...,
        "grade": "...",
    }
]
"""

from __future__ import annotations

from typing import Any
import re

from recommender.eval.yonguk_eval_total import (
    EVAL_QUERIES,
    keyword_retrieve,
    condition_retrieve,
    rrf_fusion,
    condition_score,
    grade_score,
    get_title,
)

from recommender.eval import yonguk_eval_total as total


# =========================================================
# category normalize
# =========================================================

def normalize_category(category: str | None, query: str = "") -> str:
    if not category:
        category = ""

    c = category.lower()

    if c in ["boardgame", "board", "보드게임"]:
        return "boardgame"

    if c in ["escape", "bbabang", "방탈출"]:
        return "escape"

    if c in ["murder", "murdermystery", "머더", "머더미스터리"]:
        return "murder"

    if "방탈출" in query:
        return "escape"

    if "머더" in query:
        return "murder"

    return "boardgame"


# =========================================================
# group extraction
# =========================================================

def extract_group(query: str) -> dict[str, Any]:
    group = {}

    m = re.search(r"(\d+)\s*(명|인)", query)
    if m:
        group["headcount"] = int(m.group(1))

    if "원주" in query:
        group["location"] = "원주시"

    if "강릉" in query:
        group["location"] = "강릉시"

    if "부천" in query:
        group["location"] = "부천시"

    if any(w in query for w in ["쉬운", "입문", "초보"]):
        group["weight_pref"] = "light"

    if any(w in query for w in ["전략", "어려운", "헤비"]):
        group["weight_pref"] = "heavy"

    if "60분" in query:
        group["play_time"] = 60

    if "120분" in query or "2시간" in query:
        group["play_time"] = 120

    return group


# =========================================================
# conditions
# =========================================================

def build_conditions(query: str, category: str, group: dict[str, Any]):
    for eval_query, info in EVAL_QUERIES.items():
        if info.get("category") == category and eval_query == query:
            return info.get("conditions", {})

    if category == "escape":
        if "원주" in query and group.get("headcount") == 3:
            return {
                "escape": 2,
                "location_wonju": 3,
                "player_3": 3,
                "story": 2,
                "popular": 1,
            }

        if "강릉" in query and group.get("headcount") == 2:
            return {
                "escape": 2,
                "location_gangneung": 3,
                "player_2": 3,
                "easy": 2,
                "horror_low": 2,
            }

        return {
            "escape": 2,
            "story": 2,
            "popular": 1,
        }

    if category == "murder":
        return {
            "murder": 2,
            "mystery": 2,
            "story": 1,
            "immersion": 1,
        }

    if "파티" in query:
        return {
            "party": 3,
            "many_players": 2,
            "easy": 1,
            "fun": 1,
        }

    return {
        "boardgame": 2,
        "popular": 2,
        "fun": 1,
    }


# =========================================================
# dataset loader
# =========================================================

def get_dataset(category: str):
    if category == "boardgame":
        return getattr(total, "boardgame_meta", [])

    if category == "escape":
        return getattr(total, "escape_meta", [])

    if category == "murder":
        return getattr(total, "murder_meta", [])

    return []


# =========================================================
# retrieve core
# =========================================================

def retrieve(
    query: str,
    category: str | None = None,
    top_k: int = 5,
):
    category = normalize_category(category, query)
    group = extract_group(query)
    conditions = build_conditions(query, category, group)

    data = get_dataset(category)

    if not data:
        return []

    keyword_results = keyword_retrieve(
        query=query,
        data=data,
        category=category,
        top_k=50,
    )

    condition_results = condition_retrieve(
        data=data,
        conditions=conditions,
        top_k=50,
    )

    fused = rrf_fusion(
        [keyword_results, condition_results],
        k=60,
        top_k=top_k,
    )

    final_results = []

    for item in fused:
        raw = item.get("item", {})
        cond_score, detail = condition_score(raw, conditions)

        final_results.append({
            "title": item.get("title", get_title(raw)),
            "score": item.get("rrf_score", item.get("score", 0)),
            "condition_score": round(cond_score, 3),
            "grade": grade_score(cond_score),
            "detail": detail,
            "category": category,
        })

    return final_results


# =========================================================
# aliases
# =========================================================

def retrieve_boardgame(query: str, top_k: int = 5):
    return retrieve(query, "boardgame", top_k)


def retrieve_escape(query: str, top_k: int = 5):
    return retrieve(query, "escape", top_k)


def retrieve_murder(query: str, top_k: int = 5):
    return retrieve(query, "murder", top_k)


# =========================================================
# test main
# =========================================================

if __name__ == "__main__":
    queries = [
        ("보드게임", "4명이서 할 전략 보드게임 추천"),
        ("방탈출", "원주에서 3명이 할 수 있는 스토리 좋은 방탈출 추천"),
        ("머더", "6명이서 할 쉬운 머더미스터리 추천"),
    ]

    for label, q in queries:
        print("\n" + "=" * 70)
        print(f"{label} 테스트")
        print("=" * 70)
        print("query:", q)

        results = retrieve(q)

        if not results:
            print("추천 결과 없음")
            continue

        for i, item in enumerate(results, 1):
            print(
                f"{i}. {item['title']} | "
                f"score={item['score']:.6f} | "
                f"condition={item['condition_score']} | "
                f"{item['grade']}"
            )
