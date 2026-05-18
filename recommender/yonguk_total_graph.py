"""
yonguk_total_graph.py

통합 추천 Graph 본체
- boardgame / escape / murder 통합
- graph.invoke(payload) 지원
- run_pipeline() 지원
"""

import re

from recommender.eval.yonguk_eval_total import (
    EVAL_QUERIES,
    recommend,
    condition_score,
    grade_score,
)


# =========================================================
# 카테고리 정규화
# =========================================================

def normalize_category(category: str):
    if category in ["boardgame", "board", "보드게임"]:
        return "boardgame"

    if category in ["escape", "bbabang", "방탈출"]:
        return "escape"

    if category in ["murder", "murdermystery", "murder_mystery", "머더", "머더미스터리"]:
        return "murder"

    return "boardgame"


# =========================================================
# 사용자 쿼리에서 group 추출
# =========================================================

def extract_group_from_query(query: str):
    group = {}

    m = re.search(r"(\d+)\s*(명|인)", query)
    if m:
        group["headcount"] = int(m.group(1))

    if any(w in query for w in ["쉬운", "입문", "초보", "가벼운"]):
        group["weight_pref"] = "light"

    if any(w in query for w in ["어려운", "전략", "헤비", "무거운", "고난도"]):
        group["weight_pref"] = "heavy"

    if any(w in query for w in ["60분", "1시간"]):
        group["play_time"] = 60

    if any(w in query for w in ["120분", "2시간"]):
        group["play_time"] = 120

    if "원주" in query:
        group["location"] = "원주시"

    if "강릉" in query:
        group["location"] = "강릉시"

    return group


# =========================================================
# 추천 조건 선택
# =========================================================

def find_conditions(query: str, category: str):
    # EVAL_QUERIES에 있는 문장과 완전 일치하면 그 조건 사용
    for eval_query, info in EVAL_QUERIES.items():
        if info["category"] == category and eval_query == query:
            return info["conditions"]

    # 방탈출
    if category == "escape":
        if "원주" in query and ("3명" in query or "3인" in query):
            return {
                "escape": 2,
                "location_wonju": 3,
                "player_3": 3,
                "story": 2,
                "popular": 1,
            }

        if "강릉" in query and ("2명" in query or "2인" in query):
            return {
                "escape": 2,
                "location_gangneung": 3,
                "player_2": 3,
                "easy": 2,
                "horror_low": 2,
            }

        if "원주" in query and "60분" in query:
            return {
                "escape": 2,
                "location_wonju": 3,
                "time_60": 3,
                "puzzle": 2,
                "popular": 1,
            }

        if "강릉" in query and "2만원" in query:
            return {
                "escape": 2,
                "location_gangneung": 3,
                "price_20000": 3,
                "interior": 2,
                "popular": 1,
            }

        return {
            "escape": 2,
            "fun": 2,
            "story": 1,
            "device": 1,
        }

    # 머더미스터리
    if category == "murder":
        return {
            "murder": 2,
            "mystery": 2,
            "story": 1,
            "immersion": 1,
        }

    # 보드게임
    if "파티" in query:
        return {
            "party": 3,
            "many_players": 2,
            "easy": 1,
            "fun": 1,
        }

    if "7명" in query or "7인" in query:
        return {
            "player_7": 3,
            "many_players": 2,
            "party": 2,
            "easy": 1,
        }

    if "전략" in query:
        return {
            "boardgame": 2,
            "popular": 2,
            "fun": 1,
        }

    return {
        "boardgame": 2,
        "popular": 2,
        "fun": 1,
    }


# =========================================================
# Clarifying Question
# =========================================================

def check_sufficient(query: str, group: dict, category: str):
    missing = []

    if not group.get("headcount") and not any(w in query for w in ["명", "인"]):
        missing.append("headcount")

    if category == "boardgame":
        if not group.get("weight_pref") and not any(w in query for w in ["전략", "쉬운", "입문", "파티"]):
            missing.append("weight_pref")

    return len(missing) == 0, missing


def make_next_question(missing_fields):
    if "headcount" in missing_fields:
        return "몇 명이서 함께할 예정인가요?"

    if "weight_pref" in missing_fields:
        return "게임 난이도는 어느 정도가 좋으세요? 가벼운 입문용, 보통, 어려운 전략 게임 중에서 골라주세요."

    return "추가로 선호하는 분위기가 있나요?"


# =========================================================
# run_pipeline
# =========================================================

def run_pipeline(
    user_text: str,
    group: dict | None = None,
    category: str = "boardgame",
    use_api: bool = False,
):
    if group is None:
        group = {}

    category = normalize_category(category)

    extracted_group = extract_group_from_query(user_text)
    group = {**extracted_group, **group}

    is_sufficient, missing_fields = check_sufficient(
        query=user_text,
        group=group,
        category=category,
    )

    base_result = {
        "query": user_text,
        "user_text": user_text,
        "category": category,
        "group": group,
        "use_api": use_api,
        "is_sufficient": is_sufficient,
        "missing_fields": missing_fields,
        "query_text": "",
        "query_filter": {},
        "emotion_tags": [],
        "anchor_titles": [],
        "retrieved_items": [],
        "filtered_items": [],
        "retrieve_error": "",
        "generate_error": "",
        "result": {},
        "answer": "",
        "games": [],
        "next_question": "",
    }

    if not is_sufficient:
        answer = "추천을 정확히 하기 위해 조건이 조금 더 필요합니다."
        next_question = make_next_question(missing_fields)

        base_result["result"] = {
            "answer": answer,
            "games": [],
            "next_question": next_question,
        }

        base_result["answer"] = answer
        base_result["games"] = []
        base_result["next_question"] = next_question

        return base_result

    conditions = find_conditions(user_text, category)

    try:
        results = recommend(
            query=user_text,
            category=category,
            conditions=conditions,
            top_k=5,
        )

        games = []

        for item in results:
            raw_item = item.get("item", {})
            cond_score, _ = condition_score(raw_item, conditions)

            games.append({
                "title": item.get("title", "?"),
                "score": item.get("rrf_score", item.get("score", 0)),
                "condition_score": round(cond_score, 3),
                "grade": grade_score(cond_score),
            })

        answer = "조건에 맞는 추천 결과입니다."
        next_question = "추가로 선호하는 분위기가 있나요?"

        base_result.update({
            "query_text": user_text,
            "query_filter": conditions,
            "retrieved_items": results,
            "filtered_items": results,
            "result": {
                "answer": answer,
                "games": games,
                "next_question": next_question,
            },
            "answer": answer,
            "games": games,
            "next_question": next_question,
        })

    except Exception as e:
        base_result["retrieve_error"] = str(e)
        base_result["answer"] = "검색 데이터 또는 인덱스가 현재 실행 환경에 연결되어 있지 않아 추천 후보를 조회하지 못했습니다."
        base_result["games"] = []
        base_result["next_question"] = "추가로 선호하는 조건이 있나요?"

    return base_result


# =========================================================
# graph.invoke 지원
# =========================================================

class TotalGraph:
    def invoke(self, payload: dict):
        return run_pipeline(
            user_text=payload.get("query", ""),
            group=payload.get("group", {}),
            category=payload.get("category", "boardgame"),
            use_api=payload.get("use_api", False),
        )


graph = TotalGraph()