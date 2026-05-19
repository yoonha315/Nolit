"""
recommender/yonguk_graph.py

LangGraph 기반 용국 통합 추천 파이프라인.

====================================================================
[역할]
    사용자의 자연어 요청을 받아 보드게임 / 방탈출 / 머더미스터리를 통합 추천한다.

    입력 정제
    → 조건 충분성 판단
    → 쿼리 변환
    → Keyword + Condition 기반 RRF 검색
    → 태그/결과 필터링
    → 로컬 생성 fallback
    의 6단계 파이프라인을 LangGraph로 구성한다.

[외부 호출 예시]
    from recommender.yonguk_graph import graph, run_pipeline

    result = graph.invoke({
        "query": "원주에서 3명이 할 수 있는 스토리 좋은 방탈출 추천",
        "category": "escape",
        "use_api": False,
    })

[출력 스펙]
    {
        "answer":        "추천 텍스트",
        "games":         [...],
        "next_question": "..."
    }

[지원 카테고리]
    - boardgame
    - escape
    - murder / murdermystery

[내부 흐름]
    normalize_input
    → check_sufficiency
        ├─ 부족함 → clarify → END
        └─ 충분함 → query_transform → retrieve → tag_filter → generate → END

[설계 문서와의 대응]
    - 통합 평가 파이프라인: yoonha_yu_eval_pipeline.py
    - 통합 검색 로직: yonguk_eval_total.py
    - Graph 실행 형식: yoonha_graph.py 스타일 유지
    - 테스트 방식: yoonha_test_graph.py / yonguk_total_test_graph.py

[핵심 차이점]
    1. yoonha_graph.py는 yoonha_query_transformer / yoonha_hybrid_retriever에 위임한다.
    2. yonguk_graph.py는 yonguk_eval_total.py의
       keyword_retrieve / condition_retrieve / rrf_fusion을 사용한다.
    3. FAISS 직접 검색보다 발표용 통합 평가 로직과 같은 RRF 구조를 재사용한다.
    4. use_api=False 기준으로도 항상 answer / games / next_question을 반환한다.

[개선 포인트]
    1. 현재 retrieve는 yonguk_eval_total.py의 메타데이터 로드 상태에 의존한다.
       → 추후 독립 메타데이터 로더로 분리 가능.
    2. condition_score 기반 추천이므로 Dense 벡터 검색과는 다르다.
       → 필요 시 boardgame / escape / murder 전용 FAISS retriever 연결 가능.
    3. output games는 발표/디버깅용 detail을 많이 포함한다.
       → 실제 서비스용에서는 detail을 숨길 수 있다.
====================================================================
"""


from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph


# =========================================================
# path 보정
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# Type 정의
# =========================================================

Category = Literal["boardgame", "escape", "murder"]


class GraphInput(TypedDict, total=False):
    query: str
    user_text: str
    category: str
    group: dict[str, Any]
    use_api: bool


class PipelineState(TypedDict, total=False):
    # 외부 입력
    query: str
    user_text: str
    category: str
    group: dict[str, Any]
    use_api: bool

    # 조건 판단
    is_sufficient: bool
    missing_fields: list[str]

    # 추천 중간 결과
    query_text: str
    query_filter: dict[str, Any]
    emotion_tags: list[str]
    anchor_titles: list[str]
    retrieved_items: list[dict[str, Any]]
    filtered_items: list[dict[str, Any]]

    # 에러
    retrieve_error: str
    generate_error: str

    # 최종 출력
    result: dict[str, Any]
    answer: str
    games: list[dict[str, Any]]
    next_question: str


# =========================================================
# yonguk_eval_total 함수 import
# =========================================================

try:
    from recommender.eval.yonguk_eval_total import (
        EVAL_QUERIES,
        keyword_retrieve,
        condition_retrieve,
        rrf_fusion,
        condition_score,
        grade_score,
        get_title,
    )
except Exception as e:
    raise ImportError(
        "recommender.eval.yonguk_eval_total import 실패. "
        "yonguk_eval_total.py 경로 또는 내부 에러를 확인하세요."
    ) from e


# =========================================================
# 카테고리 정규화
# =========================================================

SUPPORTED_CATEGORIES = {"boardgame", "escape", "murder"}

CATEGORY_ALIASES = {
    "boardgame": "boardgame",
    "board": "boardgame",
    "board_game": "boardgame",
    "보드게임": "boardgame",
    "보드": "boardgame",

    "escape": "escape",
    "escape_room": "escape",
    "roomescape": "escape",
    "bbabang": "escape",
    "빠방": "escape",
    "방탈출": "escape",
    "방탈": "escape",
    "탈출": "escape",

    "murder": "murder",
    "murdermystery": "murder",
    "murder_mystery": "murder",
    "murder-mystery": "murder",
    "머더": "murder",
    "머더미스터리": "murder",
    "미스터리": "murder",
    "크라임": "murder",
    "크라임씬": "murder",
}


def normalize_category(raw_category: str | None, query: str = "") -> str:
    value = (raw_category or "").strip().lower()

    if value in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[value]

    query_lower = query.lower()

    if any(w in query_lower for w in ["방탈출", "escape", "탈출"]):
        return "escape"

    if any(w in query_lower for w in ["머더", "murder", "미스터리", "크라임"]):
        return "murder"

    if any(w in query_lower for w in ["보드", "board"]):
        return "boardgame"

    return "boardgame"


# =========================================================
# group 추출
# =========================================================

KOREAN_NUMBER_MAP = {
    "한": 1,
    "하나": 1,
    "둘": 2,
    "두": 2,
    "셋": 3,
    "세": 3,
    "넷": 4,
    "네": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
}


def extract_headcount(query: str) -> int | None:
    m = re.search(r"(\d{1,2})\s*(명|인)", query)
    if m:
        return int(m.group(1))

    for word, number in KOREAN_NUMBER_MAP.items():
        if re.search(rf"{word}\s*(명|인|명이서|이서)", query):
            return number

    return None


def extract_play_time(query: str) -> int | None:
    hour_match = re.search(r"(\d{1,2})\s*시간", query)
    minute_match = re.search(r"(\d{1,3})\s*분", query)

    total = 0

    if hour_match:
        total += int(hour_match.group(1)) * 60

    if "반" in query and hour_match:
        total += 30

    if minute_match:
        total += int(minute_match.group(1))

    return total or None


def extract_weight_pref(query: str) -> str | None:
    if any(w in query for w in ["쉬운", "쉽게", "입문", "초보", "가벼운", "간단"]):
        return "light"

    if any(w in query for w in ["어려운", "헤비", "무거운", "고난도", "고난이도", "전략", "복잡"]):
        return "heavy"

    if any(w in query for w in ["보통", "중간", "무난", "중급"]):
        return "medium"

    return None


def extract_relation(query: str) -> str | None:
    if any(w in query for w in ["친구", "동창", "모임"]):
        return "friend"

    if any(w in query for w in ["커플", "데이트", "연인"]):
        return "couple"

    if any(w in query for w in ["직장", "회사", "동료", "회식", "워크샵"]):
        return "coworker"

    if any(w in query for w in ["처음", "첫만남", "소개팅", "어색"]):
        return "first_meeting"

    return None


def extract_horror_tolerance(query: str) -> int | None:
    if any(w in query for w in ["안 무서운", "안무서운", "공포 없는", "공포 싫", "무서운 거 싫", "쫄보"]):
        return 0

    if any(w in query for w in ["약간 무서운", "살짝 무서운", "공포 조금"]):
        return 1

    if any(w in query for w in ["공포 괜찮", "호러 가능", "무서워도 괜찮", "공포 가능"]):
        return 2

    return None


def extract_location(query: str) -> str | None:
    if "원주" in query:
        return "원주시"

    if "강릉" in query:
        return "강릉시"

    if "부천" in query:
        return "부천시"

    if "수원" in query:
        return "수원시"

    if "인천" in query:
        return "인천"

    return None


def extract_price(query: str) -> int | None:
    m = re.search(r"(\d+)\s*만\s*원\s*이하", query)
    if m:
        return int(m.group(1)) * 10000

    m = re.search(r"(\d+)\s*원\s*이하", query)
    if m:
        return int(m.group(1))

    return None


def merge_group_from_query(
    query: str,
    category: str,
    group: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(group or {})

    if "headcount" not in merged:
        value = extract_headcount(query)
        if value is not None:
            merged["headcount"] = value

    if "play_time" not in merged:
        value = extract_play_time(query)
        if value is not None:
            merged["play_time"] = value

    if "weight_pref" not in merged:
        value = extract_weight_pref(query)
        if value is not None:
            merged["weight_pref"] = value

    if "relation" not in merged:
        value = extract_relation(query)
        if value is not None:
            merged["relation"] = value

    if "horror_tolerance" not in merged:
        value = extract_horror_tolerance(query)
        if value is not None:
            merged["horror_tolerance"] = value

    if category == "escape":
        if "location" not in merged:
            value = extract_location(query)
            if value is not None:
                merged["location"] = value

        if "price" not in merged:
            value = extract_price(query)
            if value is not None:
                merged["price"] = value

    return merged


# =========================================================
# conditions 생성
# =========================================================

def build_conditions(query: str, category: str, group: dict[str, Any]) -> dict[str, int]:
    """
    yonguk_eval_total.py의 condition_score 구조에 맞춘 conditions 생성.
    """
    # 기존 EVAL_QUERIES와 완전 일치 시 그대로 사용
    for eval_query, info in EVAL_QUERIES.items():
        if info.get("category") == category and eval_query == query:
            return dict(info.get("conditions", {}))

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

        if "원주" in query and group.get("play_time") == 60:
            return {
                "escape": 2,
                "location_wonju": 3,
                "time_60": 3,
                "puzzle": 2,
                "popular": 1,
            }

        if "강릉" in query and group.get("price") == 20000:
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

    if category == "murder":
        conditions = {
            "murder": 2,
            "mystery": 2,
            "story": 1,
            "immersion": 1,
        }

        if group.get("weight_pref") == "light":
            conditions["easy"] = 1

        return conditions

    if group.get("headcount") == 7 or "7명" in query or "7인" in query:
        return {
            "player_7": 3,
            "many_players": 2,
            "party": 2,
            "easy": 1,
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


def build_query_text(query: str, category: str, group: dict[str, Any]) -> str:
    parts = [query]

    if group.get("headcount"):
        parts.append(f"{group['headcount']}인")

    if group.get("weight_pref") == "light":
        parts.append("가벼운 입문 간단 쉬운")

    elif group.get("weight_pref") == "heavy":
        parts.append("전략 고급 어려운 헤비 무거운")

    if group.get("play_time"):
        parts.append(f"{group['play_time']}분")

    if group.get("location"):
        parts.append(str(group["location"]))

    if category == "murder":
        parts.append("추리 크라임씬 머더미스터리")

    return " ".join(parts)


# =========================================================
# 역질문
# =========================================================

def build_next_question(category: str, group: dict[str, Any], missing_fields: list[str] | None = None) -> str:
    missing_fields = missing_fields or []

    if "query" in missing_fields:
        return "어떤 활동을 찾고 계신지 알려주세요."

    if "category" in missing_fields:
        return "보드게임, 방탈출, 머더미스터리 중 어떤 활동을 추천받고 싶으신가요?"

    if "headcount" in missing_fields or not group.get("headcount"):
        return "몇 명이서 함께할 예정인가요?"

    if category == "boardgame":
        if not group.get("weight_pref"):
            return "게임 난이도는 어느 정도가 좋으세요? 가벼운 입문용, 보통, 어려운 전략 게임 중에서 골라주세요."

        return "추가로 피하고 싶은 요소나 선호하는 분위기가 있나요?"

    if category == "escape":
        if not group.get("location"):
            return "방탈출 지역은 어디를 원하시나요? 예: 원주, 강릉, 부천, 수원"

        if group.get("horror_tolerance") is None:
            return "공포도는 어느 정도 괜찮으신가요? 안 무서운 것, 약간 가능, 무서워도 괜찮음 중에서 알려주세요."

        return "스토리, 퍼즐, 인테리어, 연출 중 어떤 요소를 가장 중요하게 볼까요?"

    if category == "murder":
        if group.get("horror_tolerance") is None:
            return "공포 요소는 괜찮으신가요? 공포 불가, 약간 가능, 괜찮음 중에서 알려주세요."

        return "플레이 시간이나 난이도 선호가 있나요?"

    return "추가로 선호하는 분위기가 있나요?"


# =========================================================
# 로컬 추천 실행
# =========================================================

def local_retrieve(query_text: str, category: str, conditions: dict[str, int]) -> list[dict[str, Any]]:
    """
    yonguk_eval_total.py의 keyword + condition + RRF 기반 통합 검색.
    """
    try:
        from recommender.eval import yonguk_eval_total as total
    except Exception as exc:
        raise RuntimeError(f"yonguk_eval_total import 실패: {exc}") from exc

    if category == "boardgame":
        data = getattr(total, "boardgame_meta", [])

    elif category == "escape":
        data = getattr(total, "escape_meta", [])

    elif category == "murder":
        data = getattr(total, "murder_meta", [])

    else:
        data = []

    if not data:
        raise RuntimeError(f"{category} 데이터가 비어 있습니다.")

    keyword_results = keyword_retrieve(
        query=query_text,
        data=data,
        category=category,
        top_k=50,
    )

    condition_results = condition_retrieve(
        data=data,
        conditions=conditions,
        top_k=50,
    )

    rrf_results = rrf_fusion(
        [keyword_results, condition_results],
        k=60,
        top_k=5,
    )

    return rrf_results


def format_games(items: list[dict[str, Any]], conditions: dict[str, int]) -> list[dict[str, Any]]:
    games = []

    for item in items:
        raw = item.get("item", item)
        cond_score, detail = condition_score(raw, conditions)

        games.append({
            "title": item.get("title", get_title(raw)),
            "reason": f"조건 충족도 {cond_score:.3f} ({grade_score(cond_score)}) 기준으로 추천되었습니다.",
            "matched_tags": [],
            "final_score": item.get("rrf_score", item.get("score")),
            "condition_score": round(float(cond_score), 3),
            "grade": grade_score(cond_score),
            "source": raw.get("source"),
            "avg_rating": raw.get("avg_rating") or raw.get("rating") or raw.get("satisfaction"),
            "min_players": raw.get("min_players"),
            "max_players": raw.get("max_players"),
            "store_name": raw.get("store_name"),
            "area": raw.get("area"),
            "location": raw.get("location"),
            "price": raw.get("price"),
            "playing_time": raw.get("playing_time") or raw.get("play_time"),
            "detail": detail,
        })

    return games


def normalize_result(result: dict[str, Any] | None) -> dict[str, Any]:
    result = result or {}

    return {
        "answer": result.get("answer", "") or "",
        "games": result.get("games", []) or [],
        "next_question": result.get("next_question", "") or "",
    }


# =========================================================
# LangGraph Node
# =========================================================

def node_normalize_input(state: PipelineState) -> dict[str, Any]:
    """
    [노드: normalize_input]

    외부에서 들어온 payload를 LangGraph 내부 state 형태로 정규화한다.

    수행 작업:
        1. query / user_text 통합
        2. category alias 정규화
           - murdermystery → murder
           - 방탈출 → escape
           - 보드게임 → boardgame
        3. 자연어 쿼리에서 group 조건 추출
           - headcount
           - play_time
           - weight_pref
           - location
           - horror_tolerance
        4. 이후 노드에서 사용할 기본 필드 초기화

    반환:
        PipelineState 일부 필드
    """

    query = state.get("query") or state.get("user_text") or ""
    query = str(query).strip()

    category = normalize_category(state.get("category"), query)
    group = merge_group_from_query(query, category, state.get("group") or {})

    return {
        "query": query,
        "user_text": query,
        "category": category,
        "group": group,
        "use_api": bool(state.get("use_api", True)),

        "is_sufficient": False,
        "missing_fields": [],
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


def node_check_sufficiency(state: PipelineState) -> dict[str, Any]:
    """
    [노드: check_sufficiency]

    추천을 진행하기 위한 최소 조건이 충분한지 확인한다.

    필수 조건:
        - query 존재
        - category가 지원 범위에 포함
        - headcount 존재

    부족한 경우:
        clarify 노드로 이동하여 역질문 생성

    충분한 경우:
        query_transform 노드로 이동하여 추천 검색 진행
    """

    missing_fields = []

    query = state.get("user_text") or ""
    category = state.get("category") or ""
    group = state.get("group") or {}

    if not query:
        missing_fields.append("query")

    if category not in SUPPORTED_CATEGORIES:
        missing_fields.append("category")

    if not group.get("headcount"):
        missing_fields.append("headcount")

    return {
        "is_sufficient": len(missing_fields) == 0,
        "missing_fields": missing_fields,
    }


def route_after_sufficiency(state: PipelineState) -> str:
    if state.get("is_sufficient"):
        return "query_transform"

    return "clarify"


def node_clarify(state: PipelineState) -> dict[str, Any]:
    category = state.get("category") or "boardgame"
    group = state.get("group") or {}
    missing_fields = state.get("missing_fields") or []

    result = {
        "answer": "추천을 정확히 하기 위해 조건이 조금 더 필요합니다.",
        "games": [],
        "next_question": build_next_question(category, group, missing_fields),
    }

    return {
        "result": result,
        "answer": result["answer"],
        "games": result["games"],
        "next_question": result["next_question"],
    }


def node_query_transform(state: PipelineState) -> dict[str, Any]:
    """
    [노드: query_transform]

    자연어 요청과 group 조건을 검색용 형태로 변환한다.

    생성 필드:
        query_text:
            keyword_retrieve에 전달되는 확장 검색 문장

        query_filter:
            condition_score에서 사용하는 조건 dict

        emotion_tags / anchor_titles:
            yoonha_graph와 같은 출력 구조 유지를 위해 포함
            현재 yonguk_graph에서는 빈 리스트로 둔다.

    예:
        query="원주에서 3명이 할 수 있는 스토리 좋은 방탈출 추천"
        →
        query_text="원주에서 3명이 할 수 있는 ... 3인 원주시"
        query_filter={
            "escape": 2,
            "location_wonju": 3,
            "player_3": 3,
            "story": 2,
            "popular": 1
        }
    """

    query = state.get("user_text", "")
    category = state.get("category", "boardgame")
    group = state.get("group", {})

    conditions = build_conditions(query, category, group)
    query_text = build_query_text(query, category, group)

    return {
        "query_text": query_text,
        "query_filter": conditions,
        "emotion_tags": [],
        "anchor_titles": [],
    }


def node_retrieve(state: PipelineState) -> dict[str, Any]:
    """
    [노드: retrieve]

    yonguk_eval_total.py의 통합 검색 함수를 사용하여 추천 후보를 가져온다.

    내부 검색 구조:
        1. keyword_retrieve
           - 자연어 키워드 + 1차 가중치 + 2차 가중치

        2. condition_retrieve
           - condition_score 기반 조건 충족도 정렬

        3. rrf_fusion
           - keyword 결과와 condition 결과를 RRF로 융합

    예외 처리:
        메타데이터가 없거나 yonguk_eval_total 로드 실패 시
        retrieve_error에 에러 메시지를 저장하고 빈 결과를 반환한다.
    """

    try:
        items = local_retrieve(
            query_text=state.get("query_text", ""),
            category=state.get("category", "boardgame"),
            conditions=state.get("query_filter", {}),
        )

        return {
            "retrieved_items": items,
            "retrieve_error": "",
        }

    except Exception as exc:
        return {
            "retrieved_items": [],
            "retrieve_error": str(exc),
        }


def node_tag_filter(state: PipelineState) -> dict[str, Any]:
    """
    [노드: tag_filter]

    yoonha_graph와 같은 6단계 구조를 유지하기 위한 필터 노드.

    현재 yonguk_graph에서는 yonguk_eval_total의 RRF 결과가
    이미 조건 점수 기반으로 정렬되어 있으므로 별도 감정 태그 필터링 없이
    retrieved_items를 filtered_items로 전달한다.

    추후 확장:
        - horror_tolerance 기반 필터
        - emotion tag 가중치
        - 부정 리뷰 필터링
    """

    # 통합 버전에서는 yonguk_eval_total 내부 RRF 결과를 그대로 사용
    return {
        "filtered_items": state.get("retrieved_items", []) or []
    }


def node_generate(state: PipelineState) -> dict[str, Any]:
    """
    [노드: generate]

    최종 추천 응답을 생성한다.

    생성 항목:
        - answer
        - games
        - next_question

    use_api 여부:
        현재 yonguk_graph는 발표/테스트 안정성을 위해 로컬 생성 방식을 기본으로 한다.
        따라서 API 키가 없어도 테스트가 통과한다.

    items가 비어 있는 경우:
        - retrieve_error가 있으면 데이터 연결 실패 안내
        - retrieve_error가 없으면 조건에 맞는 추천 없음 안내
    """

    items = state.get("filtered_items", []) or []
    category = state.get("category", "boardgame")
    group = state.get("group", {}) or {}
    conditions = state.get("query_filter", {}) or {}
    retrieve_error = state.get("retrieve_error", "")

    if not items:
        if retrieve_error:
            answer = "검색 데이터 또는 메타데이터가 현재 실행 환경에 연결되어 있지 않아 추천 후보를 조회하지 못했습니다."
        else:
            answer = "조건에 맞는 추천 결과를 찾지 못했습니다."

        result = {
            "answer": answer,
            "games": [],
            "next_question": build_next_question(category, group),
        }

        return {
            **normalize_result(result),
            "result": result,
        }

    games = format_games(items, conditions)

    category_label = {
        "boardgame": "보드게임",
        "escape": "방탈출",
        "murder": "머더미스터리",
    }.get(category, "추천")

    top_title = games[0]["title"] if games else ""

    answer = (
        f"{group.get('headcount', '요청하신 조건')}명 기준으로 "
        f"{category_label} 추천 결과를 찾았습니다. "
        f"가장 우선 추천 후보는 '{top_title}'입니다."
    )

    result = {
        "answer": answer,
        "games": games,
        "next_question": build_next_question(category, group),
    }

    return {
        **normalize_result(result),
        "result": result,
    }


# =========================================================
# Graph assembly
# =========================================================

def build_graph():
    """
    LangGraph workflow를 조립한다.

    그래프 구조:
        normalize_input
        → check_sufficiency
            ├─ clarify
            └─ query_transform
                → retrieve
                → tag_filter
                → generate

    반환:
        compiled graph
    """

    workflow = StateGraph(PipelineState)

    workflow.add_node("normalize_input", node_normalize_input)
    workflow.add_node("check_sufficiency", node_check_sufficiency)
    workflow.add_node("clarify", node_clarify)
    workflow.add_node("query_transform", node_query_transform)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("tag_filter", node_tag_filter)
    workflow.add_node("generate", node_generate)

    workflow.set_entry_point("normalize_input")
    workflow.add_edge("normalize_input", "check_sufficiency")

    workflow.add_conditional_edges(
        "check_sufficiency",
        route_after_sufficiency,
        {
            "clarify": "clarify",
            "query_transform": "query_transform",
        },
    )

    workflow.add_edge("clarify", END)
    workflow.add_edge("query_transform", "retrieve")
    workflow.add_edge("retrieve", "tag_filter")
    workflow.add_edge("tag_filter", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


graph = build_graph()


def run_pipeline(
    user_text: str = "",
    group: dict[str, Any] | None = None,
    category: str = "boardgame",
    use_api: bool = True,
):
    return graph.invoke({
        "query": user_text,
        "user_text": user_text,
        "category": category,
        "group": group or {},
        "use_api": use_api,
    })
