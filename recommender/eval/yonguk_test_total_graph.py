"""
total_test_graph.py

통합 추천 Graph 테스트 파일
- boardgame / escape / murder 통합 테스트
"""

import sys
import time
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent)
)

from recommender.yonguk_total_graph import graph, run_pipeline


# =========================================================
# 유틸
# =========================================================

def print_header(text):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def print_subheader(text):
    print(f"\n--- {text} ---")


# =========================================================
# 1. Boardgame 테스트
# =========================================================

def test_boardgame_graph():
    print_header("1. 보드게임 추천 테스트")

    payload = {
        "query": "4명이서 할 전략 보드게임 추천해줘",
        "category": "boardgame",
        "use_api": False,
    }

    print_subheader("입력")
    print(payload)

    start = time.time()
    result = graph.invoke(payload)
    elapsed = time.time() - start

    print_subheader("출력")
    print(result)

    assert isinstance(result, dict)
    assert "answer" in result
    assert "games" in result
    assert "next_question" in result

    print(f"\n✅ 보드게임 테스트 통과 ({elapsed:.2f}s)")


# =========================================================
# 2. Escape 테스트
# =========================================================

def test_escape_graph():
    print_header("2. 방탈출 추천 테스트")

    payload = {
        "query": "원주에서 3명이 할 수 있는 스토리 좋은 방탈출 추천",
        "category": "escape",
        "use_api": False,
    }

    print_subheader("입력")
    print(payload)

    start = time.time()
    result = graph.invoke(payload)
    elapsed = time.time() - start

    print_subheader("출력")
    print(result)

    assert isinstance(result, dict)
    assert "answer" in result
    assert "games" in result
    assert "next_question" in result

    print(f"\n✅ 방탈출 테스트 통과 ({elapsed:.2f}s)")


# =========================================================
# 3. Murder Mystery 테스트
# =========================================================

def test_murder_graph():
    print_header("3. 머더미스터리 추천 테스트")

    payload = {
        "query": "6명이서 할 쉬운 머더미스터리 추천",
        "category": "murder",
        "use_api": False,
    }

    print_subheader("입력")
    print(payload)

    start = time.time()
    result = graph.invoke(payload)
    elapsed = time.time() - start

    print_subheader("출력")
    print(result)

    assert isinstance(result, dict)
    assert "answer" in result
    assert "games" in result
    assert "next_question" in result

    print(f"\n✅ 머더미스터리 테스트 통과 ({elapsed:.2f}s)")


# =========================================================
# 4. run_pipeline 호환성 테스트
# =========================================================

def test_run_pipeline():
    print_header("4. run_pipeline 호환성 테스트")

    start = time.time()

    result = run_pipeline(
        user_text="4명이서 할 전략 보드게임 추천해줘",
        group={
            "headcount": 4,
            "weight_pref": "heavy",
            "play_time": 120,
            "relation": "friend",
        },
        category="boardgame",
        use_api=False,
    )

    elapsed = time.time() - start

    print_subheader("출력")
    print(result)

    assert isinstance(result, dict)
    assert "answer" in result
    assert "games" in result
    assert "next_question" in result

    print(f"\n✅ run_pipeline 테스트 통과 ({elapsed:.2f}s)")


# =========================================================
# 5. Clarifying Question 테스트
# =========================================================

def test_clarifying_question():
    print_header("5. Clarifying Question 테스트")

    payload = {
        "query": "보드게임 추천해줘",
        "category": "boardgame",
        "use_api": False,
    }

    print_subheader("입력")
    print(payload)

    start = time.time()
    result = graph.invoke(payload)
    elapsed = time.time() - start

    print_subheader("출력")
    print(result)

    assert isinstance(result, dict)
    assert "next_question" in result
    assert result["next_question"] != ""

    print(f"\n✅ Clarifying Question 테스트 통과 ({elapsed:.2f}s)")


# =========================================================
# 메인 실행
# =========================================================

if __name__ == "__main__":
    print("\n🚀 total_test_graph 시작\n")

    total_start = time.time()

    test_boardgame_graph()
    test_escape_graph()
    test_murder_graph()
    test_run_pipeline()
    test_clarifying_question()

    total_elapsed = time.time() - total_start

    print(
        f"\n🎉 보드게임 / 방탈출 / 머더미스터리 "
        f"통합 테스트 완료 ({total_elapsed:.2f}s)"
    )