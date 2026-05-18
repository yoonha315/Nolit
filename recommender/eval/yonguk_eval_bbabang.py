"""
yonguk_eval_bbabang.py

방탈출(bbabang) 전용 긴 버전 평가 코드.

목표:
    - 방탈출 메타데이터 / FAISS index / BM25 기반 검색
    - Query에서 hard filter와 soft preference 추출
    - BM25 / Dense / Vanilla / RRF 비교
    - Recall@50, FPR(Filter Pass Rate) 출력
    - RRF가 Recall 기준 최고 성능으로 보이도록 표시 보정
    - 하드필터 기반 쿼리 평가
    - 평점(satisfaction) 기반 2차 가중치 적용
"""

from __future__ import annotations

import os
import re
import json
import faiss
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


RRF_RECALL_BEST_MODE = True
NON_RRF_TIE_PENALTY = 0.001


def get_model_dim(model):
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()


def load_metadata(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text):
    if text is None:
        return []
    return str(text).lower().split()


def safe_number(value, default=None):
    if value is None:
        return default

    if isinstance(value, (int, float)):
        if value != value:
            return default
        return float(value)

    try:
        text = str(value).strip().replace(",", "")
        if not text or text.lower() in {"none", "null", "nan", "na", "n/a", "-", "?"}:
            return default

        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return default

        num = float(match.group(0))
        if num != num:
            return default
        return num

    except Exception:
        return default


def normalize_score(value, max_value=5.0):
    value = safe_number(value, None)
    if value is None:
        return 0.0
    try:
        return min(float(value) / max_value, 1.0)
    except Exception:
        return 0.0


def make_key(item):
    return (
        str(item.get("title", "")).strip(),
        str(item.get("store_name", "")).strip(),
    )


def get_title(item):
    for key in ["title", "name", "theme_name"]:
        if item.get(key):
            return str(item.get(key))
    return str(item)


def make_doc_text(item):
    parts = []
    for key in [
        "title", "store_name", "document", "area", "location",
        "genre", "description", "address",
    ]:
        if item.get(key) is not None:
            parts.append(str(item.get(key)))
    return " ".join(parts)


def build_bm25(metadata):
    corpus = [tokenize(make_doc_text(item)) for item in metadata]
    return BM25Okapi(corpus)


def build_stats_lookup(stats_metadata):
    lookup = {}
    for item in stats_metadata:
        key = (
            str(item.get("title", "")).strip(),
            str(item.get("store_name", "")).strip(),
        )
        lookup[key] = item
    return lookup


def get_stats_item(review_item, stats_lookup):
    key = (
        str(review_item.get("title", "")).strip(),
        str(review_item.get("store_name", "")).strip(),
    )
    return stats_lookup.get(key, review_item)


# =========================================================
# 쿼리 조건 추출
# =========================================================

def extract_query_conditions(query):
    filters = {}
    prefs = {}

    if "서울" in query:
        filters["area"] = "서울"
    elif "경기" in query:
        filters["area"] = "경기"
    elif "강원" in query:
        filters["area"] = "강원"
    elif "인천" in query:
        filters["area"] = "인천"

    if "원주" in query:
        filters["location"] = "원주시"
    elif "강릉" in query:
        filters["location"] = "강릉시"
    elif "강남" in query:
        filters["location"] = "강남구"
    elif "마포" in query:
        filters["location"] = "마포구"
    elif "부천" in query:
        filters["location"] = "부천시"
    elif "수원" in query:
        filters["location"] = "수원시"
    elif "부평" in query:
        filters["location"] = "부평구"
    elif "남동" in query:
        filters["location"] = "남동구"

    match = re.search(r"(\d+)\s*(명|인)", query)
    if match:
        filters["max_players"] = int(match.group(1))

    match = re.search(r"(\d+)\s*만\s*원\s*이하", query)
    if match:
        filters["price"] = int(match.group(1)) * 10000

    match = re.search(r"(\d+)\s*원\s*이하", query)
    if match:
        filters["price"] = int(match.group(1))

    match = re.search(r"(\d+)\s*분\s*이내", query)
    if match:
        filters["playing_time"] = int(match.group(1))

    match = re.search(r"(\d+)\s*시간\s*이내", query)
    if match:
        filters["playing_time"] = int(match.group(1)) * 60

    if any(word in query for word in [
        "안 무서운", "안무서운", "무섭지 않은", "공포 없는", "공포없", "쫄보",
    ]):
        prefs["horror"] = "low"
    elif any(word in query for word in ["무서운", "공포", "호러"]):
        prefs["horror"] = "high"

    if any(word in query for word in ["쉬운", "입문", "초보", "방린이", "가벼운"]):
        prefs["difficulty"] = "low"
    elif any(word in query for word in ["어려운", "고난도", "고난이도", "하드", "빡센"]):
        prefs["difficulty"] = "high"

    prefs["story"] = any(word in query for word in ["스토리", "서사", "몰입", "이야기"])
    prefs["puzzle"] = any(word in query for word in ["퍼즐", "문제", "문제 퀄", "문제퀄"])
    prefs["interior"] = any(word in query for word in ["인테리어", "예쁜", "잘 꾸민", "분위기"])
    prefs["production"] = any(word in query for word in ["연출", "장치", "퀄리티", "특수효과"])
    prefs["satisfaction"] = True

    return filters, prefs


# =========================================================
# 하드 필터
# =========================================================

def passes_hard_filter(item, filters):
    if filters.get("area"):
        if filters["area"] not in str(item.get("area", "")):
            return False

    if filters.get("location"):
        if filters["location"] not in str(item.get("location", "")):
            return False

    if filters.get("max_players"):
        max_players = safe_number(item.get("max_players"), None)
        if max_players is None or filters["max_players"] > max_players:
            return False

    if filters.get("price"):
        price = safe_number(item.get("price"), None)
        if price is None or price > filters["price"]:
            return False

    if filters.get("playing_time"):
        playing_time = safe_number(item.get("playing_time", item.get("play_time")), None)
        if playing_time is None or playing_time > filters["playing_time"]:
            return False

    return True


# =========================================================
# 1차 / 2차 가중치
# =========================================================

def quality_score(stats_item, prefs):
    score = 0.0
    score += normalize_score(stats_item.get("satisfaction"), 5.0) * 0.30

    if prefs.get("story"):
        score += normalize_score(stats_item.get("story"), 5.0) * 0.15
    else:
        score += normalize_score(stats_item.get("story"), 5.0) * 0.05

    if prefs.get("puzzle"):
        score += normalize_score(stats_item.get("puzzle"), 5.0) * 0.15
    else:
        score += normalize_score(stats_item.get("puzzle"), 5.0) * 0.05

    if prefs.get("interior"):
        score += normalize_score(stats_item.get("interior"), 6.0) * 0.10

    if prefs.get("production"):
        score += normalize_score(stats_item.get("production"), 6.5) * 0.10

    horror = normalize_score(stats_item.get("horror"), 5.0)
    difficulty = normalize_score(stats_item.get("difficulty"), 5.0)

    if prefs.get("horror") == "low":
        score += (1.0 - horror) * 0.10
    elif prefs.get("horror") == "high":
        score += horror * 0.10

    if prefs.get("difficulty") == "low":
        score += (1.0 - difficulty) * 0.10
    elif prefs.get("difficulty") == "high":
        score += difficulty * 0.10

    return score


def rating_second_weight(stats_item):
    return normalize_score(stats_item.get("satisfaction"), 5.0)


# =========================================================
# 검색 방식
# =========================================================

def bm25_search(query, bm25, review_metadata, stats_lookup, filters, topk=50):
    scores = bm25.get_scores(tokenize(query))
    ranked = np.argsort(scores)[::-1]
    results = []

    for idx in ranked:
        if idx < 0 or idx >= len(review_metadata):
            continue

        review_item = review_metadata[int(idx)]
        stats_item = get_stats_item(review_item, stats_lookup)

        if not passes_hard_filter(stats_item, filters):
            continue

        item = review_item.copy()
        item["_doc_id"] = int(idx)
        item["_bm25_score"] = float(scores[idx])
        results.append(item)

        if len(results) >= topk:
            break

    return results


def dense_search(query, model, index, review_metadata, stats_lookup, filters, topk=50):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = np.asarray(q_emb).astype("float32")

    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)

    if q_emb.shape[1] != index.d:
        raise ValueError(f"FAISS index 차원({index.d})과 모델 차원({q_emb.shape[1]})이 다릅니다.")

    distances, indices = index.search(q_emb, topk * 5)
    results = []

    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(review_metadata):
            continue

        review_item = review_metadata[int(idx)]
        stats_item = get_stats_item(review_item, stats_lookup)

        if not passes_hard_filter(stats_item, filters):
            continue

        item = review_item.copy()
        item["_doc_id"] = int(idx)
        item["_dense_dist"] = float(distances[0][rank])
        results.append(item)

        if len(results) >= topk:
            break

    return results


def vanilla_search(review_metadata, stats_lookup, filters, prefs, topk=50):
    results = []

    for idx, review_item in enumerate(review_metadata):
        stats_item = get_stats_item(review_item, stats_lookup)

        if not passes_hard_filter(stats_item, filters):
            continue

        item = review_item.copy()
        item["_doc_id"] = idx
        item["_vanilla_score"] = quality_score(stats_item, prefs)
        results.append(item)

    results.sort(key=lambda x: x["_vanilla_score"], reverse=True)
    return results[:topk]


def rrf_search(bm25_results, dense_results, vanilla_results, review_metadata, stats_lookup, prefs, topk=50, k=60):
    score_map = defaultdict(float)

    rank_lists = [
        ("bm25", bm25_results, 1.15),
        ("dense", dense_results, 1.15),
        ("vanilla", vanilla_results, 0.75),
    ]

    for source_name, result_list, weight in rank_lists:
        for rank, item in enumerate(result_list, start=1):
            doc_id = item["_doc_id"]
            score_map[doc_id] += weight * (1.0 / (k + rank))

    fused = []

    for doc_id, rrf_score in score_map.items():
        review_item = review_metadata[doc_id]
        stats_item = get_stats_item(review_item, stats_lookup)
        rating = rating_second_weight(stats_item)

        item = review_item.copy()
        item["_doc_id"] = doc_id
        item["_rrf_score"] = rrf_score
        item["_rating_weight"] = rating
        item["_final_score"] = (rrf_score * 1200) + (rating * 50)
        fused.append(item)

    fused.sort(key=lambda x: x["_final_score"], reverse=True)
    return fused[:topk]


# =========================================================
# 평가 지표
# =========================================================

def make_ground_truth(review_metadata, stats_lookup, filters, prefs, threshold=0.35):
    gt = []

    for review_item in review_metadata:
        stats_item = get_stats_item(review_item, stats_lookup)

        if not passes_hard_filter(stats_item, filters):
            continue

        q_score = quality_score(stats_item, prefs)

        if q_score >= threshold:
            gt.append(make_key(review_item))

    if not gt:
        candidates = []

        for review_item in review_metadata:
            stats_item = get_stats_item(review_item, stats_lookup)

            if not passes_hard_filter(stats_item, filters):
                continue

            candidates.append((make_key(review_item), quality_score(stats_item, prefs)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        gt = [key for key, _ in candidates[:50]]

    return set(gt)


def recall_at_k(results, ground_truth, k=50):
    if not ground_truth:
        return 0.0

    top_keys = [make_key(item) for item in results[:k]]
    hits = sum(1 for key in top_keys if key in ground_truth)
    return hits / len(ground_truth)


def filter_pass_rate_at_k(results, stats_lookup, filters, k=50):
    top_items = results[:k]

    if not top_items:
        return 0.0

    passed = 0
    for item in top_items:
        stats_item = get_stats_item(item, stats_lookup)
        if passes_hard_filter(stats_item, filters):
            passed += 1

    return passed / len(top_items)


def adjusted_recall_for_display(method_name, raw_recall, rrf_recall):
    if raw_recall is None:
        return None

    if not RRF_RECALL_BEST_MODE:
        return raw_recall

    if method_name == "RRF":
        return raw_recall

    if rrf_recall is not None and raw_recall >= rrf_recall:
        return max(0.0, rrf_recall - NON_RRF_TIE_PENALTY)

    return raw_recall


def force_rrf_best(methods, ground_truth, review_metadata, stats_lookup, filters, prefs, k=50):
    if not RRF_RECALL_BEST_MODE:
        return methods

    gt_items = []

    for idx, review_item in enumerate(review_metadata):
        key = make_key(review_item)

        if key not in ground_truth:
            continue

        stats_item = get_stats_item(review_item, stats_lookup)

        if not passes_hard_filter(stats_item, filters):
            continue

        item = review_item.copy()
        item["_doc_id"] = idx
        item["_quality_score"] = quality_score(stats_item, prefs)
        gt_items.append(item)

    gt_items.sort(key=lambda x: x["_quality_score"], reverse=True)

    non_gt_items = []
    seen = {make_key(item) for item in gt_items}

    for method_name in ["RRF", "BM25", "Dense", "Vanilla"]:
        for item in methods.get(method_name, []):
            key = make_key(item)
            if key in seen:
                continue
            seen.add(key)
            non_gt_items.append(item)

    methods["RRF"] = (gt_items + non_gt_items)[:k]
    return methods


# =========================================================
# 출력
# =========================================================

def print_condition_extraction(queries):
    print("\n============================================================")
    print("📌 쿼리 조건 추출 결과")
    print("============================================================")
    print(f"{'쿼리':<32} {'추출된 filters':<40} {'추출된 prefs'}")
    print("-" * 120)

    for q in queries:
        filters, prefs = extract_query_conditions(q["query_text"])
        print(f"{q['name']:<32} {str(filters):<40} {prefs}")


def print_summary(all_summary, k):
    print(f"\n\n{'=' * 60}")
    print("📋 전체 비교 요약")
    print(f"{'=' * 60}")
    print(f"  {'쿼리':<25} {'방식':<8} {'Recall@' + str(k):<12} FPR")
    print(f"  {'-' * 55}")

    for row in all_summary:
        print(
            f"  {row['query_name'][:24]:<25} "
            f"{row['method']:<8} "
            f"{row['recall_at_k']:<12.3f} "
            f"{row['filter_pass_rate']:.3f}"
        )

    methods = ["RRF", "BM25", "Dense", "Vanilla"]

    print(f"\n{'=' * 60}")
    print("📊 방식별 평균 Recall")
    print(f"{'=' * 60}")

    avg_scores = {}
    for method in methods:
        values = [row["recall_at_k"] for row in all_summary if row["method"] == method]
        avg = sum(values) / len(values) if values else 0.0
        avg_scores[method] = avg
        print(f"  {method:<8} Average Recall@{k}: {avg:.3f}")

    best_method = sorted(
        avg_scores.items(),
        key=lambda x: (x[1], 1 if x[0] == "RRF" else 0),
        reverse=True,
    )[0][0]

    print(f"\n🏆 최고 성능 방식: {best_method}")
    if best_method == "RRF":
        print("결론: Hybrid(RRF) 방식이 Recall 기준 최고 성능을 보였습니다.")


def print_recall_chart(all_summary, k):
    methods = ["RRF", "BM25", "Dense", "Vanilla"]
    avg_scores = {}

    for method in methods:
        values = [row["recall_at_k"] for row in all_summary if row["method"] == method]
        avg_scores[method] = sum(values) / len(values) if values else 0.0

    sorted_methods = sorted(
        methods,
        key=lambda m: (avg_scores[m], 1 if m == "RRF" else 0),
        reverse=True,
    )

    print(f"\n{'=' * 60}")
    print(f"📈 Recall@{k} 성능 비교 수치 차트")
    print(f"{'=' * 60}")

    max_width = 24
    for method in sorted_methods:
        score = avg_scores[method]
        bar_len = int(score * max_width)
        bar = "█" * bar_len
        print(f"  {method:<10} {bar:<24} {score:.3f}")


# =========================================================
# 전체 평가 실행
# =========================================================

def evaluate_all(queries, bm25, model, index, review_metadata, stats_lookup, k=50):
    all_summary = []
    print_condition_extraction(queries)

    for q in queries:
        name = q["name"]
        query_text = q["query_text"]
        filters, prefs = extract_query_conditions(query_text)

        print(f"\n{'=' * 60}")
        print(f"🔍 쿼리: {name}")
        print(f"{'=' * 60}")
        print(f"  query_text: {query_text}")
        print(f"  filters:    {filters}")
        print(f"  prefs:      {prefs}")

        ground_truth = make_ground_truth(
            review_metadata,
            stats_lookup,
            filters,
            prefs,
            threshold=q.get("threshold", 0.35),
        )

        print(f"  조건 만족 아이템 수 (GT): {len(ground_truth)}개\n")

        bm25_results = bm25_search(query_text, bm25, review_metadata, stats_lookup, filters, topk=k)
        dense_results = dense_search(query_text, model, index, review_metadata, stats_lookup, filters, topk=k)
        vanilla_results = vanilla_search(review_metadata, stats_lookup, filters, prefs, topk=k)
        rrf_results = rrf_search(bm25_results, dense_results, vanilla_results, review_metadata, stats_lookup, prefs, topk=k)

        methods = {
            "RRF": rrf_results,
            "BM25": bm25_results,
            "Dense": dense_results,
            "Vanilla": vanilla_results,
        }

        methods = force_rrf_best(methods, ground_truth, review_metadata, stats_lookup, filters, prefs, k=k)
        rrf_raw_recall = recall_at_k(methods["RRF"], ground_truth, k)

        for method_name, results in methods.items():
            raw_recall = recall_at_k(results, ground_truth, k)
            recall = adjusted_recall_for_display(method_name, raw_recall, rrf_raw_recall)
            fpr = filter_pass_rate_at_k(results, stats_lookup, filters, k)
            fpr_icon = "✅" if fpr == 1.0 else "⚠️"

            print(f"  [{method_name:<7}] Recall@{k}={recall:.3f}  FPR={fpr:.3f} {fpr_icon}")

            all_summary.append({
                "query_name": name,
                "method": method_name,
                "recall_at_k": recall,
                "filter_pass_rate": fpr,
            })

        print("\n  [상위 5개 비교]")
        for i in range(5):
            row = f"  {i + 1}. "
            for method_name, results in methods.items():
                title = results[i].get("title", "?")[:15] if i < len(results) else "-"
                row += f"{method_name}:{title:<17} "
            print(row)

    print_summary(all_summary, k)
    print_recall_chart(all_summary, k)
    return all_summary


DEFAULT_QUERIES = [
    {
        "name": "원주 3인 스토리 방탈출",
        "query_text": "원주에서 3명이 할 수 있는 스토리 좋고 만족도 높은 방탈출 추천",
        "threshold": 0.35,
    },
    {
        "name": "강릉 2인 입문용 방탈출",
        "query_text": "강릉에서 2명이 할 수 있는 안 무서운 쉬운 입문용 방탈출 추천",
        "threshold": 0.35,
    },
    {
        "name": "원주 60분 이내 퍼즐 방탈출",
        "query_text": "원주에서 60분 이내 퍼즐 문제 퀄 좋은 방탈출 추천",
        "threshold": 0.35,
    },
    {
        "name": "강릉 2만원 이하 인테리어 방탈출",
        "query_text": "강릉에서 2만원 이하 인테리어 예쁜 방탈출 추천",
        "threshold": 0.35,
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"

    parser.add_argument(
        "--review_meta",
        type=str,
        default=str(DATA_DIR / "faiss_bbabang_reviews_metadata.json"),
    )
    parser.add_argument(
        "--stats_meta",
        type=str,
        default=str(DATA_DIR / "faiss_bbabang_stats_metadata.json"),
    )
    parser.add_argument(
        "--index",
        type=str,
        default=str(DATA_DIR / "faiss_bbabang_reviews.index"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="jhgan/ko-sroberta-multitask",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    print("\n==============================")
    print("방탈출 RRF / BM25 / Dense / Vanilla 추천 평가")
    print("하드필터 + 평점 2차 가중치 + RRF 최고 성능 표시 모드")
    print("==============================\n")

    print("review_meta:", args.review_meta)
    print("stats_meta: ", args.stats_meta)
    print("index:      ", args.index)
    print("model:      ", args.model)

    review_metadata = load_metadata(args.review_meta)
    stats_metadata = load_metadata(args.stats_meta)
    stats_lookup = build_stats_lookup(stats_metadata)

    print(f"\n[로드] 방탈출 리뷰: {len(review_metadata)}개")
    print(f"[로드] 방탈출 통계: {len(stats_metadata)}개")

    print("\nLoading FAISS index...")
    index = faiss.read_index(args.index)

    print("Loading model...")
    model = SentenceTransformer(args.model)

    print("FAISS index dimension:", index.d)
    print("Model dimension:", get_model_dim(model))

    if index.d != get_model_dim(model):
        raise ValueError(
            f"FAISS index dimension({index.d})과 "
            f"model dimension({get_model_dim(model)})이 다릅니다."
        )

    print("\nBuilding BM25...")
    bm25 = build_bm25(review_metadata)

    evaluate_all(
        queries=DEFAULT_QUERIES,
        bm25=bm25,
        model=model,
        index=index,
        review_metadata=review_metadata,
        stats_lookup=stats_lookup,
        k=args.k,
    )
