"""
yonguk_bbabang_retriever.py
- 빠방/방탈출(escape) 전용 Retriever
- FAISS + metadata 기반 BM25 / Dense / Hybrid 검색 지원

주의:
1) 아래 경로는 프로젝트 위치에 맞게 자동 탐색합니다.
2) faiss index / metadata 파일명이 다르면 DEFAULT_PATHS만 수정하세요.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PATHS = {
    "index": [
        PROJECT_ROOT / "faiss_bbabang_stats.index",
        PROJECT_ROOT / "faiss_escape_stats.index",
        PROJECT_ROOT / "data" / "faiss_bbabang_stats.index",
        PROJECT_ROOT / "data" / "faiss_escape_stats.index",
    ],
    "metadata": [
        PROJECT_ROOT / "faiss_bbabang_stats_metadata.json",
        PROJECT_ROOT / "faiss_bbabang_stats_meta.json",
        PROJECT_ROOT / "faiss_escape_stats_metadata.json",
        PROJECT_ROOT / "data" / "faiss_bbabang_stats_metadata.json",
        PROJECT_ROOT / "data" / "faiss_escape_stats_metadata.json",
    ],
    "bm25": [
        PROJECT_ROOT / "faiss_bbabang_stats_bm25.pkl",
        PROJECT_ROOT / "faiss_escape_stats_bm25.pkl",
        PROJECT_ROOT / "data" / "faiss_bbabang_stats_bm25.pkl",
    ],
}


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    return _normalize_text(text).split()


def _get_doc_text(meta: Dict[str, Any]) -> str:
    keys = [
        "text", "content", "description", "desc", "summary", "review",
        "title", "name", "theme", "store", "genre", "difficulty",
    ]
    parts: List[str] = []
    for key in keys:
        value = meta.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            parts.extend(map(str, value))
        else:
            parts.append(str(value))
    if not parts:
        parts = [json.dumps(meta, ensure_ascii=False)]
    return " ".join(parts)


@dataclass
class SearchResult:
    rank: int
    score: float
    metadata: Dict[str, Any]


class SimpleBM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.tokens = [_tokenize(doc) for doc in docs]
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(t) for t in self.tokens) / max(len(self.tokens), 1)
        self.df: Dict[str, int] = {}
        for toks in self.tokens:
            for tok in set(toks):
                self.df[tok] = self.df.get(tok, 0) + 1
        self.n = len(self.tokens)

    def get_scores(self, query: str) -> np.ndarray:
        q_tokens = _tokenize(query)
        scores = np.zeros(self.n, dtype=np.float32)
        for i, doc_tokens in enumerate(self.tokens):
            if not doc_tokens:
                continue
            freqs: Dict[str, int] = {}
            for tok in doc_tokens:
                freqs[tok] = freqs.get(tok, 0) + 1
            dl = len(doc_tokens)
            for tok in q_tokens:
                if tok not in freqs:
                    continue
                df = self.df.get(tok, 0)
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                tf = freqs[tok]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[i] += idf * (tf * (self.k1 + 1) / denom)
        return scores


class BbabangRetriever:
    def __init__(
        self,
        index_path: Optional[str | Path] = None,
        metadata_path: Optional[str | Path] = None,
        bm25_path: Optional[str | Path] = None,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.index_path = Path(index_path) if index_path else _first_existing(DEFAULT_PATHS["index"])
        self.metadata_path = Path(metadata_path) if metadata_path else _first_existing(DEFAULT_PATHS["metadata"])
        self.bm25_path = Path(bm25_path) if bm25_path else _first_existing(DEFAULT_PATHS["bm25"])
        self.model_name = model_name

        if self.metadata_path is None:
            raise FileNotFoundError(
                "빠방 metadata 파일을 찾지 못했습니다. "
                "faiss_bbabang_stats_metadata.json 또는 경로를 확인하세요."
            )

        self.metadata = self._load_metadata(self.metadata_path)
        self.docs = [_get_doc_text(m) for m in self.metadata]
        self.bm25 = self._load_or_build_bm25()
        self.index = self._load_faiss_index()
        self.model = None

    def _load_metadata(self, path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ["items", "data", "metadata", "documents"]:
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError("metadata는 list 형태이거나 list를 포함한 dict여야 합니다.")
        return [x if isinstance(x, dict) else {"text": str(x)} for x in data]

    def _load_or_build_bm25(self) -> SimpleBM25:
        if self.bm25_path and self.bm25_path.exists():
            try:
                with self.bm25_path.open("rb") as f:
                    obj = pickle.load(f)
                if hasattr(obj, "get_scores"):
                    return obj
            except Exception:
                pass
        return SimpleBM25(self.docs)

    def _load_faiss_index(self):
        if self.index_path is None or not self.index_path.exists():
            return None
        if faiss is None:
            return None
        return faiss.read_index(str(self.index_path))

    def _ensure_model(self):
        if self.model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers가 설치되어 있지 않습니다. pip install sentence-transformers")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def get_embedding(self, text: str) -> np.ndarray:
        model = self._ensure_model()
        emb = model.encode([text], normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    def retrieve_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        scores = np.asarray(self.bm25.get_scores(query), dtype=np.float32)
        if len(scores) == 0:
            return []
        idxs = np.argsort(scores)[::-1][:top_k]
        return [self._format_result(i, float(scores[i]), rank + 1) for rank, i in enumerate(idxs)]

    def retrieve_dense(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            return []
        q = self.get_embedding(query)
        scores, idxs = self.index.search(q, top_k)
        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(idxs[0], scores[0]), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append(self._format_result(int(idx), float(score), rank))
        return results

    def retrieve_vanilla(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retrieve_dense(query, top_k=top_k) or self.retrieve_bm25(query, top_k=top_k)

    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.65) -> List[Dict[str, Any]]:
        bm25_scores = np.asarray(self.bm25.get_scores(query), dtype=np.float32)
        bm25_norm = self._minmax(bm25_scores)

        dense_map: Dict[int, float] = {}
        if self.index is not None:
            q = self.get_embedding(query)
            scores, idxs = self.index.search(q, min(max(top_k * 5, 20), len(self.metadata)))
            dense_raw = np.asarray(scores[0], dtype=np.float32)
            dense_norm = self._minmax(dense_raw)
            for idx, score in zip(idxs[0], dense_norm):
                if 0 <= idx < len(self.metadata):
                    dense_map[int(idx)] = float(score)

        final_scores: List[Tuple[int, float]] = []
        for i in range(len(self.metadata)):
            dense_score = dense_map.get(i, 0.0)
            bm25_score = float(bm25_norm[i]) if i < len(bm25_norm) else 0.0
            score = alpha * dense_score + (1 - alpha) * bm25_score
            if score > 0:
                final_scores.append((i, score))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        return [self._format_result(i, score, rank + 1) for rank, (i, score) in enumerate(final_scores[:top_k])]

    def _format_result(self, idx: int, score: float, rank: int) -> Dict[str, Any]:
        meta = dict(self.metadata[idx])
        return {
            "rank": rank,
            "score": score,
            "category": "escape",
            "source": "bbabang",
            "title": meta.get("title") or meta.get("name") or meta.get("theme") or f"escape-{idx}",
            "metadata": meta,
            "text": _get_doc_text(meta),
        }

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        mn, mx = float(np.min(arr)), float(np.max(arr))
        if abs(mx - mn) < 1e-9:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - mn) / (mx - mn)


_RETRIEVER: Optional[BbabangRetriever] = None


def get_retriever() -> BbabangRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = BbabangRetriever()
    return _RETRIEVER


def retrieve(query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
    return get_retriever().retrieve(query, top_k=top_k, **kwargs)


def retrieve_bm25(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return get_retriever().retrieve_bm25(query, top_k=top_k)


def retrieve_dense(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return get_retriever().retrieve_dense(query, top_k=top_k)


def retrieve_vanilla(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return get_retriever().retrieve_vanilla(query, top_k=top_k)


def get_embedding(text: str) -> np.ndarray:
    return get_retriever().get_embedding(text)


if __name__ == "__main__":
    q = "강남에서 무섭고 난이도 있는 방탈출 추천"
    for row in retrieve(q, top_k=5):
        print(row["rank"], row["score"], row["title"])
