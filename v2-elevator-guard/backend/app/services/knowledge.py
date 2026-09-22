"""电梯安全知识库检索。

默认用内置的 BM25 关键词检索，零依赖、开箱可用；
如果环境里装了 chromadb 且配了 embedding key，会自动改用向量检索。
两者不可用时也不会让流程失败，只是在报告里标注"未检索到依据"。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..config import settings

META_PATTERN = re.compile(r"<!--\s*(?P<body>.*?)\s*-->", re.DOTALL)
COLLECTION = "elevator_safety"


@dataclass
class Chunk:
    entry_id: str
    title: str
    text: str
    source: str
    action: str = ""
    risk_level: str = ""

    def metadata(self) -> dict:
        return {
            "title": self.title,
            "source": self.source,
            "action": self.action,
            "risk_level": self.risk_level,
        }


@dataclass
class Hit:
    entry_id: str
    title: str
    source: str
    text: str
    score: float
    risk_level: str = ""
    action: str = ""

    def snippet(self, limit: int = 240) -> str:
        body = re.sub(r"\s+", " ", self.text).strip()
        body = re.sub(r"\*\*(.+?)\*\*", r"\1", body)
        return body if len(body) <= limit else body[: limit - 1] + "…"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())


def load_chunks(directory: Path | None = None) -> list[Chunk]:
    base = directory or settings.knowledge_dir
    chunks: list[Chunk] = []
    if not base.exists():
        return chunks
    for path in sorted(base.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for block in re.split(r"^##\s+", text, flags=re.MULTILINE)[1:]:
            lines = block.splitlines()
            if not lines:
                continue
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            meta: dict[str, str] = {}
            match = META_PATTERN.search(body)
            if match:
                for part in match.group("body").split("|"):
                    if ":" in part:
                        key, value = part.split(":", 1)
                        meta[key.strip()] = value.strip()
                body = META_PATTERN.sub("", body).strip()
            if not body:
                continue
            slug = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "-", title).strip("-")
            chunks.append(
                Chunk(
                    entry_id=f"{path.stem}:{slug}",
                    title=title,
                    text=body,
                    source=meta.get("source", path.stem),
                    action=meta.get("action", ""),
                    risk_level=meta.get("risk", ""),
                )
            )
    return chunks


class KnowledgeBase:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._collection = None
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._lengths: list[int] = []
        self.backend = "keyword"
        self.detail = ""

    # ---- 基础 -----------------------------------------------------------
    @property
    def chunks(self) -> list[Chunk]:
        if not self._chunks:
            self._chunks = load_chunks()
        return self._chunks

    def count(self) -> int:
        return len(self.chunks)

    def _build_keyword_index(self) -> None:
        self._tf = []
        self._df = Counter()
        self._lengths = []
        for chunk in self.chunks:
            tokens = _tokens(chunk.title + " " + chunk.text)
            counter = Counter(tokens)
            self._tf.append(counter)
            self._lengths.append(len(tokens) or 1)
            for token in set(tokens):
                self._df[token] += 1

    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        if not settings.embedding_api_key:
            return None
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_base_url,
                timeout=settings.llm_timeout,
                max_retries=1,
            )
            response = client.embeddings.create(
                model=settings.embedding_model, input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as exc:  # noqa: BLE001
            self.detail = f"embedding 调用失败，改用关键词检索（{type(exc).__name__}）"
            return None

    def _connect_chroma(self) -> bool:
        if self._collection is not None:
            return True
        try:
            import chromadb

            settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(settings.chroma_dir))
            self._collection = client.get_or_create_collection(COLLECTION)
            return True
        except Exception:  # noqa: BLE001
            self._collection = None
            return False

    def rebuild(self) -> tuple[int, str, str]:
        chunks = load_chunks()
        self._chunks = chunks
        if not chunks:
            self.backend, self.detail = "none", "knowledge 目录下没有可用的 markdown"
            return 0, self.backend, self.detail

        embeddings = self._embed([f"{c.title}\n{c.text}" for c in chunks])
        if embeddings is not None and self._connect_chroma():
            try:
                existing = self._collection.get(include=[])
                if existing and existing.get("ids"):
                    self._collection.delete(ids=existing["ids"])
                self._collection.add(
                    ids=[c.entry_id for c in chunks],
                    documents=[c.text for c in chunks],
                    metadatas=[c.metadata() for c in chunks],
                    embeddings=embeddings,
                )
                self.backend = "chroma"
                self.detail = f"向量库已写入 {len(chunks)} 条"
                return len(chunks), self.backend, self.detail
            except Exception as exc:  # noqa: BLE001
                self.detail = f"写入 Chroma 失败，改用关键词检索（{type(exc).__name__}）"

        self._build_keyword_index()
        self.backend = "keyword"
        self.detail = self.detail or f"关键词索引已建立，共 {len(chunks)} 条"
        return len(chunks), self.backend, self.detail

    def ensure_ready(self) -> None:
        """尽力准备索引，任何异常都不抛出。"""
        try:
            if self._connect_chroma():
                existing = self._collection.get(include=[])
                if existing and existing.get("ids"):
                    self.backend = "chroma"
                    self.detail = f"已加载向量库（{len(existing['ids'])} 条）"
                    return
            if self.chunks:
                self.rebuild()
        except Exception as exc:  # noqa: BLE001
            self.detail = f"知识库初始化失败（{type(exc).__name__}）"
            if self.chunks and not self._tf:
                self._build_keyword_index()

    # ---- 检索 -----------------------------------------------------------
    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or settings.rag_top_k
        if not self.chunks:
            return []

        if self._collection is not None:
            vectors = self._embed([query])
            if vectors is not None:
                try:
                    result = self._collection.query(
                        query_embeddings=vectors,
                        n_results=min(top_k, max(1, len(self.chunks))),
                    )
                    hits: list[Hit] = []
                    for index, entry_id in enumerate(result["ids"][0]):
                        meta = (result["metadatas"][0][index] or {})
                        distance = result["distances"][0][index]
                        hits.append(
                            Hit(
                                entry_id=entry_id,
                                title=meta.get("title", entry_id),
                                source=meta.get("source", ""),
                                text=result["documents"][0][index],
                                score=round(1.0 / (1.0 + float(distance)), 4),
                                risk_level=meta.get("risk_level", ""),
                                action=meta.get("action", ""),
                            )
                        )
                    if hits:
                        return hits
                except Exception:  # noqa: BLE001
                    self._build_keyword_index()

        if not self._tf:
            self._build_keyword_index()
        return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> list[Hit]:
        tokens = _tokens(query)
        if not tokens or not self._tf:
            return []
        total = max(1, len(self._tf))
        average = sum(self._lengths) / total
        scored: list[tuple[float, int]] = []
        for index, counter in enumerate(self._tf):
            score = 0.0
            for token in tokens:
                if token not in counter:
                    continue
                idf = math.log(1 + (total - self._df[token] + 0.5) / (self._df[token] + 0.5))
                freq = counter[token]
                denominator = freq + 1.5 * (0.25 + 0.75 * self._lengths[index] / average)
                score += idf * freq * 2.5 / denominator
            if score > 0:
                scored.append((score, index))
        scored.sort(reverse=True)
        if not scored:
            return []
        ceiling = scored[0][0] or 1.0
        hits: list[Hit] = []
        for score, index in scored[:top_k]:
            chunk = self.chunks[index]
            hits.append(
                Hit(
                    entry_id=chunk.entry_id,
                    title=chunk.title,
                    source=chunk.source,
                    text=chunk.text,
                    score=round(score / ceiling, 4),
                    risk_level=chunk.risk_level,
                    action=chunk.action,
                )
            )
        return hits


_kb: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb
