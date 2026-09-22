"""OpenAI 兼容的大模型客户端。

没配 key 时 available 为 False，各引擎据此走降级路径，流程不中断。
"""

from __future__ import annotations

import json
import time

from ..config import settings


def extract_json(text: str) -> dict | None:
    """从模型回复里抠出第一个 JSON 对象。"""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


class LLMClient:
    def __init__(self) -> None:
        self._client = None

    @property
    def available(self) -> bool:
        return bool(settings.llm_api_key)

    @property
    def model(self) -> str:
        return settings.llm_model

    def _ensure(self):
        if self._client is None:
            if not settings.llm_api_key:
                raise RuntimeError("未配置 LLM_API_KEY")
            from openai import OpenAI

            self._client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout=settings.llm_timeout,
                max_retries=1,
            )
        return self._client

    def chat(self, system: str, user: str, json_mode: bool = True) -> tuple[str, int]:
        """返回 (回复文本, 耗时毫秒)。"""
        client = self._ensure()
        payload = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": settings.llm_temperature,
        }
        started = time.time()
        if json_mode:
            try:
                response = client.chat.completions.create(
                    **payload, response_format={"type": "json_object"}
                )
                return (
                    response.choices[0].message.content or "",
                    int((time.time() - started) * 1000),
                )
            except Exception:  # noqa: BLE001
                # 部分兼容网关不实现 response_format，退回普通模式再试一次。
                pass
        response = client.chat.completions.create(**payload)
        return response.choices[0].message.content or "", int((time.time() - started) * 1000)


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
