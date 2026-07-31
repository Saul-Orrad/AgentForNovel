import os
from pathlib import Path
from typing import Optional

import yaml

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel


def _load_config() -> dict:
    """从 env/openai_config.yaml 加载配置，返回空字典则使用环境变量回退"""
    config_path = Path(__file__).parent.parent.parent / "env" / "openai_config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class ModelFactory:
    """LLM 模型工厂 —— 统一管理模型创建与配置"""

    # 默认配置
    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_APPLY_TEMPERATURE = 0.8
    DEFAULT_AUDIT_TEMPERATURE = 0.3

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        _config = _load_config()
        self._model = model or os.getenv("OPENAI_MODEL") or _config.get("model") or self.DEFAULT_MODEL
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL") or _config.get("base_url") or None
        self._api_key = api_key or os.getenv("OPENAI_API_KEY") or _config.get("api_key") or None

    def _build_kwargs(self, temperature: float, **extra) -> dict:
        kwargs = {
            "model": self._model,
            "temperature": temperature,
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if self._api_key:
            kwargs["api_key"] = self._api_key
        kwargs.update(extra)
        return kwargs

    def create_apply_llm(self, **extra) -> BaseChatModel:
        """创建 ApplyAgent 使用的 LLM"""
        return ChatOpenAI(**self._build_kwargs(self.DEFAULT_APPLY_TEMPERATURE, **extra))

    def create_audit_llm(self, **extra) -> BaseChatModel:
        """创建 AuditAgent 使用的 LLM"""
        return ChatOpenAI(**self._build_kwargs(self.DEFAULT_AUDIT_TEMPERATURE, **extra))

    def create_llm(self, temperature: float = 0.7, **extra) -> BaseChatModel:
        """创建通用 LLM"""
        return ChatOpenAI(**self._build_kwargs(temperature, **extra))


# 全局默认工厂实例
_default_factory: ModelFactory | None = None


def get_default_factory() -> ModelFactory:
    global _default_factory
    if _default_factory is None:
        _default_factory = ModelFactory()
    return _default_factory


def create_apply_llm(**extra) -> BaseChatModel:
    """便捷函数：创建 ApplyAgent 使用的 LLM"""
    return get_default_factory().create_apply_llm(**extra)


def create_audit_llm(**extra) -> BaseChatModel:
    """便捷函数：创建 AuditAgent 使用的 LLM"""
    return get_default_factory().create_audit_llm(**extra)