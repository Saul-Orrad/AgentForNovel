from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import create_extract_llm


class ExtractAgent:

    def __init__(self, llm: BaseChatModel | None = None):
        self._llm = llm or create_extract_llm()
        self._system_prompt = self._load_prompt()

    @staticmethod
    def _load_prompt() -> str:
        prompt_path = Path(__file__).parent.parent.parent / "config" / "prompt" / "extract_agent.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return ""

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def invoke(self, state: dict) -> dict:
        messages = [SystemMessage(content=self._system_prompt)]

        novel_content = state.get("novel_content", "")
        dynamic_prompt = state.get("dynamic_prompt", "")
        human_message = f"【原文】: {novel_content} 【提取提示词】：{dynamic_prompt}"

        if human_message:
            messages.append(HumanMessage(content=human_message))

        response = self._llm.invoke(messages)

        return {
            "extract_result": response.content,
        }
