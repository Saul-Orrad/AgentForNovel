from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import create_extract_llm


class ExtractAgent:
    """抽取节点 —— 根据调用方传入的 dynamic_prompt 从小说内容中提取关键信息，将结果返回给调用方"""

    def __init__(self, llm: BaseChatModel | None = None):
        self._llm = llm or create_extract_llm()
        self._system_prompt = self._load_prompt()

    @staticmethod
    def _load_prompt() -> str:
        prompt_path = Path(__file__).parent.parent.parent / "config" / "prompt" / "extract_agent.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "从小说内容中提取关键信息，根据提示返回结构化结果。"

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def invoke(self, state: dict) -> dict:
        """
        Args:
            state: 包含 novel_content（小说内容）和 dynamic_prompt（调用方传入的提取提示词）
        Returns:
            dict: 包含 extract_result（提取结果）
        """
        messages = [SystemMessage(content=self._system_prompt)]

        novel_content = state.get("novel_content", "")
        dynamic_prompt = state.get("dynamic_prompt", "")

        user_parts = []
        if novel_content:
            user_parts.append(f"## 小说内容\n{novel_content}")
        if dynamic_prompt:
            user_parts.append(f"## 提取任务\n{dynamic_prompt}")
        else:
            user_parts.append("请从以上小说内容中提取关键信息。")

        user_message = "\n\n".join(user_parts)
        messages.append(HumanMessage(content=user_message))

        response = self._llm.invoke(messages)

        return {
            "extract_result": response.content,
        }