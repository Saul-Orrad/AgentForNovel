from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import create_detail_llm
from src.agents.extract_agent import ExtractAgent


class DetailAugmentationAgent:
    """细节增强节点 —— 对小说内容进行细节扩充，必要时调用 ExtractAgent 提取关键信息"""

    def __init__(self, llm: BaseChatModel | None = None, extract_agent: ExtractAgent | None = None):
        self._llm = llm or create_detail_llm()
        self._extract_agent = extract_agent or ExtractAgent()
        self._system_prompt = self._load_prompt()

    @staticmethod
    def _load_prompt() -> str:
        prompt_path = Path(__file__).parent.parent.parent / "config" / "prompt" / "detail_augmentation_agent.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "对小说内容进行细节扩充，增强场景描写和人物刻画。"

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def invoke(self, state: dict) -> dict:
        messages = [SystemMessage(content=self._system_prompt)]

        novel_content = state.get("novel_content", "")
        audit_feedback = state.get("audit_feedback", "")

        user_parts = []
        if novel_content:
            user_parts.append(f"## 当前小说内容\n{novel_content}")
        if audit_feedback:
            user_parts.append(f"## 审核反馈\n{audit_feedback}\n\n请根据审核反馈对小说进行细节增强。")
        else:
            user_parts.append("请对小说内容进行细节扩充。")

        user_message = "\n\n".join(user_parts)
        messages.append(HumanMessage(content=user_message))

        # 调用 extract_agent 获取关键信息（如果 dynamic_prompt 有值）
        dynamic_prompt = state.get("dynamic_prompt", "")
        extract_result = ""
        if dynamic_prompt:
            extract_state = {
                "novel_content": novel_content,
                "dynamic_prompt": dynamic_prompt,
            }
            extract_output = self._extract_agent.invoke(extract_state)
            extract_result = extract_output.get("extract_result", "")
            if extract_result:
                messages.append(HumanMessage(content=f"## 提取的关键信息\n{extract_result}"))

        response = self._llm.invoke(messages)

        return {
            "novel_content": response.content,
            "source_agent": "detail_augmentation",
            "target_agent": "dialogue_complementation",
            "extract_result": extract_result,
            "audit_result": False,
            "audit_feedback": "",
        }