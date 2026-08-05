from pathlib import Path
from datetime import datetime, timezone

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import create_dialogue_llm
from src.agents.extract_agent import ExtractAgent

def get_dynamic_prompt() -> str:
    prompt_path = Path(__file__).parent.parent.parent / "config" / "prompt" / "dynamic_prompt.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return ""

class DialogueComplementationAgent:

    def __init__(self, llm: BaseChatModel | None = None, extract_agent: ExtractAgent | None = None):
        self._llm = llm or create_dialogue_llm()
        self._extract_agent = extract_agent or ExtractAgent()
        self._system_prompt = self._load_prompt()

    @staticmethod
    def _load_prompt() -> str:
        prompt_path = Path(__file__).parent.parent.parent / "config" / "prompt" / "dialogue_complementation_agent.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return ""

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def invoke(self, state: dict) -> dict:
        t0 = datetime.now(timezone.utc)
        novel_content = state.get("novel_content", "")
        print(f"[dialogue_complementation] 开始，输入 {len(novel_content)} 字符")

        messages = [SystemMessage(content=self._system_prompt)]

        dynamic_prompt = get_dynamic_prompt()
        human_message = f"【原文】: {novel_content} 【风格】：{dynamic_prompt}"

        if human_message:
            messages.append(HumanMessage(content=human_message))

        response = self._llm.invoke(messages)

        elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
        print(f"[dialogue_complementation] 完成，输出 {len(response.content)} 字符，耗时 {elapsed:.1f}s")

        return {
            "task_id": state.get("task_id"),
            "novel_content": response.content,
            "source_agent": "dialogue_complementation",
            "target_agent": "audit",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }