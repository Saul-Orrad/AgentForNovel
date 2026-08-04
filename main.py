"""AgentForNovel —— 基于 LangGraph 的小说半自动化创作系统（最小化运行）

流程: detail -> dialogue，仅经过 detail 和 dialogue 两个 agent。
"""

import os
from dotenv import load_dotenv

from src.graph import create_novel_writing_graph

load_dotenv()


def main():
    """主入口：运行小说写作流程（最小化）"""
    novel_content = input("请输入初始小说内容（可为空）: ").strip()
    dynamic_prompt = input("请输入提取提示词（可为空）: ").strip()

    graph = create_novel_writing_graph()
    print("\n开始小说写作流程: detail -> dialogue\n")

    result = graph.run(novel_content=novel_content, dynamic_prompt=dynamic_prompt)

    print("\n" + "=" * 60)
    print("创作完成！")
    print("=" * 60)
    print("\n--- 最终小说内容 ---\n")
    print(result["novel_content"])


if __name__ == "__main__":
    main()