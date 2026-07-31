"""AgentForNovel —— 基于 LangGraph 的小说半自动化创作系统

ApplyAgent（写作）与 AuditAgent（审核）通过 LangGraph 循环流转，
提示词分别来自 config/apply_agent_prompt.md 和 config/audit_agent_prompt.md。
"""

import os
from dotenv import load_dotenv

from src.graph import create_novel_writing_graph

load_dotenv()


def main():
    """主入口：运行小说写作流程"""
    task = input("请输入小说创作任务: ").strip()
    if not task:
        print("任务不能为空，已退出。")
        return

    max_iter = input("请输入最大循环次数 (默认 3): ").strip()
    max_iterations = int(max_iter) if max_iter.isdigit() else 3

    graph = create_novel_writing_graph()
    print(f"\n开始小说创作，最大循环次数: {max_iterations}\n")

    result = graph.run(task=task, max_iterations=max_iterations)

    print("\n" + "=" * 60)
    print("创作完成！")
    print(f"总迭代次数: {result['iteration']}")
    print(f"审核状态: {'通过' if result['approved'] else '未通过（已达最大迭代次数）'}")
    print("=" * 60)
    print("\n--- 最终小说内容 ---\n")
    print(result["novel_content"])
    print("\n--- 最后审核反馈 ---\n")
    print(result["audit_feedback"])


if __name__ == "__main__":
    main()