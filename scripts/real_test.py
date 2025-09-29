import os
import sys
import json
import time

# 兼容直接运行脚本的导入路径
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.rag_pipeline import rag_pipeline
from src.adaptive_rag_pipeline import adaptive_rag_pipeline


def run_query(query, document_type=None, strategy_hint=None):
    metadata = {"strategy_hint": strategy_hint} if strategy_hint else {}
    result = adaptive_rag_pipeline.answer_query(
        query=query,
        document_type=document_type,
        metadata=metadata,
    )
    # 提取关键字段用于核查
    out = {
        "query": query,
        "document_type": result.get("document_type"),
        "intent_type": result.get("intent_analysis", {}).get("intent_type"),
        "retrieved_documents": result.get("retrieved_documents"),
        "strategy": {
            "retrieval_type": result.get("strategy", {}).get("retrieval_type"),
            "top_k": result.get("strategy", {}).get("top_k"),
            "prompt_template": result.get("strategy", {}).get("prompt_template"),
        },
        "answer_preview": (result.get("answer") or "")[:200],
        "metadata": result.get("metadata"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main():
    # 文档路径
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("test_data", "Merchant Quick loan TRS.docx")
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        print(f"文档不存在: {abs_path}")
        sys.exit(1)

    # 导入文档到向量库
    print("开始导入文档到向量库...")
    ok = rag_pipeline.add_single_document(abs_path)
    print(f"导入结果: {'成功' if ok else '失败'}，当前向量数: {rag_pipeline.get_vector_count()}")

    # 查询1：报告概览（使用映射键 report_overview）
    print("\n查询1：报告概览（strategy_hint=report_overview）")
    run_query(
        query="请基于该文档提供报告概览与核心要点。",
        document_type="report_doc",
        strategy_hint="report_overview",
    )

    # 查询2：行动项（直接使用策略ID meeting_minutes_action_items）
    print("\n查询2：行动项（strategy_hint=meeting_minutes_action_items）")
    run_query(
        query="请列出可执行的行动项及负责人建议。",
        document_type="meeting_minutes",
        strategy_hint="meeting_minutes_action_items",
    )


if __name__ == "__main__":
    main()