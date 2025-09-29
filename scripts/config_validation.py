import os
import sys
import json
import time
from typing import Any, Dict, List

# 兼容直接运行，确保能找到 src 包
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.document_loader import document_loader
from src.query_intent_classifier import query_intent_classifier
from src.adaptive_rag_pipeline import adaptive_rag_pipeline
from src.rag_pipeline import rag_pipeline


def _safe_add_document(path: str) -> Dict[str, Any]:
    try:
        ok = rag_pipeline.add_single_document(path)
        return {"path": path, "added": bool(ok)}
    except Exception as e:
        return {"path": path, "added": False, "error": str(e)}


def validate_chunking(sample_path: str) -> Dict[str, Any]:
    start = time.time()
    docs = []
    try:
        docs = document_loader.load_document(sample_path)
    except Exception as e:
        return {"file": sample_path, "error": f"load_document failed: {e}"}

    elapsed = time.time() - start
    count = len(docs)
    # 提取分块后的文档类型与元数据
    doc_type = docs[0].metadata.get("document_type") if docs else "unknown"
    strategy_meta = {
        "chunking_strategies": document_loader.processing_config.get("chunking_strategies", {}),
        "semantic_chunking": document_loader.processing_config.get("semantic_chunking", {}),
        "concurrency": document_loader.processing_config.get("concurrency", {}),
    }
    return {
        "file": sample_path,
        "doc_type": doc_type,
        "chunk_count": count,
        "elapsed_sec": round(elapsed, 3),
        "strategy_meta": strategy_meta,
    }


def validate_query_routing() -> List[Dict[str, Any]]:
    tests = [
        ("帮助", None),
        ("请根据文档说明总结关键要点", None),
        ("请清空知识库", None),
        ("你好", None),
    ]
    results = []
    for q, doc_type in tests:
        decision = query_intent_classifier.get_routing_decision(q, document_type=doc_type)
        results.append({
            "query": q,
            "intent": decision.get("intent", {}),
            "action": decision.get("action"),
            "params": decision.get("params", {}),
            "preset_answer": decision.get("preset_answer"),
        })
    return results


def validate_rag_strategies(sample_queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for item in sample_queries:
        q = item.get("query")
        meta = item.get("metadata", {})
        try:
            ans = adaptive_rag_pipeline.answer_query(q, metadata=meta)
            out.append({
                "query": q,
                "metadata": meta,
                "retrieved_docs": ans.get("retrieved_docs", []),
                "retrieved_count": len(ans.get("retrieved_docs", [])),
                "strategy": ans.get("strategy", {}),
                "answer_preview": (ans.get("answer", "") or "")[:200],
            })
        except Exception as e:
            out.append({
                "query": q,
                "metadata": meta,
                "error": str(e),
            })
    return out


def write_results(path: str, records: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 为提升在部分 Windows 工具中的可读性，使用带 BOM 的 UTF-8-SIG
    with open(path, "a", encoding="utf-8-sig") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    # 1) 准备样例文档
    samples = [
        os.path.join(PROJECT_ROOT, "test_data", "Merchant Quick loan TRS.docx"),
        os.path.join(PROJECT_ROOT, "test_data", "documents", "test_contract.txt"),
    ]

    add_reports = [_safe_add_document(p) for p in samples if os.path.exists(p)]

    # 2) 验证分块策略与语义分块
    chunk_reports = [validate_chunking(p) for p in samples if os.path.exists(p)]

    # 3) 验证查询路由与预设答案
    routing_reports = validate_query_routing()

    # 4) 验证多策略查询（覆盖 strategy_hint 与 document_type）
    rag_tests = [
        {
            "query": "请给出该系统的报告概览要点",
            "metadata": {"strategy_hint": "report_overview", "document_type": "report_doc"},
        },
        {
            "query": "这次会议的行动项和责任人分别是什么？",
            "metadata": {"strategy_hint": "meeting_minutes_action_items", "document_type": "meeting_minutes"},
        },
        {
            "query": "请用英文总结关键功能模块",
            "metadata": {"strategy_hint": "multilingual_summary", "document_type": "multilingual_doc", "language": "en"},
        },
        {
            "query": "列出该技术文档中的关键接口参数",
            "metadata": {"strategy_hint": "technical_detail", "document_type": "technical_doc"},
        },
    ]
    rag_reports = validate_rag_strategies(rag_tests)

    # 5) 汇总并写入结果
    results_path = os.path.join(PROJECT_ROOT, "test_data", "results", "config_validation.jsonl")
    write_results(results_path, [
        {"type": "add", "data": add_reports},
        {"type": "chunking", "data": chunk_reports},
        {"type": "routing", "data": routing_reports},
        {"type": "rag", "data": rag_reports},
    ])

    # 控制台输出简要摘要
    print("[Add]", json.dumps(add_reports, ensure_ascii=False))
    print("[Chunking]", json.dumps(chunk_reports, ensure_ascii=False))
    print("[Routing]", json.dumps(routing_reports, ensure_ascii=False))
    print("[RAG]", json.dumps(rag_reports, ensure_ascii=False))
    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()
# 在 Windows 控制台下确保标准输出使用 UTF-8，减少乱码概率
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass