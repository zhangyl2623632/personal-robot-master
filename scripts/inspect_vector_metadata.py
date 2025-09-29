import json
import os
import sys
from collections import defaultdict

# 允许直接从 src 目录导入模块（无需包化）
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
# 让解释器识别 'src.xxx' 形式的导入
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.vector_store import vector_store_manager

def main():
    target_source = "Merchant Quick loan TRS.docx"
    print(f"检查目标来源: {target_source}")

    # 通过元数据过滤获取该来源的所有分片
    docs = vector_store_manager.search_by_metadata({"source": target_source}, k=200)
    print(f"命中文档分片数量: {len(docs)}")

    agg = defaultdict(set)
    samples = []
    for i, doc in enumerate(docs or []):
        md = getattr(doc, "metadata", {}) or {}
        # 聚合所有元数据键的唯一值
        for k, v in md.items():
            if v is None:
                continue
            if isinstance(v, list):
                for item in v:
                    agg[k].add(str(item))
            else:
                agg[k].add(str(v))
        if i < 3:
            samples.append({"index": i+1, "page": md.get("page"), "metadata": md})

    # 仅展示与作者/日期相关的关键键
    interesting_keys = [
        "author", "created", "modified", "date", "last_modified_by", "release", "version"
    ]

    result = {
        "vector_count_for_source": len(docs or []),
        "keys_present": sorted(list(agg.keys())),
        "values_by_key": {k: sorted(list(agg.get(k, set()))) for k in interesting_keys},
        "sample_metadata": samples,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()