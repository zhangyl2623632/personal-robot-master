import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 临时修改向量存储路径
official_env = os.environ.get('VECTOR_STORE_PATH')
os.environ['VECTOR_STORE_PATH'] = './vector_store_new'

from src.rag_pipeline import rag_pipeline

print("使用临时向量存储路径: ./vector_store_new")
print("正在处理data目录中的文档...")

# 处理文档
success = rag_pipeline.process_documents('./data')

if success:
    vector_count = rag_pipeline.get_vector_count()
    print(f"文档处理成功！向量存储中文档数量: {vector_count}")
    print("\n测试查询: 如何理解周易？")
    answer = rag_pipeline.answer_query("如何理解周易？")
    print(f"回答: {answer}")
else:
    print("文档处理失败，请检查日志获取更多信息")

# 恢复原始环境变量
if official_env:
    os.environ['VECTOR_STORE_PATH'] = official_env