from src.vector_store import vector_store_manager

# 检查向量数量
vector_count = vector_store_manager.get_vector_count()
print('向量数量:', vector_count)

# 搜索相关文档
docs = vector_store_manager.similarity_search('贷款')
print('相关文档数量:', len(docs))

# 如果有相关文档，打印第一个文档的部分内容
if docs:
    print('第一个文档内容前200字符:', docs[0].page_content[:200])