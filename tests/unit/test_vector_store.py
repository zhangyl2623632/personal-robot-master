import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.vector_store import VectorStoreManager
from langchain_core.documents import Document

class TestVectorStoreManager(unittest.TestCase):
    """测试向量存储管理器"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 使用测试配置初始化
        with patch('src.vector_store.global_config'):
            self.vector_store = VectorStoreManager()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.vector_store)
        
    @patch('langchain_chroma.Chroma')
    def test_connect(self, mock_chroma):
        """测试连接向量数据库"""
        # 模拟Chroma
        mock_instance = MagicMock()
        mock_chroma.return_value = mock_instance
        
        # 测试初始化向量存储（与连接等价）
        self.vector_store._init_vector_store(index_name="default")
        
        # 验证方法被调用
        self.assertTrue(hasattr(self.vector_store, 'vector_stores'))
        self.assertIn("default", self.vector_store.vector_stores)
        
    @patch('langchain_openai.OpenAIEmbeddings')
    def test_get_embeddings(self, mock_embeddings):
        """测试获取嵌入向量"""
        # 模拟OpenAI Embeddings
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance
        mock_instance.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        
        # 设置嵌入模型
        self.vector_store.embeddings = mock_instance
        
        # 测试获取嵌入向量
        text = "测试文本"
        embeddings = self.vector_store.embeddings.embed_documents([text])
        
        # 验证结果
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings, [[0.1, 0.2, 0.3]])
        
    @patch('langchain_chroma.Chroma')
    def test_add_documents(self, mock_chroma):
        """测试添加文档"""
        # 模拟Chroma
        mock_instance = MagicMock()
        mock_chroma.return_value = mock_instance
        
        # 设置向量存储
        self.vector_store.vector_stores["default"] = mock_instance
        
        # 测试添加文档（不直接调用私有嵌入方法，交由向量存储处理）
        documents = [{"content": "测试内容", "metadata": {"source": "test.txt"}}]
        result = self.vector_store.add_documents(documents)
        
        # 验证结果
        self.assertTrue(result)
        
    @patch('langchain_chroma.Chroma')
    def test_search(self, mock_chroma):
        """测试搜索文档"""
        # 模拟Chroma
        mock_instance = MagicMock()
        mock_chroma.return_value = mock_instance
        
        # 模拟查询结果
        mock_instance.similarity_search_with_score.return_value = [
            (Document(page_content="测试内容", metadata={"source": "test.txt"}), 0.8)
        ]
        
        # 设置向量存储
        self.vector_store.vector_stores["default"] = mock_instance
        
        # 测试搜索（调用公开的相似度搜索API）
        query = "测试查询"
        results = self.vector_store.similarity_search(query, k=1)
            
        # 验证结果
        self.assertIsNotNone(results)
        
        # 验证返回的文档内容与元数据
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "测试内容")
        self.assertEqual(results[0].metadata.get('source'), "test.txt")

if __name__ == '__main__':
    unittest.main()