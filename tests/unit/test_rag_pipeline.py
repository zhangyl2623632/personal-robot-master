import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rag_pipeline import RAGPipeline

class TestRAGPipeline(unittest.TestCase):
    """测试RAG流水线"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 使用测试配置初始化
        with patch('src.rag_pipeline.document_loader'), \
             patch('src.rag_pipeline.vector_store_manager'), \
             patch('src.rag_pipeline.llm_client'):
            self.pipeline = RAGPipeline()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.pipeline)
        self.assertEqual(len(self.pipeline.conversation_history), 0)
        
    def test_health_check(self):
        """测试健康检查"""
        # 模拟组件状态
        self.pipeline._health_status = {
            'document_loader': True,
            'vector_store': True,
            'llm_client': True
        }
        
        # 执行健康检查
        status = self.pipeline._health_check(force=True)
        
        # 验证结果
        self.assertTrue(isinstance(status, dict))
        self.assertTrue('document_loader' in status)
        self.assertTrue('vector_store' in status)
        self.assertTrue('llm_client' in status)
        
    @patch('src.rag_pipeline.vector_store_manager')
    def test_get_vector_count(self, mock_vector_store):
        """测试获取向量数量"""
        # 模拟向量存储返回值
        mock_vector_store.get_vector_count.return_value = 100
        
        # 将流水线中的向量存储管理器替换为mock
        self.pipeline.vector_store_manager = mock_vector_store

        # 调用方法
        count = self.pipeline.get_vector_count()
        
        # 验证结果
        self.assertEqual(count, 100)
        mock_vector_store.get_vector_count.assert_called_once()
        
    @patch('src.rag_pipeline.adaptive_rag_pipeline')
    def test_query(self, mock_adaptive_pipeline):
        """测试查询功能"""
        # 模拟自适应RAG流水线的answer_query方法
        mock_adaptive_pipeline.answer_query.return_value = {"answer": "测试回答"}
        
        # 执行查询
        query = "这是一个测试问题"
        result = self.pipeline.answer_query(query)
        
        # 验证结果
        self.assertEqual(result, "测试回答")
        
    def test_add_to_conversation_history(self):
        """测试添加对话历史"""
        # 初始化空对话历史
        self.pipeline.conversation_history = []
        
        # 添加对话
        self.pipeline.add_to_conversation_history("user", "用户问题")
        
        # 验证结果
        self.assertEqual(len(self.pipeline.conversation_history), 1)
        self.assertEqual(self.pipeline.conversation_history[0]["role"], "user")
        self.assertEqual(self.pipeline.conversation_history[0]["content"], "用户问题")

if __name__ == '__main__':
    unittest.main()