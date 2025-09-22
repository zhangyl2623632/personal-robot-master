import os
import sys
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

# 导入被测试的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.rag_pipeline import RAGPipeline
from src.document_loader import DocumentLoader
from src.vector_store import VectorStoreManager
from src.llm_client import BaseLLMClient

class TestRAGPipeline(unittest.TestCase):
    """RAG流水线集成测试"""
    
    def setUp(self):
        """每个测试方法执行前的设置"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 模拟配置
        class MockConfig:
            DOCUMENTS_PATH = self.temp_dir
            VECTOR_STORE_PATH = os.path.join(self.temp_dir, "vector_store")
            EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
            CHUNK_SIZE = 500
            CHUNK_OVERLAP = 100
            SIMILARITY_THRESHOLD = 0.15
            MODEL_PROVIDER = "deepseek"
            MODEL_NAME = "deepseek-chat"
            API_KEY = "test-api-key"
            TEMPERATURE = 0.1
            MAX_TOKENS = 2048
            TIMEOUT = 30
        self.mock_config = MockConfig()
        
        # 模拟组件
        self.mock_document_loader = MagicMock(spec=DocumentLoader)
        self.mock_vector_store_manager = MagicMock(spec=VectorStoreManager)
        self.mock_llm_client = MagicMock(spec=BaseLLMClient)
        
        # 替换RAGPipeline的组件
        with patch('src.rag_pipeline.document_loader', self.mock_document_loader):
            with patch('src.rag_pipeline.vector_store_manager', self.mock_vector_store_manager):
                with patch('src.rag_pipeline.llm_client', self.mock_llm_client):
                    self.rag_pipeline = RAGPipeline()
    
    def tearDown(self):
        """每个测试方法执行后的清理"""
        # 清理临时目录
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_process_documents(self):
        """测试处理文档功能"""
        # 设置mock行为
        mock_documents = [Document(page_content="测试文档", metadata={"source": "test.txt"})]
        self.mock_document_loader.load_directory.return_value = mock_documents
        self.mock_document_loader.split_documents.return_value = mock_documents
        self.mock_vector_store_manager.add_documents.return_value = True
        
        # 执行测试
        result = self.rag_pipeline.process_documents(self.temp_dir)
        
        # 验证结果
        self.assertTrue(result)
        self.mock_document_loader.load_directory.assert_called_once_with(self.temp_dir)
        self.mock_document_loader.split_documents.assert_called_once_with(mock_documents)
        self.mock_vector_store_manager.add_documents.assert_called_once_with(mock_documents)
    
    def test_add_single_document(self):
        """测试添加单个文档功能"""
        # 设置mock行为
        temp_file = os.path.join(self.temp_dir, "test.txt")
        with open(temp_file, "w") as f:
            f.write("测试文档内容")
        
        mock_document = Document(page_content="测试文档内容", metadata={"source": temp_file})
        self.mock_document_loader.load_document.return_value = [mock_document]
        self.mock_document_loader.split_documents.return_value = [mock_document]
        self.mock_vector_store_manager.add_documents.return_value = True
        
        # 执行测试
        result = self.rag_pipeline.add_single_document(temp_file)
        
        # 验证结果
        self.assertTrue(result)
        self.mock_document_loader.load_document.assert_called_once_with(temp_file)
        self.mock_document_loader.split_documents.assert_called_once_with([mock_document])
        self.mock_vector_store_manager.add_documents.assert_called_once_with([mock_document])
    
    def test_answer_query(self):
        """测试回答查询功能"""
        # 设置mock行为
        mock_context = Document(page_content="相关上下文信息", metadata={"source": "test.txt"})
        self.mock_vector_store_manager.similarity_search.return_value = [mock_context]
        self.mock_llm_client.generate_response.return_value = "这是基于上下文的回答"
        
        # 执行测试
        answer = self.rag_pipeline.answer_query("测试查询")
        
        # 验证结果
        self.assertEqual(answer, "这是基于上下文的回答")
        self.mock_vector_store_manager.similarity_search.assert_called_once()
        self.mock_llm_client.generate_response.assert_called_once()
    
    def test_answer_query_no_context(self):
        """测试无上下文回答查询"""
        # 设置mock行为
        self.mock_vector_store_manager.similarity_search.return_value = []
        self.mock_llm_client.generate_response.return_value = "根据现有资料，无法回答该问题。"
        
        # 执行测试
        answer = self.rag_pipeline.answer_query("测试查询")
        
        # 验证结果
        self.assertEqual(answer, "根据现有资料，无法回答该问题。")
    
    def test_preprocess_query_overview(self):
        """测试预处理概述类查询"""
        # 执行测试 - 测试包含概述关键词的查询
        processed_query = self.rag_pipeline.preprocess_query("请概述本文档的主要内容")
        
        # 验证结果 - 应该转换为标准概述请求
        self.assertIn("提取文档标题", processed_query)
        
        # 执行测试 - 测试不包含概述关键词的查询
        normal_query = "测试普通查询"
        processed_normal_query = self.rag_pipeline.preprocess_query(normal_query)
        
        # 验证结果 - 应该保持不变
        self.assertEqual(processed_normal_query, normal_query)
    
    def test_conversation_history(self):
        """测试对话历史功能"""
        # 设置mock行为
        self.mock_vector_store_manager.similarity_search.return_value = []
        self.mock_llm_client.generate_response.return_value = "回答1"
        
        # 执行第一次查询
        answer1 = self.rag_pipeline.answer_query("查询1")
        
        # 验证对话历史
        self.assertEqual(len(self.rag_pipeline.conversation_history), 2)  # 查询和回答
        self.assertEqual(self.rag_pipeline.conversation_history[0]["role"], "user")
        self.assertEqual(self.rag_pipeline.conversation_history[0]["content"], "查询1")
        
        # 清空对话历史
        self.rag_pipeline.clear_conversation_history()
        self.assertEqual(len(self.rag_pipeline.conversation_history), 0)
    
    def test_validate_api_key(self):
        """测试API密钥验证"""
        # 设置mock行为
        self.mock_llm_client.validate_api_key.return_value = True
        
        # 执行测试
        is_valid = self.rag_pipeline.validate_api_key()
        
        # 验证结果
        self.assertTrue(is_valid)
        self.mock_llm_client.validate_api_key.assert_called_once()
    
    def test_get_vector_count(self):
        """测试获取向量数量"""
        # 设置mock行为
        self.mock_vector_store_manager.get_vector_count.return_value = 100
        
        # 执行测试
        count = self.rag_pipeline.get_vector_count()
        
        # 验证结果
        self.assertEqual(count, 100)
        self.mock_vector_store_manager.get_vector_count.assert_called_once()

class TestRAGPipelineEdgeCases(unittest.TestCase):
    """RAG流水线边界情况测试"""
    
    def setUp(self):
        """每个测试方法执行前的设置"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 模拟配置
        class MockConfig:
            DOCUMENTS_PATH = self.temp_dir
            VECTOR_STORE_PATH = os.path.join(self.temp_dir, "vector_store")
        self.mock_config = MockConfig()
        
        # 模拟组件
        self.mock_document_loader = MagicMock(spec=DocumentLoader)
        self.mock_vector_store_manager = MagicMock(spec=VectorStoreManager)
        self.mock_llm_client = MagicMock(spec=BaseLLMClient)
        
        # 替换RAGPipeline的组件
        with patch('src.rag_pipeline.document_loader', self.mock_document_loader):
            with patch('src.rag_pipeline.vector_store_manager', self.mock_vector_store_manager):
                with patch('src.rag_pipeline.llm_client', self.mock_llm_client):
                    self.rag_pipeline = RAGPipeline()
    
    def test_process_empty_directory(self):
        """测试处理空目录"""
        # 设置mock行为
        self.mock_document_loader.load_directory.return_value = []
        
        # 执行测试
        result = self.rag_pipeline.process_documents(self.temp_dir)
        
        # 验证结果
        self.assertFalse(result)
    
    def test_process_documents_failure(self):
        """测试处理文档失败情况"""
        # 设置mock行为
        self.mock_document_loader.load_directory.side_effect = Exception("加载失败")
        
        # 执行测试
        result = self.rag_pipeline.process_documents(self.temp_dir)
        
        # 验证结果
        self.assertFalse(result)
    
    def test_answer_empty_query(self):
        """测试回答空查询"""
        # 执行测试
        answer = self.rag_pipeline.answer_query("")
        
        # 验证结果 - 空查询应该返回None
        # 注意：这里不需要mock，因为在实际代码中，空查询会被直接拒绝，不会调用LLM
        self.assertIsNone(answer)

if __name__ == "__main__":
    unittest.main()