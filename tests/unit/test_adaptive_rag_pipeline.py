import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.adaptive_rag_pipeline import AdaptiveRAGPipeline

class TestAdaptiveRAGPipeline(unittest.TestCase):
    """测试自适应RAG流水线"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 使用测试配置初始化
        self.pipeline = AdaptiveRAGPipeline()
        
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.strategies)
        self.assertIsNotNone(self.pipeline.prompt_templates)
        
    def test_select_strategy(self):
        """测试策略选择"""
        # 直接根据文档类型和意图类型选择策略
        strategy = self.pipeline._select_strategy("general", "specific_detail")

        # 验证返回了有效的策略
        self.assertIsNotNone(strategy)
        self.assertIsInstance(strategy, dict)

    def test_select_strategy_with_hint_report_overview(self):
        """测试通过 strategy_hint 选择报告总览策略"""
        strategy = self.pipeline._select_strategy("report_doc", "overview", strategy_hint="report_overview")
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.get('prompt_template'), 'summary')
        self.assertEqual(strategy.get('retrieval_type'), 'semantic_only')
        self.assertEqual(strategy.get('top_k'), 3)

    def test_select_strategy_with_hint_meeting_action_items(self):
        """测试通过 strategy_hint 选择会议纪要行动项策略"""
        # 新策略应已加载
        self.assertIn('meeting_minutes_action_items', self.pipeline.strategies)
        strategy = self.pipeline._select_strategy("meeting_minutes", "specific_detail", strategy_hint="meeting_minutes_action_items")
        self.assertIsNotNone(strategy)
        # 验证核心参数保持一致（成本与质量平衡）
        self.assertEqual(strategy.get('retrieval_type'), 'hybrid')
        self.assertEqual(strategy.get('top_k'), 5)
        
    @patch('src.adaptive_rag_pipeline.vector_store_manager')
    def test_retrieve_documents(self, mock_vector_store):
        """测试文档检索"""
        # 模拟向量存储检索结果
        mock_docs = [
            {'content': '文档1内容', 'metadata': {'source': 'test1.txt'}, 'score': 0.9},
            {'content': '文档2内容', 'metadata': {'source': 'test2.txt'}, 'score': 0.8}
        ]
        mock_vector_store.similarity_search.return_value = mock_docs
        
        # 测试文档检索
        query = "测试查询"
        strategy = {
            'top_k': 2,
            'score_threshold': 0.5
        }
        # 使用管线的混合检索方法
        docs = self.pipeline._perform_hybrid_retrieval(query, strategy)
        
        # 验证检索结果
        self.assertEqual(len(docs), 2)
        
    @patch('src.adaptive_rag_pipeline.global_llm_client')
    def test_generate_response(self, mock_llm_client):
        """测试响应生成"""
        # 模拟LLM响应
        mock_llm_client.generate_response.return_value = "这是一个测试回答"
        
        # 测试响应生成
        query = "测试问题"
        context = "测试上下文"
        
        # 通过格式化提示构造包含system和user的字典
        formatted = self.pipeline._format_final_prompt(query, context, self.pipeline.default_prompt_template)
        # 使用管线内部的生成回答重试方法
        response = self.pipeline._generate_answer_with_retry(formatted)
        
        # 验证生成的响应
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)

if __name__ == '__main__':
    unittest.main()