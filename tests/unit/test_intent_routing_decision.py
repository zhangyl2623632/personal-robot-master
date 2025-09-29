import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.query_intent_classifier import QueryIntentClassifier


class TestIntentRoutingDecision(unittest.TestCase):
    """测试 YAML 驱动的 get_routing_decision 路由决策"""

    def setUp(self):
        self.classifier = QueryIntentClassifier()

    def test_retrieval_needed_routing(self):
        """检索型意图应路由到 RAG，默认参数"""
        query = "请基于文档说明接口参数"
        decision = self.classifier.get_routing_decision(query)
        self.assertEqual(decision['action'], 'rag')
        self.assertEqual(decision['params'].get('top_k'), 6)
        self.assertEqual(decision['params'].get('retrieval_strategy'), 'hybrid')

    def test_direct_answer_preset(self):
        """直接回答型应命中预设答案或触发短回复"""
        query = "健康检查"
        decision = self.classifier.get_routing_decision(query)
        self.assertIn(decision['action'], ['preset_or_llm', 'refuse_or_short_reply'])
        # 当 action 为 preset_or_llm 时，应该能命中预设答案
        if decision['action'] == 'preset_or_llm':
            self.assertIsNotNone(decision.get('preset_answer'))

    def test_direct_answer_supported_doc_types(self):
        """新增高频问题：支持文档类型，应命中预设答案"""
        query = "支持文档类型"
        decision = self.classifier.get_routing_decision(query)
        self.assertEqual(decision['action'], 'preset_or_llm')
        self.assertIsNotNone(decision.get('preset_answer'))

    def test_tool_call_routing(self):
        """工具调用型应路由到 tool 并包含允许的工具列表"""
        query = "请清空知识库"
        decision = self.classifier.get_routing_decision(query)
        self.assertEqual(decision['action'], 'tool')
        allowed = decision['params'].get('allowed_tools', [])
        self.assertTrue(len(allowed) > 0)

    def test_chitchat_routing(self):
        """闲聊型应路由到拒绝或短回复"""
        query = "讲个笑话"
        decision = self.classifier.get_routing_decision(query)
        self.assertEqual(decision['action'], 'refuse_or_short_reply')

    def test_overrides_technical_doc_retrieval(self):
        """技术文档检索偏好覆盖 top_k 与 strategy_hint"""
        query = "请基于文档说明接口参数"
        decision = self.classifier.get_routing_decision(query, document_type='technical_doc')
        self.assertEqual(decision['action'], 'rag')
        self.assertEqual(decision['params'].get('top_k'), 8)
        self.assertEqual(decision['params'].get('strategy_hint'), "技术文档_具体细节查询")

    def test_overrides_technical_doc_tool(self):
        """技术文档工具偏好覆盖 allowed_tools"""
        query = "请清空知识库"
        decision = self.classifier.get_routing_decision(query, document_type='technical_doc')
        self.assertEqual(decision['action'], 'tool')
        self.assertEqual(decision['params'].get('allowed_tools'), ["direct_tool_call"])

    def test_overrides_requirement_doc_direct_answer(self):
        """需求文档直接回答偏好启用预设"""
        query = "你是谁"
        decision = self.classifier.get_routing_decision(query, document_type='requirement_doc')
        self.assertIn(decision['action'], ['preset_or_llm', 'refuse_or_short_reply'])
        if decision['action'] == 'preset_or_llm':
            self.assertTrue(decision['params'].get('preset_enabled', True))

    def test_overrides_report_doc_overview(self):
        """报告文档概览偏好应设置 strategy_hint 与 top_k"""
        query = "请给出报告的概览"
        decision = self.classifier.get_routing_decision(query, document_type='report_doc')
        self.assertEqual(decision['action'], 'rag')
        self.assertEqual(decision['params'].get('strategy_hint'), 'report_overview')
        self.assertEqual(decision['params'].get('top_k'), 3)

    def test_overrides_meeting_minutes_action_items(self):
        """会议纪要行动项偏好应设置 strategy_hint 与 top_k"""
        query = "列出会议纪要的行动项"
        decision = self.classifier.get_routing_decision(query, document_type='meeting_minutes')
        self.assertEqual(decision['action'], 'rag')
        self.assertEqual(decision['params'].get('strategy_hint'), 'meeting_minutes_action_items')
        self.assertEqual(decision['params'].get('top_k'), 5)


if __name__ == '__main__':
    unittest.main()