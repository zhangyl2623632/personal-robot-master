import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.document_classifier import DocumentClassifier

class TestDocumentClassifier(unittest.TestCase):
    """测试文档分类器"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 使用测试配置初始化
        with patch('src.document_classifier.global_config', create=True):
            self.classifier = DocumentClassifier()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.classifier)
        
    def test_classify_document(self):
        """测试文档分类（对齐当前实现）"""
        document_content = "本合同由甲方和乙方签订，双方同意以下条款..."
        # 提供伪造文件路径并mock存在与大小
        fake_path = os.path.join(os.path.dirname(__file__), 'contract.txt')
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=100), \
             patch('os.path.getmtime', return_value=0):
            # 直接传入内容以避免文件读取
            result = self.classifier.classify_document(fake_path, document_content)
        # 验证结果结构
        self.assertIn('document_type', result)
        self.assertIn('confidence', result)
        
    def test_extract_metadata(self):
        """测试元数据提取（适配现有实现）"""
        document_content = "这是一份测试文档，包含一些关键词如合同、条款等"
        fake_path = os.path.join(os.path.dirname(__file__), 'contract.txt')
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=100), \
             patch('os.path.getmtime', return_value=0):
            metadata = self.classifier._extract_metadata(fake_path, document_content)
        self.assertIsInstance(metadata, dict)
        self.assertIn('file_name', metadata)
        # 如果存在文件，可能包含文件大小和修改时间
        self.assertIn('file_size', metadata)
        
    def test_rule_based_classification(self):
        """测试规则分类流程（通过公开接口）"""
        document_content = "本合同由甲方和乙方签订，双方同意以下条款..."
        fake_path = os.path.join(os.path.dirname(__file__), 'contract.txt')
        # 动态设置类型以确保匹配
        self.classifier.document_types = [
            {
                'id': 'contract',
                'name': '合同',
                'keywords': ['合同', '条款', '甲方', '乙方'],
                'file_extensions': ['.txt']
            }
        ]
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=100), \
             patch('os.path.getmtime', return_value=0):
            result = self.classifier.classify_document(fake_path, document_content)
        self.assertEqual(result['document_type'], 'contract')
        self.assertGreaterEqual(result['confidence'], 0.5)
        
    def test_default_classification_without_ml(self):
        """测试在无ML和LLM情况下的默认分类流程"""
        document_content = "这是一份普通说明文档。"
        fake_path = os.path.join(os.path.dirname(__file__), 'note.txt')
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=100), \
             patch('os.path.getmtime', return_value=0):
            result = self.classifier.classify_document(fake_path, document_content)
        self.assertIn(result['document_type'], ['general', 'unknown'])

if __name__ == '__main__':
    unittest.main()