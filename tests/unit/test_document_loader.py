import unittest
import sys
import os
from unittest.mock import patch, MagicMock, mock_open

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.document_loader import DocumentLoader

class TestDocumentLoader(unittest.TestCase):
    """测试文档加载器"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 使用测试配置初始化
        with patch('src.document_loader.global_config'):
            self.document_loader = DocumentLoader()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.document_loader)
        
    @patch('src.document_loader.TextLoader')
    def test_load_text_document(self, mock_text_loader):
        """测试加载文本文档"""
        # 模拟文本加载器
        mock_instance = MagicMock()
        mock_text_loader.return_value = mock_instance
        mock_instance.load.return_value = [
            MagicMock(page_content="测试内容", metadata={"source": "test.txt"})
        ]
        
        # 创建临时测试文件路径
        test_file_path = os.path.join(os.path.dirname(__file__), "test.txt")
        
        # 测试加载文本文档
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=True), \
             patch('os.path.getsize', return_value=100), \
             patch('os.stat', return_value=MagicMock(st_mtime=0, st_size=100)), \
             patch('builtins.open', mock_open(read_data="测试内容")):
            result = self.document_loader.load_document(test_file_path)
        
        # 验证结果
        self.assertIsNotNone(result)
        
    @patch('src.document_loader.PDFLoader')
    def test_load_pdf_document(self, mock_pdf_loader):
        """测试加载PDF文档"""
        # 模拟PDF加载器
        mock_instance = MagicMock()
        mock_pdf_loader.return_value = mock_instance
        mock_instance.load.return_value = [
            MagicMock(page_content="PDF测试内容", metadata={"source": "test.pdf", "page": 1})
        ]
        
        # 创建临时测试文件路径
        test_file_path = os.path.join(os.path.dirname(__file__), "test.pdf")
        
        # 测试加载PDF文档
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=True), \
             patch('os.path.getsize', return_value=100), \
             patch('os.stat', return_value=MagicMock(st_mtime=0, st_size=100)):
            # 模拟二进制读取PDF头
            m = mock_open()
            m.return_value.read.return_value = b'%PDF'
            with patch('builtins.open', m):
                result = self.document_loader.load_document(test_file_path)
        
        # 验证结果
        self.assertIsNotNone(result)
    
    def test_split_text(self):
        """测试文本分块（适配现有私有方法）"""
        test_text = "这是一个测试文本。" * 3
        sentences = self.document_loader._split_text_into_sentences(test_text)
        self.assertGreaterEqual(len(sentences), 3)

if __name__ == '__main__':
    unittest.main()