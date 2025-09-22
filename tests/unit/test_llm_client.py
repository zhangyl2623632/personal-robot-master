import unittest
import os
import sys
import unittest
import traceback
from unittest.mock import patch, MagicMock

# 导入被测试的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm_client import BaseLLMClient, DeepSeekClient, OpenAIClient, MoonshotClient

class TestLLMClient(unittest.TestCase):
    """大模型客户端单元测试"""
    
    def setUp(self):
        """每个测试方法执行前的设置"""
        # 模拟配置
        class MockConfig:
            MODEL_PROVIDER = "deepseek"
            MODEL_NAME = "deepseek-chat"
            MODEL_URL = "https://api.deepseek.com/v1/chat/completions"
            API_KEY = "test-api-key"
            DEEPSEEK_API_KEY = "test-deepseek-key"
            OPENAI_API_KEY = "test-openai-key"
            DASHSCOPE_API_KEY = "test-dashscope-key"
            MOONSHOT_API_KEY = "test-moonshot-key"
            TEMPERATURE = 0.1
            MAX_TOKENS = 2048
            TIMEOUT = 30
        self.mock_config = MockConfig()
    
    @patch('src.llm_client.DeepSeekClient._call_api')
    def test_generate_response(self, mock_call_api):
        """测试生成响应功能"""
        # 设置mock行为，模拟成功的API调用
        mock_response = {
            "choices": [{
                "message": {
                    "content": "这是测试响应内容"
                }
            }]
        }
        mock_call_api.return_value = mock_response
        
        # 初始化客户端
        client = DeepSeekClient(config=self.mock_config)
        
        # 执行测试
        response = client.generate_response("测试查询")
        
        # 验证结果
        self.assertEqual(response, "这是测试响应内容")
        mock_call_api.assert_called_once()
    
    @patch('src.llm_client.DeepSeekClient._call_api')
    def test_generate_response_with_context(self, mock_call_api):
        """测试带上下文生成响应"""
        # 设置mock行为，模拟成功的API调用
        mock_response = {
            "choices": [{
                "message": {
                    "content": "这是带上下文的测试响应"
                }
            }]
        }
        mock_call_api.return_value = mock_response
        
        # 初始化客户端
        client = DeepSeekClient(config=self.mock_config)
        
        # 执行测试
        context = ["这是上下文信息"]
        response = client.generate_response("测试查询", context=context)
        
        # 验证结果
        self.assertEqual(response, "这是带上下文的测试响应")
        mock_call_api.assert_called_once()
    
    @patch('src.llm_client.DeepSeekClient._call_api')
    def test_error_handling(self, mock_call_api):
        """测试错误处理"""
        # 设置mock行为 - 返回None表示API调用失败
        mock_call_api.return_value = None
        
        # 初始化客户端
        client = DeepSeekClient(config=self.mock_config)
        
        # 执行测试
        response = client.generate_response("测试查询")
        
        # 验证结果 - 当所有重试都失败时，generate_response会返回None
        self.assertIsNone(response)
    
    @patch('src.llm_client.DeepSeekClient._call_api')
    def test_api_key_validation(self, mock_call_api):
        """测试API密钥验证"""
        # 设置mock行为
        mock_response = {
            "choices": [{
                "message": {
                    "content": "验证成功"
                }
            }]
        }
        mock_call_api.return_value = mock_response
        
        # 初始化客户端
        client = DeepSeekClient(config=self.mock_config)
        
        # 执行测试
        is_valid = client.validate_api_key()
        
        # 验证结果
        self.assertTrue(is_valid)
    
    @patch('src.llm_client.DeepSeekClient._call_api')
    def test_api_key_invalidation(self, mock_call_api):
        """测试无效API密钥"""
        # 设置mock行为 - 返回None表示API调用失败
        mock_call_api.return_value = None
        
        # 初始化客户端
        client = DeepSeekClient(config=self.mock_config)
        
        # 执行测试
        is_valid = client.validate_api_key()
        
        # 验证结果
        self.assertFalse(is_valid)
    
    def test_client_initialization(self):
        """测试客户端初始化"""
        # 测试DeepSeek客户端初始化
        deepseek_client = DeepSeekClient(config=self.mock_config)
        self.assertEqual(deepseek_client.api_key, "test-deepseek-key")
        
        # 测试OpenAI客户端初始化
        openai_client = OpenAIClient(config=self.mock_config)
        self.assertEqual(openai_client.api_key, "test-deepseek-key")
        
        # 测试Moonshot客户端初始化
        moonshot_client = MoonshotClient(config=self.mock_config)
        self.assertEqual(moonshot_client.api_key, "test-deepseek-key")

class TestBaseLLMClient(unittest.TestCase):
    """基础LLM客户端抽象类测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建一个模拟配置对象
        from unittest.mock import MagicMock
        self.mock_config = MagicMock()
        self.mock_config.DEEPSEEK_API_KEY = "test-deepseek-key"
        self.mock_config.MODEL_PROVIDER = "deepseek"
        self.mock_config.MODEL_NAME = "deepseek-chat"
        self.mock_config.TIMEOUT = 30
        self.mock_config.TEMPERATURE = 0.1
        self.mock_config.MAX_TOKENS = 2048
    
    def test_abstract_methods(self):
        """测试抽象方法是否正确定义"""
        # 尝试直接实例化抽象类应该会失败
        with self.assertRaises(TypeError):
            BaseLLMClient()
    
    
    def test_streaming_response(self):
        """测试流式响应（通过具体实现类测试）"""
        # 创建一个测试用的具体子类
        class StreamingTestClient(BaseLLMClient):
            def __init__(self, config=None):
                # 手动设置配置，避免MagicMock比较问题
                class SimpleConfig:
                    MODEL_PROVIDER = "deepseek"
                    MODEL_NAME = "deepseek-chat"
                    RETRY_MAX_ATTEMPTS = 3
                    RETRY_BACKOFF_FACTOR = 1.5
                    RETRY_STATUS_FORCELIST = [429, 500, 502, 503, 504]
                    TIMEOUT = 30
                self.config = SimpleConfig()
                super().__init__(self.config)
                
            def _get_headers(self):
                return {}
            
            def _get_api_url(self):
                return ""
            
            def _build_payload(self, messages, stream=False):
                return {}
            
            def _call_api_stream(self, messages):
                # 模拟流式响应
                for i in range(3):
                    yield f"流式响应部分 {i+1}"
            
            def validate_api_key(self):
                return True
        
        # 初始化客户端
        client = StreamingTestClient()
        
        # 执行测试
        stream_responses = list(client.generate_response("测试查询", stream=True))
        
        # 验证结果
        self.assertEqual(len(stream_responses), 3)
        self.assertEqual(stream_responses[0], "流式响应部分 1")

if __name__ == "__main__":
    unittest.main()