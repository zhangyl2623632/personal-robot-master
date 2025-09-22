# llm_client.py
# 通用大模型客户端，支持 DeepSeek / OpenAI / Qwen / Moonshot / SparkDesk 等
# 支持严格模式、结构化输出、元数据增强、意图识别

import os
import re
import logging
import requests
import time
import traceback
from requests.adapters import Retry, HTTPAdapter
import json
from typing import List, Dict, Any, Optional, Union, Generator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# 假设你有一个 config 模块，若没有，可替换为 dotenv 或硬编码
try:
    from src.config import global_config
except ImportError:
    # 如果没有 config，提供默认配置（你可自行修改）
    class MockConfig:
        MODEL_PROVIDER = "deepseek"  # 可选: deepseek, openai, qwen, moonshot, spark
        MODEL_NAME = "deepseek-chat"
        MODEL_URL = "https://api.deepseek.com/v1/chat/completions"
        API_KEY = os.getenv("DEEPSEEK_API_KEY") or "sk-xxx"
        TEMPERATURE = 0.1
        MAX_TOKENS = 2048
        TIMEOUT = 30

    # 使用mock配置
    global_config = MockConfig()

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 定义模型提供商枚举
class ModelProvider(str, Enum):
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    QWEN = "qwen"
    MOONSHOT = "moonshot"
    SPARK = "spark"
    # 可以继续扩展

# 定义一些工具函数
# ======================== 
# 🛠️ 工具函数 
# ======================== 
def preprocess_query(query: str) -> str:
    """预处理用户查询"""
    # 去除多余空格和换行符
    processed = ' '.join(query.strip().split())
    # 转小写（可选，根据需要）
    # processed = processed.lower()
    return processed

def enhance_context_with_metadata(context: str) -> str:
    """用元数据增强上下文"""
    # 提取元数据
    metadata = extract_metadata_from_text(context)
    
    # 添加元数据到上下文
    enhanced_context = ""
    if 'title' in metadata:
        enhanced_context += f"【标题】{metadata['title']}\n\n"
    
    enhanced_context += context
    return enhanced_context

def extract_metadata_from_text(text: str) -> Dict[str, str]:
    """从文本中提取元数据"""
    metadata = {}
    lines = text.strip().split('\n')[:200]  # 只处理前200行

    # 正则模式匹配常见元数据
    patterns = {
        'version': r'(Version|版本)[\s:]+([\d\.]+)',
        'date': r'(Date|日期)[\s:]+([\d\-\/]+)',
        'author': r'(Author|作者)[\s:]+([^\n]+)',
        'release': r'(Release|发布)[\s:]+([^\n]+)',
    }

    # 提取元数据
    for key, pattern in patterns.items():
        for line in lines[:50]:  # 只在前50行查找
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip()

    # 启发式提取主标题
    if 'title' not in metadata:
        for line in lines[:10]:
            stripped = line.strip()
            if stripped and len(stripped) > 5 and not re.search(r'^(Version|Date|Author|Release|Prepared|Document)', stripped, re.I):
                metadata['title'] = stripped
                break

    # 提取关键段落（功能/配置/费用/流程）
    key_sections = {
        'main_functions': [], 
        'system_config': [], 
        'fee_structure': [], 
        'process_flow': [],
        'data_structures': []
    }
    current_section = None

    for i, line in enumerate(lines[:200]):
        stripped = line.strip()
        if not stripped: continue

        # 识别章节标题（支持中英文）
        if re.search(r'功能|Functions|Features', stripped, re.I):
            current_section = 'main_functions'
        elif re.search(r'配置|Configuration|Settings', stripped, re.I):
            current_section = 'system_config'
        elif re.search(r'费用|Fee|Cost|Price|费率', stripped, re.I):
            current_section = 'fee_structure'
        elif re.search(r'流程|Process|Steps|步骤', stripped, re.I):
            current_section = 'process_flow'
        elif re.search(r'数据结构|Data Structure|表格|Table', stripped, re.I):
            current_section = 'data_structures'
        elif re.search(r'^\d+\.', stripped):  # 数字标题
            pass  # 保持当前section
        elif stripped and current_section and not stripped.endswith(':') and i < len(lines) - 1 and lines[i+1].strip():
            # 将内容添加到当前section
            key_sections[current_section].append(stripped)

    # 将列表转换为字符串
    for key, value in key_sections.items():
        if value:
            metadata[key] = '\n'.join(value[:5])  # 只保留前5个项目

    return metadata

# ======================== 
# 🎯 客户端抽象基类 
# ======================== 
class BaseLLMClient(ABC):
    """LLM 客户端抽象基类"""

    def __init__(self, config=None):
        self.config = config or global_config
        self.provider = ModelProvider(self.config.MODEL_PROVIDER.lower())
        # 使用getattr确保即使配置项不存在也不会抛出异常
        # 根据模型提供商获取对应的API密钥
        if self.provider == ModelProvider.DEEPSEEK:
            self.api_key = getattr(self.config, 'DEEPSEEK_API_KEY', None) or os.getenv('DEEPSEEK_API_KEY')
        elif self.provider == ModelProvider.QWEN:
            self.api_key = getattr(self.config, 'QWEN_API_KEY', None) or os.getenv('QWEN_API_KEY')
        elif self.provider == ModelProvider.OPENAI:
            self.api_key = getattr(self.config, 'OPENAI_API_KEY', None) or os.getenv('OPENAI_API_KEY')
        elif self.provider == ModelProvider.MOONSHOT:
            self.api_key = getattr(self.config, 'MOONSHOT_API_KEY', None) or os.getenv('MOONSHOT_API_KEY')
        else:
            # 默认使用通用API_KEY
            self.api_key = getattr(self.config, 'API_KEY', None) or os.getenv('API_KEY')
        
        self.model_name = self.config.MODEL_NAME
        self.timeout = getattr(self.config, 'TIMEOUT', 30)
        self.temperature = getattr(self.config, 'TEMPERATURE', 0.1)
        self.max_tokens = getattr(self.config, 'MAX_TOKENS', 2048)
        
        # 重试配置
        self.max_retries = getattr(self.config, 'RETRY_MAX_ATTEMPTS', 3)
        self.backoff_factor = getattr(self.config, 'RETRY_BACKOFF_FACTOR', 1.5)
        self.status_forcelist = getattr(self.config, 'RETRY_STATUS_FORCELIST', [429, 500, 502, 503, 504])
        
        # 创建带重试机制的会话
        self.session = self._create_session()
        
        # 保存最近一次成功的响应作为备用
        self._last_successful_response = None
        
        # 健康状态标志
        self._is_healthy = True

    def _create_session(self) -> requests.Session:
        """创建带重试机制的requests会话"""
        session = requests.Session()
        retry = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=self.status_forcelist,
            allowed_methods=["POST"],  # 只对POST请求重试
            raise_on_status=False  # 不抛出异常，让调用者处理状态码
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    @abstractmethod
    def _get_headers(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def _get_api_url(self) -> str:
        pass

    @abstractmethod
    def _build_payload(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        pass

    def generate_response(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> Union[Optional[str], Generator[str, None, None]]:  # 流式返回生成器
        """生成回答，带重试机制"""
        try:
            messages = self._build_messages(prompt, context, history)
            
            # 记录开始时间
            start_time = time.time()
            
            if stream:
                return self._call_api_with_retry_stream(messages)
            else:
                response = self._call_api_with_retry(messages)
                
                # 记录响应时间
                response_time = time.time() - start_time
                logger.info(f"生成回答成功，长度: {len(response) if response else 0} 字符，响应时间: {response_time:.2f}秒")
                
                # 保存成功的响应
                if response:
                    self._last_successful_response = response
                    self._is_healthy = True
                
                return response
        except Exception as e:
            logger.error(f"生成回答异常: {str(e)}")
            traceback.print_exc()
            # 返回上次成功的响应作为后备
            if self._last_successful_response:
                logger.warning("使用上次成功的响应作为后备")
                return self._last_successful_response
            # 提供默认回复
            return self._get_default_response(prompt)

    def _get_default_response(self, prompt: str) -> str:
        """获取默认回复，当所有API调用都失败时"""
        self._is_healthy = False
        logger.warning(f"API服务暂时不可用，返回默认回复")
        
        # 根据查询类型返回不同的默认回复
        query_lower = prompt.lower()
        if any(word in query_lower for word in ["概述", "介绍", "总结", "主要内容"]):
            return "根据现有资料，无法回答该问题。当前服务暂时不可用，请稍后再试。"
        elif any(word in query_lower for word in ["表格", "费率", "价格", "费用"]):
            return "根据现有资料，无法回答该问题。当前服务暂时不可用，请稍后再试。"
        elif any(word in query_lower for word in ["流程", "步骤", "如何"]):
            return "根据现有资料，无法回答该问题。当前服务暂时不可用，请稍后再试。"
        else:
            return "根据现有资料，无法回答该问题。当前服务暂时不可用，请稍后再试。"

    def _build_messages(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = [{"role": "system", "content": global_config.SYSTEM_PROMPT}]

        if history:
            messages.extend(history)

        processed_query = preprocess_query(prompt)

        if context:
            enhanced_contexts = []
            for i, ctx in enumerate(context):
                enhanced_ctx = enhance_context_with_metadata(ctx)
                enhanced_contexts.append(f"【上下文 {i+1}】\n{enhanced_ctx}")

            context_text = "\n\n".join(enhanced_contexts)
            user_content = f"{context_text}\n\n---\n\n【用户问题】\n{processed_query}"
        else:
            user_content = processed_query

        messages.append({"role": "user", "content": user_content})
        return messages

    def _call_api_with_retry(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """带重试机制的API调用（非流式）"""
        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            try:
                response = self._call_api(messages)
                if response and "choices" in response and len(response["choices"]) > 0:
                    content = response["choices"][0]["message"]["content"]
                    logger.info(f"API调用成功 (尝试 {attempt}/{self.max_retries})")
                    return content
                else:
                    if attempt < self.max_retries:
                        wait_time = self.backoff_factor ** (attempt - 1) + 0.5 * (attempt - 1)
                        logger.warning(f"API响应格式异常，{wait_time:.2f}秒后重试 (尝试 {attempt}/{self.max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"API调用失败，已达到最大重试次数 ({self.max_retries})")
                        return None
            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor ** (attempt - 1) + 0.5 * (attempt - 1)
                    logger.warning(f"API调用异常: {str(e)}, {wait_time:.2f}秒后重试 (尝试 {attempt}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API调用异常，已达到最大重试次数 ({self.max_retries}): {str(e)}")
                    traceback.print_exc()
                    return None
        return None

    def _call_api_with_retry_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """带重试机制的流式API调用"""
        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            try:
                logger.info(f"开始流式API调用 (尝试 {attempt}/{self.max_retries})")
                # 获取流式生成器
                stream_generator = self._call_api_stream(messages)
                
                # 如果生成器存在，则逐块yield内容
                if stream_generator:
                    for chunk in stream_generator:
                        yield chunk
                    # 如果成功yield完所有内容，返回
                    logger.info(f"流式API调用成功完成 (尝试 {attempt}/{self.max_retries})")
                    return
                
            except Exception as e:
                logger.error(f"流式API调用异常: {str(e)}")
                traceback.print_exc()
            
            # 如果不是最后一次尝试，等待并重试
            if attempt < self.max_retries:
                wait_time = self.backoff_factor ** (attempt - 1) + 0.5 * (attempt - 1)
                logger.warning(f"流式API调用失败，{wait_time:.2f}秒后重试 (尝试 {attempt}/{self.max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"流式API调用失败，已达到最大重试次数 ({self.max_retries})")
                # 提供默认回复
                yield "很抱歉，我暂时无法为您提供该信息。请稍后再试。"

    def validate_api_key(self) -> bool:
        """验证 API 密钥是否有效"""
        if not self.api_key:
            return False
        test_messages = [{"role": "user", "content": "Hello"}]
        try:
            resp = self._call_api(test_messages)
            return resp is not None
        except Exception:
            return False

    def is_healthy(self) -> bool:
        """检查API服务是否健康"""
        return self._is_healthy

    def refresh_client(self):
        """刷新客户端实例，重新从工厂创建"""
        # 重新导入最新的全局配置，确保使用最新的模型设置
        from src.config import global_config
        # 创建新的客户端实例
        new_client = LLMClientFactory.create_client(global_config)
        
        # 直接更新全局客户端实例，避免模块导入问题
        import src.llm_client
        src.llm_client.llm_client = new_client
        
        # 同时更新可能引用了该客户端的其他模块
        try:
            import src.web_interface
            if hasattr(src.web_interface, 'llm_client'):
                src.web_interface.llm_client = new_client
        except ImportError:
            pass  # 如果web_interface模块不可用，忽略错误
        
        # 特别更新RAGPipeline中的客户端实例，因为它在初始化时保存了对旧客户端的引用
        try:
            from src.rag_pipeline import rag_pipeline
            if hasattr(rag_pipeline, 'llm_client'):
                rag_pipeline.llm_client = new_client
                logger.info("已成功更新RAGPipeline中的客户端实例")
        except ImportError:
            pass  # 如果rag_pipeline模块不可用，忽略错误
        
        logger.info(f"客户端已刷新: 新提供商={new_client.provider}, 新模型={new_client.model_name}")

    def _call_api(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """执行实际的API调用"""
        try:
            # 在调用模型前打印日志，显示使用的模型和模型key的配置情况
            logger.info(f"准备调用模型 - 提供商: {self.provider}, 模型名称: {self.model_name}")
            logger.info(f"API密钥状态: {'已配置' if self.api_key else '未配置'}")
            
            url = self._get_api_url()
            headers = self._get_headers()
            payload = self._build_payload(messages, stream=False)
            
            logger.debug(f"API调用: URL={url}, 消息数量={len(messages)}")
            
            # 发送请求
            response = self.session.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=self.timeout
            )
            
            # 检查响应状态
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API调用失败: 状态码 {response.status_code}")
                logger.error(f"响应内容: {response.text}")
                return None
        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            traceback.print_exc()
            return None

    def _call_api_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """执行流式API调用"""
        try:
            url = self._get_api_url()
            headers = self._get_headers()
            payload = self._build_payload(messages, stream=True)
            
            logger.debug(f"流式API调用: URL={url}, 消息数量={len(messages)}")
            
            # 发送流式请求
            with self.session.post(
                url, 
                headers=headers, 
                json=payload, 
                stream=True, 
                timeout=self.timeout
            ) as response:
                if response.status_code == 200:
                    # 处理流式响应
                    for chunk in response.iter_lines():
                        if chunk:
                            # 移除 'data: ' 前缀并解析JSON
                            try:
                                chunk_str = chunk.decode('utf-8')
                                if chunk_str.startswith('data: '):
                                    chunk_str = chunk_str[6:]
                                if chunk_str.strip() == '[DONE]':
                                    break
                                data = json.loads(chunk_str)
                                if 'choices' in data and data['choices']:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                logger.warning(f"无法解析流式响应块: {chunk_str}")
                else:
                    logger.error(f"流式API调用失败: 状态码 {response.status_code}")
                    logger.error(f"响应内容: {response.text}")
        except Exception as e:
            logger.error(f"流式API调用异常: {str(e)}")
            traceback.print_exc()

# ========================
# 🚀 DeepSeek 客户端实现
# ========================
class DeepSeekClient(BaseLLMClient):
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _get_api_url(self) -> str:
        return getattr(self.config, 'MODEL_URL', "https://api.deepseek.com/v1/chat/completions")

    def _build_payload(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream
        }


# ========================
# 🧩 OpenAI / 兼容客户端
# ========================
class OpenAIClient(BaseLLMClient):
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _get_api_url(self) -> str:
        return getattr(self.config, 'MODEL_URL', "https://api.openai.com/v1/chat/completions")

    def _build_payload(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream
        }


# ========================
# 🌙 Moonshot (月之暗面) 客户端
# ========================
class MoonshotClient(BaseLLMClient):
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _get_api_url(self) -> str:
        return getattr(self.config, 'MODEL_URL', "https://api.moonshot.cn/v1/chat/completions")

    def _build_payload(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream
        }

# ========================
# 🎯 Qwen (通义千问) HTTP客户端 —— 最终修复版
# ========================
class QwenClient(BaseLLMClient):

    def __init__(self, config=None):
        # 确保使用传入的配置，不使用默认参数以避免引用旧配置
        if config is None:
            from src.config import global_config
            config = global_config
            
        super().__init__(config)
        
        # 父类已经处理了API密钥的获取，但我们再确认一次
        if not self.api_key:
            logger.warning("未设置 QWEN_API_KEY，调用可能失败！")

    
    def _get_headers(self) -> Dict[str, str]:

        return {

            "Content-Type": "application/json",

            "Authorization": f"Bearer {self.api_key}"

        }

    
    def _get_api_url(self) -> str:

        # 使用 DashScope 兼容 OpenAI 的接口

        return getattr(self.config, 'MODEL_URL', "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

    
    def _build_payload(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:

        # 使用 OpenAI 兼容格式

        return {

            "model": self.model_name,  # 如 "qwen-plus", "qwen-turbo", "qwen-max"

            "messages": messages,

            "temperature": self.temperature,

            "max_tokens": self.max_tokens,

            "stream": stream

        }

# ========================
# 🎯 工厂类：根据配置创建客户端
# ========================
class LLMClientFactory:
    @staticmethod
    def create_client(config=None) -> BaseLLMClient:
        config = config or global_config
        provider = config.MODEL_PROVIDER.lower()

        client_map = {
            "deepseek": DeepSeekClient,
            "openai": OpenAIClient,
            "moonshot": MoonshotClient,
            "qwen": QwenClient,
            # "qwen_dashscope": QwenDashScopeClient,  # 暂未实现
            # 可继续扩展
        }

        if provider in client_map:
            return client_map[provider](config)
        else:
            logger.warning(f"未知模型提供商: {provider}，默认使用 DeepSeekClient")
            return DeepSeekClient(config)

# 创建LLM客户端实例
llm_client = LLMClientFactory.create_client()