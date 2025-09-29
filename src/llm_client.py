# llm_client.py
# 通用大模型客户端，支持 DeepSeek / OpenAI / Qwen / Moonshot / SparkDesk 等
# 支持严格模式、结构化输出、元数据增强、意图识别

import os
import re
import logging
import requests
import time
import traceback
import random
import hashlib
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from requests.adapters import Retry, HTTPAdapter
import json
from typing import List, Dict, Any, Optional, Union, Generator, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, ValidationError

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

# 添加更详细的日志记录器
api_logger = logging.getLogger('llm_client.api')
api_logger.setLevel(logging.INFO)

# 定义响应缓存类
class ResponseCache:
    """LLM响应缓存"""
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl  # 缓存有效期（秒）
        
    def _get_cache_key(self, prompt: str, context: Optional[List[str]] = None, history: Optional[List[Dict[str, str]]] = None) -> str:
        """生成缓存键"""
        combined = prompt
        if context:
            combined += '|||'.join(context)
        if history:
            combined += '|||'.join([f"{h['role']}:{h['content']}" for h in history])
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[str]:
        """获取缓存项"""
        if key in self.cache:
            item = self.cache[key]
            # 检查是否过期
            if datetime.now() < item['expires_at']:
                api_logger.debug(f"缓存命中: {key[:10]}...")
                return item['response']
            else:
                api_logger.debug(f"缓存过期: {key[:10]}...")
                del self.cache[key]  # 清理过期缓存
        return None
    
    def set(self, key: str, response: str):
        """设置缓存项"""
        # 如果缓存已满，移除最旧的项
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            api_logger.debug(f"缓存已满，移除最旧项")
            
        # 设置缓存项
        self.cache[key] = {
            'response': response,
            'expires_at': datetime.now() + timedelta(seconds=self.ttl)
        }
        api_logger.debug(f"缓存设置: {key[:10]}...")
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        api_logger.debug("缓存已清空")

# 创建全局缓存实例
response_cache = ResponseCache()

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
        # 日期允许更宽松的格式，以匹配“28th August, 2025”等
        'date': r'(Date|日期)[\s:]+([^\n]+)',
        'author': r'(Author|作者)[\s:]+([^\n]+)',
        'prepared_by': r'(Prepared by|Prepared)[\s:]+([^\n]+)',
        'release': r'(Release|发布)[\s:]+([^\n]+)',
        'release_date': r'(Release Date)[\s:]+([^\n]+)',
    }

    # 提取元数据
    for key, pattern in patterns.items():
        for line in lines[:50]:  # 只在前50行查找
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                # 关键修复：使用匹配值组而不是标签组
                # group(1) 是标签（如 Author/作者），group(2) 才是实际值
                value_group_index = 2
                # 对非数值类的提取做清理
                value = match.group(value_group_index).strip()
                # 移除可能的尾部标点或多余空格
                value = re.sub(r'[\s\-:]+$', '', value)
                metadata[key] = value

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

    # 规范化：将“Prepared by”映射为作者；将发布日期映射为创建时间
    if 'author' not in metadata and 'prepared_by' in metadata:
        metadata['author'] = metadata['prepared_by']
    if 'created' not in metadata:
        for tkey in ['release_date', 'release', 'date']:
            if tkey in metadata:
                metadata['created'] = metadata[tkey]
                break

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
        # 允许将超时配置为二元组 (connect_timeout, read_timeout) 以提高稳健性
        # 默认采用较宽松的设置以降低网络波动导致的连接重置
        self.timeout = getattr(self.config, 'TIMEOUT', (10, 60))
        self.temperature = getattr(self.config, 'TEMPERATURE', 0.1)
        self.max_tokens = getattr(self.config, 'MAX_TOKENS', 2048)
        
        # 重试配置
        self.max_retries = getattr(self.config, 'RETRY_MAX_ATTEMPTS', 3)
        self.backoff_factor = getattr(self.config, 'RETRY_BACKOFF_FACTOR', 1.5)
        self.status_forcelist = getattr(self.config, 'RETRY_STATUS_FORCELIST', [429, 500, 502, 503, 504])
        
        # 负载均衡配置
        self.api_urls = getattr(self.config, f'{self.provider.upper()}_API_URLS', [self._get_default_api_url()])
        self.current_url_index = 0
        
        # 限流配置
        self.rate_limit_per_minute = getattr(self.config, f'{self.provider.upper()}_RATE_LIMIT', 60)
        self.request_timestamps = []
        
        # 故障转移配置
        self.failover_enabled = getattr(self.config, 'FAILOVER_ENABLED', True)
        self.failover_providers = getattr(self.config, 'FAILOVER_PROVIDERS', [])
        
        # 创建带重试机制的会话
        self.session = self._create_session()
        
        # 保存最近一次成功的响应作为备用
        self._last_successful_response = None
        
        # 健康状态标志
        self._is_healthy = True
        
        # 响应计数器
        self.response_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'total_tokens_used': 0
        }
        
        # 结构化输出配置
        self.structured_output_enabled = getattr(self.config, 'STRUCTURED_OUTPUT_ENABLED', False)
        self.structured_output_schema = getattr(self.config, 'STRUCTURED_OUTPUT_SCHEMA', None)

    def _get_default_api_url(self) -> str:
        """获取默认API URL"""
        urls = {
            ModelProvider.DEEPSEEK: "https://api.deepseek.com/v1/chat/completions",
            ModelProvider.OPENAI: "https://api.openai.com/v1/chat/completions",
            ModelProvider.MOONSHOT: "https://api.moonshot.cn/v1/chat/completions",
            ModelProvider.QWEN: "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        }
        return urls.get(self.provider, urls[ModelProvider.DEEPSEEK])
    
    def _create_session(self) -> requests.Session:
        """创建带重试机制的requests会话"""
        session = requests.Session()
        # 智能重试策略：指数退避 + 随机抖动
        retry = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=self.status_forcelist,
            allowed_methods=["POST"],  # 只对POST请求重试
            raise_on_status=False,  # 不抛出异常，让调用者处理状态码
            respect_retry_after_header=True  # 尊重Retry-After头部
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20, pool_block=False)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        # 添加请求/响应钩子用于日志记录
        session.hooks['response'].append(self._response_hook)
        
        return session
    
    def _response_hook(self, response, *args, **kwargs):
        """响应钩子，用于记录详细的API调用信息"""
        if hasattr(response, 'request'):
            request = response.request
            api_logger.debug(f"API调用: {request.method} {request.url}")
            api_logger.debug(f"状态码: {response.status_code}")
            
            # 记录请求体大小（避免记录敏感信息）
            if request.body:
                api_logger.debug(f"请求体大小: {len(request.body)} 字节")
                
            # 记录响应体大小
            response_size = len(response.content) if response.content else 0
            api_logger.debug(f"响应体大小: {response_size} 字节")
    
    def _check_rate_limit(self):
        """检查并应用速率限制"""
        current_time = time.time()
        # 移除1分钟前的时间戳
        self.request_timestamps = [t for t in self.request_timestamps if current_time - t < 60]
        
        # 如果达到速率限制，等待
        if len(self.request_timestamps) >= self.rate_limit_per_minute:
            wait_time = 60 - (current_time - self.request_timestamps[0])
            if wait_time > 0:
                api_logger.info(f"达到速率限制，等待 {wait_time:.2f} 秒")
                time.sleep(wait_time)
        
        # 记录当前请求时间
        self.request_timestamps.append(current_time)
    
    def _get_next_api_url(self) -> str:
        """获取下一个API URL（轮询负载均衡）"""
        url = self.api_urls[self.current_url_index]
        # 更新索引用于下次调用
        self.current_url_index = (self.current_url_index + 1) % len(self.api_urls)
        return url
    
    def _validate_structured_response(self, response: str, schema: Optional[BaseModel] = None) -> Tuple[bool, Any]:
        """验证结构化响应是否符合schema"""
        if not schema:
            return True, response
        
        try:
            # 尝试从响应中提取JSON
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                # 使用Pydantic验证
                validated_data = schema(**data)
                return True, validated_data.model_dump()
            else:
                # 尝试直接解析整个响应
                data = json.loads(response)
                validated_data = schema(**data)
                return True, validated_data.model_dump()
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            api_logger.error(f"结构化响应验证失败: {str(e)}")
            return False, None
    
    def _perform_failover(self) -> bool:
        """执行故障转移到备用提供商"""
        if not self.failover_enabled or not self.failover_providers:
            api_logger.warning("故障转移未启用或未配置备用提供商")
            return False
        
        # 获取下一个可用的提供商
        for provider_name in self.failover_providers:
            try:
                provider = ModelProvider(provider_name.lower())
                api_logger.info(f"尝试故障转移到提供商: {provider}")
                
                # 更新当前提供商配置
                self.provider = provider
                # 获取新提供商的API密钥
                if provider == ModelProvider.DEEPSEEK:
                    self.api_key = getattr(self.config, 'DEEPSEEK_API_KEY', None) or os.getenv('DEEPSEEK_API_KEY')
                elif provider == ModelProvider.QWEN:
                    self.api_key = getattr(self.config, 'QWEN_API_KEY', None) or os.getenv('QWEN_API_KEY')
                elif provider == ModelProvider.OPENAI:
                    self.api_key = getattr(self.config, 'OPENAI_API_KEY', None) or os.getenv('OPENAI_API_KEY')
                elif provider == ModelProvider.MOONSHOT:
                    self.api_key = getattr(self.config, 'MOONSHOT_API_KEY', None) or os.getenv('MOONSHOT_API_KEY')
                
                # 验证新提供商是否可用
                if self.validate_api_key():
                    api_logger.info(f"故障转移成功: {provider}")
                    return True
            except Exception as e:
                api_logger.error(f"故障转移到 {provider_name} 失败: {str(e)}")
                
        return False

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
        stream: bool = False,
        cache_enabled: bool = True,
        structured_schema: Optional[BaseModel] = None
    ) -> Union[Optional[str], Dict[str, Any], Generator[str, None, None]]:  # 流式返回生成器
        """生成回答，带重试机制、缓存和结构化输出验证"""
        # 更新请求计数
        self.response_stats['total_requests'] += 1
        
        # 非流式请求且启用缓存时，尝试从缓存获取
        if not stream and cache_enabled:
            cache_key = response_cache._get_cache_key(prompt, context, history)
            cached_response = response_cache.get(cache_key)
            if cached_response:
                self.response_stats['cache_hits'] += 1
                api_logger.info(f"从缓存返回响应，缓存命中率: {self._get_cache_hit_rate():.2f}%")
                
                # 如果需要结构化输出，验证缓存的响应
                if structured_schema:
                    is_valid, validated_data = self._validate_structured_response(cached_response, structured_schema)
                    if is_valid:
                        return validated_data
                    # 如果缓存的响应不符合schema，继续获取新响应
                    api_logger.warning("缓存的响应不符合schema，获取新响应")
                else:
                    return cached_response
        
        try:
            # 检查速率限制
            self._check_rate_limit()
            
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
                
                # 更新成功计数
                if response:
                    self.response_stats['successful_requests'] += 1
                    self._last_successful_response = response
                    self._is_healthy = True
                    
                    # 如果启用缓存，且响应质量达标，保存到缓存
                    if cache_enabled and self._should_cache_response(response):
                        cache_key = response_cache._get_cache_key(prompt, context, history)
                        response_cache.set(cache_key, response)
                    
                    # 处理结构化输出验证
                    if structured_schema or self.structured_output_enabled:
                        schema = structured_schema or self.structured_output_schema
                        if schema:
                            is_valid, validated_data = self._validate_structured_response(response, schema)
                            if is_valid:
                                return validated_data
                            else:
                                # 验证失败，尝试再次生成
                                api_logger.warning("结构化输出验证失败，尝试重新生成")
                                return self.generate_response(prompt, context, history, stream, False, structured_schema)
                
                return response
        except Exception as e:
            logger.error(f"生成回答异常: {str(e)}")
            traceback.print_exc()
            
            # 更新失败计数
            self.response_stats['failed_requests'] += 1
            
            # 尝试故障转移
            if not stream and self.failover_enabled and self._perform_failover():
                api_logger.info("故障转移后重新尝试请求")
                return self.generate_response(prompt, context, history, stream, cache_enabled, structured_schema)
            
            # 返回上次成功的响应作为后备
            if self._last_successful_response:
                logger.warning("使用上次成功的响应作为后备")
                return self._last_successful_response
            
            # 提供默认回复
            default_response = self._get_default_response(prompt)
            # 如果启用结构化输出，为默认回复创建结构化格式
            if structured_schema:
                try:
                    # 尝试创建一个符合schema的最小响应
                    return {"error": True, "message": default_response}
                except:
                    pass
            return default_response

    def _should_cache_response(self, response_text: str) -> bool:
        """判断响应是否适合写入缓存，避免缓存通用拒答或过短内容"""
        try:
            text = (response_text or "").strip()
            if len(text) < 30:
                api_logger.info("响应过短，不写入缓存")
                return False
            generic_patterns = [
                "无法回答", "抱歉", "我不知道", "暂时不可用", "稍后再试",
                "不能提供", "无权访问"
            ]
            if any(pat in text for pat in generic_patterns):
                api_logger.info("检测到通用拒答语句，不写入缓存")
                return False
            return True
        except Exception:
            return True
    
    def _get_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        total = self.response_stats['total_requests']
        if total == 0:
            return 0.0
        return (self.response_stats['cache_hits'] / total) * 100
    
    def get_response_stats(self) -> Dict[str, int]:
        """获取响应统计信息"""
        return self.response_stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.response_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'total_tokens_used': 0
        }

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
            
    def generate_streaming_response(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        cache_enabled: bool = False  # 流式响应默认不使用缓存
    ) -> Generator[str, None, None]:
        """生成流式回答，提供对流式响应的直接访问"""
        # 确保stream参数为True
        return self.generate_response(prompt, context, history, stream=True, cache_enabled=cache_enabled)
        
    def generate_batched_response(
        self,
        prompts: List[str],
        contexts: Optional[List[List[str]]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        batch_size: int = 5
    ) -> List[Optional[str]]:
        """批量生成回答，优化多个相似查询的处理"""
        results = []
        
        # 如果没有提供contexts，为每个prompt创建空context
        if contexts is None:
            contexts = [None] * len(prompts)
            
        # 确保contexts和prompts长度一致
        assert len(prompts) == len(contexts), "prompts和contexts长度必须一致"
        
        # 批量处理
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_contexts = contexts[i:i+batch_size]
            
            # 为每个查询生成回答
            for j, (prompt, context) in enumerate(zip(batch_prompts, batch_contexts)):
                try:
                    result = self.generate_response(prompt, context, history, stream=False)
                    results.append(result)
                except Exception as e:
                    logger.error(f"批量处理第{i+j}个查询失败: {str(e)}")
                    results.append(None)
            
            # 避免触发API速率限制
            if i + batch_size < len(prompts):
                time.sleep(1)  # 每个批次之间等待1秒
        
        return results

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
                # 智能退避策略：指数退避 + 随机抖动
                if attempt > 1:
                    base_wait = self.backoff_factor ** (attempt - 1)
                    # 添加10-30%的随机抖动，避免多个请求同时重试
                    jitter = random.uniform(0.1, 0.3)
                    wait_time = base_wait * (1 + jitter)
                    api_logger.info(f"等待 {wait_time:.2f} 秒后重试 (尝试 {attempt}/{self.max_retries})")
                    time.sleep(wait_time)
                
                response = self._call_api(messages)
                
                # 处理特殊错误码
                if response:
                    # 提取token使用信息（如果有）
                    if 'usage' in response and 'total_tokens' in response['usage']:
                        self.response_stats['total_tokens_used'] += response['usage']['total_tokens']
                    
                    if "choices" in response and len(response["choices"]) > 0:
                        content = response["choices"][0]["message"]["content"]
                        api_logger.info(f"API调用成功 (尝试 {attempt}/{self.max_retries})")
                        return content
                    else:
                        api_logger.warning(f"API响应缺少choices字段")
                else:
                    api_logger.warning(f"API返回空响应")
                    
            except requests.exceptions.Timeout:
                api_logger.error(f"API请求超时 (尝试 {attempt}/{self.max_retries})")
            except requests.exceptions.ConnectionError:
                api_logger.error(f"API连接错误 (尝试 {attempt}/{self.max_retries})")
            except requests.exceptions.RequestException as e:
                api_logger.error(f"API请求异常 (尝试 {attempt}/{self.max_retries}): {str(e)}")
            except Exception as e:
                api_logger.error(f"未预期的异常 (尝试 {attempt}/{self.max_retries}): {str(e)}")
                traceback.print_exc()
        
        api_logger.error(f"API调用失败，已达到最大重试次数 ({self.max_retries})")
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
                    chunk_count = 0
                    for chunk in stream_generator:
                        if chunk:
                            yield chunk
                            chunk_count += 1
                    # 如果成功yield完所有内容，返回
                    logger.info(f"流式API调用成功完成 (尝试 {attempt}/{self.max_retries}, 返回{chunk_count}个块)")
                    return
                
            except requests.exceptions.Timeout:
                logger.error(f"流式API请求超时 (尝试 {attempt}/{self.max_retries})")
                # 尝试生成一个超时提示，然后继续重试
                if attempt == 1:
                    yield "[正在连接...]"
            except requests.exceptions.ConnectionError:
                logger.error(f"流式API连接错误 (尝试 {attempt}/{self.max_retries})")
            except requests.exceptions.RequestException as e:
                logger.error(f"流式API请求异常 (尝试 {attempt}/{self.max_retries}): {str(e)}")
            except Exception as e:
                logger.error(f"流式API调用异常 (尝试 {attempt}/{self.max_retries}): {str(e)}")
                traceback.print_exc()
            
            # 如果不是最后一次尝试，等待并重试
            if attempt < self.max_retries:
                # 智能退避策略：指数退避 + 随机抖动
                base_wait = self.backoff_factor ** (attempt - 1)
                jitter = random.uniform(0.1, 0.3)
                wait_time = base_wait * (1 + jitter)
                logger.warning(f"流式API调用失败，{wait_time:.2f}秒后重试 (尝试 {attempt}/{self.max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"流式API调用失败，已达到最大重试次数 ({self.max_retries})")
                # 提供默认回复
                yield "\n很抱歉，我暂时无法为您提供该信息。请稍后再试。"

    def validate_api_key(self) -> bool:
        """验证 API 密钥是否有效"""
        if not self.api_key:
            api_logger.warning("API密钥未设置")
            return False
        
        test_messages = [{"role": "user", "content": "请返回'验证成功'作为响应"}]
        try:
            # 临时降低超时时间以加快验证
            original_timeout = self.timeout
            self.timeout = 10
            resp = self._call_api(test_messages)
            self.timeout = original_timeout
            
            # 检查响应是否有效
            if resp and "choices" in resp and len(resp["choices"]) > 0:
                content = resp["choices"][0]["message"]["content"]
                # 验证响应内容
                return "验证成功" in content or content.strip() != ""
            return False
        except Exception as e:
            api_logger.error(f"API密钥验证失败: {str(e)}")
            # 恢复原始超时设置
            try:
                self.timeout = original_timeout
            except:
                pass
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
                logger.info("已成功更新web_interface中的客户端实例")
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
        
        # 特别更新AdaptiveRAGPipeline中的客户端实例，因为它在初始化时也保存了对旧客户端的引用
        try:
            from src.adaptive_rag_pipeline import adaptive_rag_pipeline
            if hasattr(adaptive_rag_pipeline, 'llm_client'):
                adaptive_rag_pipeline.llm_client = new_client
                logger.info("已成功更新AdaptiveRAGPipeline中的客户端实例")
        except ImportError:
            pass  # 如果adaptive_rag_pipeline模块不可用，忽略错误
            
        # 特别更新Agent中的客户端实例，因为它也需要使用最新的LLM客户端
        try:
            from src.agent import agent
            if hasattr(agent, 'llm_client'):
                agent.llm_client = new_client
                logger.info("已成功更新Agent中的客户端实例")
        except ImportError:
            pass  # 如果agent模块不可用，忽略错误
        
        logger.info(f"客户端已刷新: 新提供商={new_client.provider}, 新模型={new_client.model_name}")
        
        # 返回新的客户端实例，以便调用者可以直接使用
        return new_client

    def _call_api(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """执行实际的API调用"""
        try:
            # 在调用模型前打印日志，显示使用的模型和模型key的配置情况
            api_logger.info(f"准备调用模型 - 提供商: {self.provider}, 模型名称: {self.model_name}")
            api_logger.info(f"API密钥状态: {'已配置' if self.api_key else '未配置'}")
            
            # 使用负载均衡获取URL
            url = self._get_next_api_url()
            headers = self._get_headers()
            payload = self._build_payload(messages, stream=False)
            
            # 敏感信息屏蔽
            sanitized_payload = payload.copy()
            if 'messages' in sanitized_payload:
                sanitized_payload['messages'] = [
                    {k: '(内容已屏蔽)' if k == 'content' else v for k, v in msg.items()}
                    for msg in sanitized_payload['messages']
                ]
            
            api_logger.debug(f"API调用: URL={url}, 消息数量={len(messages)}")
            api_logger.debug(f"请求参数: {json.dumps(sanitized_payload, ensure_ascii=False, indent=2)}")
            
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
            elif response.status_code == 429:
                # 速率限制处理
                api_logger.warning(f"API速率限制: 状态码 {response.status_code}")
                # 检查是否有Retry-After头部
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    wait_time = int(retry_after)
                    api_logger.info(f"根据Retry-After头部，等待 {wait_time} 秒")
                    time.sleep(wait_time)
                return None
            elif response.status_code == 401:
                # 认证错误
                api_logger.error(f"API认证错误: 状态码 {response.status_code}")
                api_logger.error(f"请检查API密钥是否正确")
                return None
            elif response.status_code >= 500:
                # 服务器错误
                api_logger.error(f"API服务器错误: 状态码 {response.status_code}")
                api_logger.error(f"响应内容: {response.text[:500]}..." if len(response.text) > 500 else response.text)
                return None
            else:
                api_logger.error(f"API调用失败: 状态码 {response.status_code}")
                api_logger.error(f"响应内容: {response.text[:500]}..." if len(response.text) > 500 else response.text)
                return None
        except requests.exceptions.Timeout:
            api_logger.error(f"API请求超时")
            raise
        except requests.exceptions.ConnectionError:
            api_logger.error(f"API连接错误")
            raise
        except Exception as e:
            api_logger.error(f"API调用异常: {str(e)}")
            traceback.print_exc()
            raise
        return None

    def _call_api_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """执行流式API调用，支持更健壮的流式响应处理"""
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
                    buffer = ""
                    for chunk in response.iter_lines():
                        if chunk:
                            # 移除 'data: ' 前缀并解析JSON
                            try:
                                chunk_str = chunk.decode('utf-8')
                                # 处理可能的多个数据块
                                parts = chunk_str.split('data: ')
                                for part in parts:
                                    part = part.strip()
                                    if part and part != '[DONE]':
                                        try:
                                            data = json.loads(part)
                                            if 'choices' in data and data['choices']:
                                                delta = data['choices'][0].get('delta', {})
                                                if 'content' in delta:
                                                    # 使用更大的块来优化输出流
                                                    buffer += delta['content']
                                                    if len(buffer) >= getattr(self.config, 'STREAMING_CHUNK_SIZE', 50):
                                                        yield buffer
                                                        buffer = ""
                                        except json.JSONDecodeError:
                                            logger.warning(f"无法解析流式响应块: {part}")
                                
                                # 检查是否到达流的末尾
                                if chunk_str.strip() == '[DONE]':
                                    # 输出缓冲区中的剩余内容
                                    if buffer:
                                        yield buffer
                                    break
                            except Exception as e:
                                logger.warning(f"处理流式响应块时出错: {str(e)}")
                    
                    # 确保缓冲区中的所有内容都被输出
                    if buffer:
                        yield buffer
                    
                    # 输出一个空字符串以确保流的正确结束
                    yield ""
                    
                elif response.status_code == 429:
                    # 速率限制处理
                    logger.warning(f"流式API速率限制: 状态码 {response.status_code}")
                    yield "[系统繁忙，请稍后再试]"
                elif response.status_code == 401:
                    # 认证错误
                    logger.error(f"流式API认证错误: 状态码 {response.status_code}")
                    yield "[认证失败，请检查API密钥]"
                else:
                    logger.error(f"流式API调用失败: 状态码 {response.status_code}")
                    logger.error(f"响应内容: {response.text[:500]}..." if len(response.text) > 500 else response.text)
                    yield "[API调用失败]"
        except requests.exceptions.Timeout:
            logger.error(f"流式API请求超时")
            raise
        except requests.exceptions.ConnectionError:
            logger.error(f"流式API连接错误")
            raise
        except Exception as e:
            logger.error(f"流式API调用异常: {str(e)}")
            traceback.print_exc()
            raise

# ========================
# 🚀 DeepSeek 客户端实现
# ========================
class DeepSeekClient(BaseLLMClient):
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "close",
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
            "Accept": "application/json",
            "Connection": "close",
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
            "Accept": "application/json",
            "Connection": "close",
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
            "Accept": "application/json",
            "Connection": "close",
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

# ========================
# 🔄 全局刷新函数（提供给其他模块调用）
# ========================
def refresh_client(target_module: Optional[str] = None, config=None) -> bool:
    """全局刷新LLM客户端实例并同步到相关模块。

    兼容不同调用方式，例如在 web_interface 或 agent 中直接调用。
    可忽略传入参数，仅用于统一接口，避免 NameError。
    """
    try:
        # 优先使用实例方法的刷新逻辑，已包含模块同步更新
        try:
            llm_client.refresh_client()
            return True
        except Exception:
            # 回退方案：手动创建并替换全局客户端，再同步到相关模块
            from src.config import global_config as _global_config
            new_client = LLMClientFactory.create_client(config or _global_config)

            import src.llm_client as _lc
            _lc.llm_client = new_client

            # 同步更新 web_interface
            try:
                import src.web_interface as _wi
                if hasattr(_wi, 'llm_client'):
                    _wi.llm_client = new_client
            except Exception:
                pass

            # 同步更新 rag_pipeline
            try:
                from src.rag_pipeline import rag_pipeline as _rp
                if hasattr(_rp, 'llm_client'):
                    _rp.llm_client = new_client
            except Exception:
                pass

            # 同步更新 adaptive_rag_pipeline
            try:
                from src.adaptive_rag_pipeline import adaptive_rag_pipeline as _arp
                if hasattr(_arp, 'llm_client'):
                    _arp.llm_client = new_client
            except Exception:
                pass

            # 同步更新 agent
            try:
                from src.agent import agent as _agent
                if hasattr(_agent, 'llm_client'):
                    _agent.llm_client = new_client
            except Exception:
                pass

            return True
    except Exception as e:
        logger.error(f"刷新LLM客户端失败: {str(e)}")
        return False