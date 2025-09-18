# llm_client.py
# 通用大模型客户端，支持 DeepSeek / OpenAI / Qwen / Moonshot / SparkDesk 等
# 支持严格模式、结构化输出、元数据增强、意图识别

import os
import re
import logging
import requests
import json
from typing import List, Dict, Any, Optional, Union
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
    global_config = MockConfig()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ========================
# 🧠 智能严格模式系统提示词（增强版）
# ========================
STRICT_SMART_SYSTEM_PROMPT = """你是一个严谨的智能文档助手，必须严格依据提供的上下文回答问题。

## 智能问题分类与响应策略
1. **概述类问题**（讲述/介绍/概述/总结/主要内容）：
   - 执行全面结构化概括，包含文档所有关键维度。
   - 遵循指定格式和章节顺序，不遗漏重要信息。
   - 详细描述主要功能、目的和核心内容。
   - 列出文档的主要章节。
   - 确保回答逻辑清晰、层次分明。

2. **细节类问题**（具体数据/流程/规则/配置）：
   - 精准定位并提取上下文相关信息。
   - 对于表格数据，保持原结构完整还原。
   - 对于流程步骤，使用数字列表清晰呈现。
   - 对于技术参数，准确引用原文术语和数值。

3. **无相关信息**：
   - 直接回答："根据现有资料，无法回答该问题。"
   - 不尝试编造或猜测任何内容。

## 强制回答格式规范
- **必须使用 Markdown 格式**，确保层次清晰。
- **必须包含以下章节**（若上下文存在对应信息）：
  - ### 📄 文档基本信息
    - 包含标题、版本、日期、作者、产品码等元数据。
    - 每个信息项使用独立行呈现，便于阅读。
    
  - ### 🎯 目的与范围
    - 明确文档的主要目标和适用场景。
    - 详细说明文档的核心用途和覆盖范围。
    
  - ### 📱 核心功能 / 用户可见行为
    - 详细描述文档中定义的主要功能。
    - 列出用户可感知的具体操作和交互行为。
    
  - ### ⚙️ 后端配置 / 系统要求
    - 说明系统配置参数、依赖项和环境要求。
    - 引用技术规范和实现细节。
    
  - ### 📊 数据结构 / 费用表格
    - 完整还原上下文包含的所有表格。
    - 保持表头、行列关系和数据的准确性。
    - 若有多个表格，分别标记序号和名称。
    
  - ### ✅ 总结
    - 简明扼要地总结文档核心价值和主要内容。
    - 突出文档的关键要点和应用前景。

- **禁止使用**：
  - 编造上下文未提及的细节。
  - 使用模糊不清的措辞（如"可能"、"大概"）。
  - 脱离上下文进行自由发挥或主观判断。
  
- **语言风格**：
  - 专业、准确、简洁。
  - 使用项目符号（-）或表格（|---|）组织内容。
  - 确保术语统一，与原文保持一致。

请严格遵循以上要求，开始你的回答："""


# ========================
# 🧩 查询预处理：识别“讲述/概述”意图
# ========================
def preprocess_query(query: str) -> str:
    """预处理用户查询，将不同类型问题转化为结构化指令"""
    if not isinstance(query, str):
        return query

    # 转换为小写以进行不区分大小写的匹配
    query_lower = query.lower()
    
    # 1. 概述类问题 - 增强结构化指令
    overview_triggers = [
        "讲述", "介绍", "概述", "总结", "主要内容", "是什么", "讲什么",
        "介绍一下", "说说", "简述", "文档内容", "这份文件", "这个资料",
        "describe", "summary", "overview", "what is", "explain"
    ]
    
    if any(trigger in query_lower for trigger in overview_triggers):
        return "请根据上下文，详细结构化概述本文档，必须包含：文档基本信息、目的与范围、核心功能、后端配置、数据结构、总结，并列出主要章节。使用标题+列表/表格格式，确保内容详尽完整。"
    
    # 2. 表格类问题 - 特殊处理表格数据
    table_triggers = [
        "表格", "费率", "价格", "费用", "收费", "数据结构", "字段",
        "table", "fee", "cost", "price", "structure", "fields"
    ]
    
    if any(trigger in query_lower for trigger in table_triggers):
        return f"请根据上下文，详细回答关于'{query}'的问题。如果涉及表格数据，请使用Markdown表格格式完整还原，确保表头和数据准确无误。"
    
    # 3. 流程类问题 - 特殊处理步骤和流程
    process_triggers = [
        "流程", "步骤", "如何", "操作", "申请", "处理", "步骤",
        "process", "steps", "how to", "apply", "operation"
    ]
    
    if any(trigger in query_lower for trigger in process_triggers):
        return f"请根据上下文，详细回答关于'{query}'的问题。如果涉及流程步骤，请使用数字列表清晰呈现每个步骤的具体内容和要求。"
    
    # 4. 配置类问题 - 特殊处理配置参数
    config_triggers = [
        "配置", "参数", "设置", "系统", "要求", "环境",
        "config", "parameter", "setting", "system", "requirement", "environment"
    ]
    
    if any(trigger in query_lower for trigger in config_triggers):
        return f"请根据上下文，详细回答关于'{query}'的问题。确保准确引用配置参数、系统要求和技术规范。"
    
    # 保持原始查询不变
    return query


# ========================
# 🧠 上下文增强器：提取元数据 + 章节 + 表格片段
# ========================
def enhance_context_with_metadata(raw_text: str, max_raw_lines=20) -> str:
    """
    从原始文本中提取结构化元信息，增强上下文，便于模型理解
    """
    if not raw_text:
        return ""

    lines = raw_text.splitlines()
    metadata = {}
    patterns = {
        'title': r'(?:文档[名称名]|标题|Title|Document Name|文件名)[\s:：]*(.+)',
        'version': r'(?:版本|Version|Ver|修订版)[\s:：]*([vV]?\d+\.\d+)',
        'date': r'(?:日期|Date|Release Date|发布日期)[\s:：]*(\d{1,2}[\/\-年月\s]*\d{1,2}[\/\-月日\s]*\d{4}|\d{4}年\d{1,2}月\d{1,2}日|\d{4}[\-\/]\d{1,2}[\-\/]\d{1,2})',
        'author': r'(?:作者|Prepared by|编写|Author|编制|撰写)[\s:：]*([^\n\r,，]+)',
        'doc_name': r'(?:Document Name|文档名称)[\s:：]*([^\n\r]+)',
        'purpose': r'(?:目的|Purpose|目标)[\s:：]*([^\n\r。]+)',
        'scope': r'(?:范围|Scope|适用范围)[\s:：]*([^\n\r。]+)',
        'product_code': r'(?:产品码|Product Code|产品编号)[\s:：]*([A-Z0-9]+)',
        'main_function': r'(?:主要功能|核心功能|功能概述)[\s:：]*([^\n\r。]+)',
        'core_content': r'(?:核心内容|主要内容|内容概要)[\s:：]*([^\n\r。]+)',
    }

    # 扫描前50行提取元信息
    for line in lines[:50]:
        line = line.strip()
        if not line: continue
        for key, pattern in patterns.items():
            if key in metadata: continue
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
        elif re.match(r'^\d+\.\s*[A-Z]', stripped) or re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+', stripped):
            current_section = None  # 新章节开始

        if current_section and len(key_sections[current_section]) < 5:
            key_sections[current_section].append(stripped)

    # 提取表格（最多提取两个表格，每个10行内）
    tables = []
    current_table = []
    table_start_patterns = [
        r'\|.*\|',
        r'-{3,}',
        r'Tenure|期限|Fee|费率|Cost|费用',
        r'天\s+费\s+用',
        r'字段名|Field|参数|Parameter',
    ]
    
    for line in lines[:300]:
        line_stripped = line.strip()
        
        # 检测表格开始
        if any(re.search(p, line_stripped, re.I) for p in table_start_patterns) and not current_table:
            current_table.append(line_stripped)
        # 继续收集表格内容
        elif current_table:
            if line_stripped and not re.match(r'^[\s\-|_]+$', line_stripped):
                current_table.append(line_stripped)
            # 表格结束条件
            if not line_stripped and len(current_table) > 1:
                tables.append(current_table)
                current_table = []
                if len(tables) >= 2:  # 最多提取两个表格
                    break
        # 表格行数限制
        if len(current_table) > 10:
            tables.append(current_table)
            current_table = []
            if len(tables) >= 2:
                break
    
    # 处理最后一个未完成的表格
    if current_table and len(current_table) > 1:
        tables.append(current_table)

    # 提取主要章节
    chapters = []
    chapter_patterns = [
        r'^\d+\.\s+[^\n\r]+',  # 1. 章节标题
        r'^\d+\.\d+\s+[^\n\r]+',  # 1.1 子章节标题
        r'^[A-Z]\.\s+[^\n\r]+',  # A. 章节标题
        r'^第[一二三四五六七八九十]+章\s+[^\n\r]+',  # 第一章 章节标题
        r'^[一二三四五六七八九十]+、\s+[^\n\r]+'  # 一、章节标题
    ]
    
    for line in lines[:200]:
        line_stripped = line.strip()
        if any(re.match(p, line_stripped) for p in chapter_patterns):
            chapters.append(line_stripped)
            if len(chapters) >= 10:  # 最多提取10个章节
                break

    # 构建增强上下文
    parts = ["【增强型文档上下文 —— 专为智能问答优化】"]

    # 文档基本信息
    if metadata:
        parts.append("\n## 📄 文档基本信息")
        label_map = {
            'title': '标题', 'version': '版本', 'date': '发布日期',
            'author': '作者', 'doc_name': '文档名称', 'product_code': '产品码'
        }
        for key, value in metadata.items():
            if key in label_map:
                parts.append(f"- **{label_map[key]}**: {value}")

    # 文档核心信息
    if any([metadata.get('purpose'), metadata.get('scope'), metadata.get('main_function'), metadata.get('core_content')]):
        parts.append("\n## 🎯 目的与核心信息")
        if metadata.get('purpose'):
            parts.append(f"- **目的**: {metadata['purpose']}")
        if metadata.get('scope'):
            parts.append(f"- **适用范围**: {metadata['scope']}")
        if metadata.get('main_function'):
            parts.append(f"- **主要功能**: {metadata['main_function']}")
        if metadata.get('core_content'):
            parts.append(f"- **核心内容**: {metadata['core_content']}")

    # 关键内容摘要
    if any(key_sections.values()):
        parts.append("\n## 📑 关键内容摘要")
        if key_sections['main_functions']:
            parts.append(f"- **主要功能**: " + " | ".join(key_sections['main_functions'][:3]))
        if key_sections['system_config']:
            parts.append(f"- **系统配置**: " + " | ".join(key_sections['system_config'][:3]))
        if key_sections['fee_structure']:
            parts.append(f"- **费用结构**: " + " | ".join(key_sections['fee_structure'][:3]))
        if key_sections['process_flow']:
            parts.append(f"- **流程步骤**: " + " | ".join(key_sections['process_flow'][:3]))
        if key_sections['data_structures']:
            parts.append(f"- **数据结构**: " + " | ".join(key_sections['data_structures'][:3]))

    # 表格数据
    if tables:
        for i, table_lines in enumerate(tables):
            table_name = f"表格{i+1}: {table_lines[0][:30]}..." if len(table_lines) > 0 else f"表格{i+1}"
            parts.append(f"\n## 📊 {table_name}")
            parts.append("```\n" + "\n".join(table_lines[:10]) + "\n```")

    # 主要章节列表
    if chapters:
        parts.append("\n## 📑 文档主要章节")
        for i, chapter in enumerate(chapters):
            parts.append(f"- **{chapter}**")

    # 原始内容开头（提供更多上下文）
    raw_snippet = "\n".join([line.strip() for line in lines[:max_raw_lines] if line.strip()])
    if raw_snippet:
        parts.append(f"\n## 📜 原始内容开头片段")
        parts.append(raw_snippet)

    return "\n".join(parts)


# ========================
# 🧭 模型提供商标识
# ========================
class ModelProvider(Enum):
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    QWEN = "qwen"
    QWEN_DASHSCOPE = "qwen_dashscope"
    MOONSHOT = "moonshot"
    SPARK = "spark"  # 讯飞星火


# ========================
# 🧱 模型客户端基类（抽象）
# ========================
class BaseLLMClient(ABC):
    """LLM 客户端抽象基类"""

    def __init__(self, config=None):
        self.config = config or global_config
        self.provider = ModelProvider(self.config.MODEL_PROVIDER.lower())
        # 使用getattr确保即使配置项不存在也不会抛出异常
        self.api_key = getattr(self.config, 'DEEPSEEK_API_KEY', None) or getattr(self.config, 'API_KEY', None)
        self.model_name = self.config.MODEL_NAME
        self.timeout = getattr(self.config, 'TIMEOUT', 30)
        self.temperature = getattr(self.config, 'TEMPERATURE', 0.1)
        self.max_tokens = getattr(self.config, 'MAX_TOKENS', 2048)

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
    ) -> Union[Optional[str], Any]:  # 流式返回生成器
        """生成回答"""
        try:
            messages = self._build_messages(prompt, context, history)
            if stream:
                return self._call_api_stream(messages)
            else:
                response = self._call_api(messages)
                if response and "choices" in response and len(response["choices"]) > 0:
                    content = response["choices"][0]["message"]["content"]
                    logger.info(f"生成回答成功，长度: {len(content)} 字符")
                    return content
                else:
                    logger.error(f"API 响应格式异常: {response}")
                    return None
        except Exception as e:
            logger.error(f"生成回答失败: {str(e)}")
            return None

    def _build_messages(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = [{"role": "system", "content": STRICT_SMART_SYSTEM_PROMPT}]

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

    def _call_api(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """调用 API（非流式）"""
        try:
            headers = self._get_headers()
            payload = self._build_payload(messages, stream=False)
            response = requests.post(
                self._get_api_url(),
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API 调用失败，状态码: {response.status_code}, 响应: {response.text}")
                return None
        except Exception as e:
            logger.error(f"API 请求异常: {str(e)}")
            return None

    def _call_api_stream(self, messages: List[Dict[str, str]]):
        """流式调用（返回生成器）"""
        try:
            headers = self._get_headers()
            payload = self._build_payload(messages, stream=True)
            response = requests.post(
                self._get_api_url(),
                headers=headers,
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            if response.status_code != 200:
                logger.error(f"流式API调用失败: {response.status_code} - {response.text}")
                return

            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: "):
                        decoded = decoded[6:]
                    if decoded == "[DONE]":
                        break
                    try:
                        chunk = json.loads(decoded)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"流式请求异常: {str(e)}")
            return

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

# ========================# 🎯 Qwen (通义千问) HTTP客户端# ========================class QwenClient(BaseLLMClient):    def __init__(self, config=None):        super().__init__(config)        # 设置API密钥        if hasattr(self.config, 'DASHSCOPE_API_KEY') and self.config.DASHSCOPE_API_KEY:            self.api_key = self.config.DASHSCOPE_API_KEY        elif hasattr(self.config, 'QWEN_API_KEY') and self.config.QWEN_API_KEY:            self.api_key = self.config.QWEN_API_KEY        else:            # 尝试从环境变量中获取API密钥            import os            api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('QWEN_API_KEY')            if api_key:                self.api_key = api_key            else:                logger.warning("未设置DASHSCOPE_API_KEY或QWEN_API_KEY")    
class QwenDashScopeClient(BaseLLMClient):
    def __init__(self, config=None):
        super().__init__(config)
        # 设置API密钥
        if hasattr(self.config, 'DASHSCOPE_API_KEY') and self.config.DASHSCOPE_API_KEY:
            self.api_key = self.config.DASHSCOPE_API_KEY
        elif hasattr(self.config, 'QWEN_API_KEY') and self.config.QWEN_API_KEY:
            self.api_key = self.config.QWEN_API_KEY
        else:
            # 尝试从环境变量中获取API密钥
            import os
            api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('QWEN_API_KEY')
            if api_key:
                self.api_key = api_key
            else:
                logger.warning("未设置DASHSCOPE_API_KEY或QWEN_API_KEY")
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-SSE": "enable"  # 如果需要流式响应，可加此头（DashScope 支持）
        }

    def _get_api_url(self) -> str:
        # 阿里云 DashScope API 地址（Qwen 系列）
        # 忽略配置中的空URL，直接返回默认URL
        return 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'

    def _build_payload(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        return {
            "model": self.model_name,  # 例如：qwen-plus
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "result_format": "message"  # DashScope 推荐格式
            },
            "stream": stream
        }
        
    def _call_api(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """调用 API（非流式）"""
        try:
            url = self._get_api_url()
            headers = self._get_headers()
            payload = self._build_payload(messages, stream=False)
            
            logger.debug(f"[QWEN_API] 准备发送请求到 {url}")
            logger.debug(f"[QWEN_API] 请求头: {headers}")
            logger.debug(f"[QWEN_API] 请求体: {payload}")
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            
            logger.debug(f"[QWEN_API] 响应状态码: {response.status_code}")
            logger.debug(f"[QWEN_API] 响应头: {dict(response.headers)}")
            
            if response.status_code == 200:
                logger.debug(f"[QWEN_API] API调用成功，状态码: {response.status_code}")
                
                # 初始化变量
                complete_content = ""
                all_lines = []
                
                # 即使是非流式调用，Qwen DashScope API 也返回事件流格式
                logger.debug("[QWEN_API] 开始处理响应流...")
                
                # 直接获取整个响应文本，然后按行处理
                raw_response = response.text
                logger.debug(f"[QWEN_API] 原始响应文本长度: {len(raw_response)} 字符")
                
                # 按行分割
                lines = raw_response.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                        
                    all_lines.append(line)
                    logger.debug(f"[QWEN_API] 处理行 {i+1}: {line}")
                    
                    # 处理数据行 - 注意这里用startswith('data:')而不是'data: '
                    if line.startswith('data:'):
                        try:
                            # 提取数据部分，使用正确的切片和strip()处理
                            data_part = line[5:].strip()
                            logger.debug(f"[QWEN_API] 数据部分: {data_part}")
                            
                            chunk = json.loads(data_part)
                            logger.debug(f"[QWEN_API] 解析到数据块: {chunk}")
                            
                            # 提取内容
                            if ('output' in chunk and 
                                'choices' in chunk['output'] and 
                                isinstance(chunk['output']['choices'], list) and 
                                len(chunk['output']['choices']) > 0):
                                choice = chunk['output']['choices'][0]
                                if ('message' in choice and 
                                    'content' in choice['message']):
                                    content = choice['message']['content']
                                    logger.debug(f"[QWEN_API] 提取到内容: {content}")
                                    # 保存内容，因为最后一行通常包含完整内容
                                    complete_content = content
                        except json.JSONDecodeError as e:
                            logger.warning(f"[QWEN_API] JSON解析错误: {str(e)}, 数据部分: {data_part}")
                            continue
                
                logger.debug(f"[QWEN_API] 响应流处理完成，总行数: {len(all_lines)}")
                logger.debug(f"[QWEN_API] 最终complete_content: '{complete_content}'")
                
                # 确保返回的格式与BaseLLMClient.generate_response方法期望的格式匹配
                if complete_content:
                    # 构造一个标准格式的响应，完全符合BaseLLMClient的期望
                    standard_response = {
                        "choices": [{
                            "message": {
                                "content": complete_content
                            }
                        }]
                    }
                    logger.debug(f"[QWEN_API] 构造的标准响应: {standard_response}")
                    return standard_response
                
                logger.error("[QWEN_API] 未能提取到完整响应内容")
                logger.debug(f"[QWEN_API] 所有接收到的行: {all_lines}")
                return None
            else:
                logger.error(f"[QWEN_API] API 调用失败，状态码: {response.status_code}, 响应: {response.text}")
                return None
        except Exception as e:
            logger.error(f"[QWEN_API] API 请求异常: {str(e)}")
            import traceback
            logger.error(f"[QWEN_API] 详细错误堆栈: {traceback.format_exc()}")
            return None
            
    def _call_api_stream(self, messages):
        """调用DashScope API的流式接口"""
        url = self._get_api_url()
        headers = self._get_headers()
        payload = self._build_payload(messages, stream=True)
        
        logger.debug(f"调用DashScope流式API: {url}")
        logger.debug(f"流式请求头: {headers}")
        logger.debug(f"流式请求体: {payload}")
        
        try:
            # 简化的流式请求实现
            response = requests.post(url, headers=headers, json=payload, stream=True)
            
            logger.debug(f"流式请求状态码: {response.status_code}")
            logger.debug(f"流式响应头: {response.headers}")
            
            line_count = 0
            chunk_count = 0
            
            for line in response.iter_lines():
                if line:
                    line_count += 1
                    decoded_line = line.decode('utf-8')
                    logger.debug(f"流式第{line_count}行: {decoded_line}")
                    
                    # 只处理data行
                    if decoded_line.startswith('data:'):
                        try:
                            # 提取data部分
                            data_part = decoded_line[5:].strip()
                            logger.debug(f"数据部分: {data_part}")
                            
                            # 解析JSON
                            chunk = json.loads(data_part)
                            logger.debug(f"解析后的chunk: {chunk}")
                            
                            # 直接检查并提取content
                            if ('output' in chunk and 
                                'choices' in chunk['output'] and 
                                len(chunk['output']['choices']) > 0 and 
                                'message' in chunk['output']['choices'][0] and 
                                'content' in chunk['output']['choices'][0]['message']):
                                
                                content = chunk['output']['choices'][0]['message']['content']
                                chunk_count += 1
                                logger.debug(f"成功提取第{chunk_count}个内容块: {content}")
                                
                                # 直接yield内容
                                yield content
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON解析错误: {str(e)}, 行内容: {decoded_line}")
                        except Exception as e:
                            logger.error(f"处理数据块时出错: {str(e)}")
            
            # 确保响应被关闭
            response.close()
            
            logger.debug(f"流式调用完成，处理了{line_count}行，提取了{chunk_count}个内容块")
            
        except Exception as e:
            logger.error(f"流式API调用异常: {type(e).__name__}: {str(e)}")
            raise

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
            "qwen_dashscope": QwenDashScopeClient,
            # 可继续扩展
        }

        if provider in client_map:
            return client_map[provider](config)
        else:
            logger.warning(f"未知模型提供商: {provider}，默认使用 DeepSeekClient")
            return DeepSeekClient(config)


# ========================
# 🏁 全局客户端实例（懒加载）
# ========================
class LazyLLMClient:
    def __init__(self):
        self._client = None
        self._config = None

    def __getattr__(self, name):
        if self._client is None:
            self._client = LLMClientFactory.create_client()
            self._config = getattr(self._client, 'config', None)
            logger.info(f"LLM 客户端已初始化，提供商: {getattr(self._config, 'MODEL_PROVIDER', '未知')}")
        return getattr(self._client, name)

    def validate_api_key(self) -> bool:
        """验证 API 密钥是否有效
        
        Returns:
            API 密钥是否有效的布尔值
        """
        if self._client is None:
            try:
                self._client = LLMClientFactory.create_client()
                self._config = getattr(self._client, 'config', None)
            except Exception:
                return False
        
        if hasattr(self._client, 'validate_api_key'):
            return self._client.validate_api_key()
        
        # 默认实现：检查是否有API密钥
        return hasattr(self._client, 'api_key') and bool(getattr(self._client, 'api_key', None))

    def refresh_client(self, new_config=None):
        """刷新客户端实例，可以使用新配置"""
        logger.info("刷新 LLM 客户端实例")
        self._client = LLMClientFactory.create_client(new_config)
        self._config = getattr(self._client, 'config', None)
        logger.info(f"LLM 客户端已刷新，提供商: {getattr(self._config, 'MODEL_PROVIDER', '未知')}")
        return self

    def get_status(self):
        """获取客户端状态信息"""
        if self._client is None:
            return {
                "initialized": False,
                "status": "未初始化"
            }
        
        try:
            is_api_valid = self.validate_api_key()
            return {
                "initialized": True,
                "provider": getattr(self._config, 'MODEL_PROVIDER', '未知'),
                "model": getattr(self._config, 'MODEL_NAME', '未知'),
                "api_key_valid": is_api_valid,
                "temperature": getattr(self._config, 'TEMPERATURE', 0.1),
                "max_tokens": getattr(self._config, 'MAX_TOKENS', 2048)
            }
        except Exception as e:
            logger.error(f"获取客户端状态失败: {str(e)}")
            return {
                "initialized": True,
                "error": str(e)
            }

# 创建全局懒加载客户端实例
llm_client = LazyLLMClient()