import logging
import os
import re
import time
import traceback
import json
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from datetime import datetime
import numpy as np
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.tools import Tool, tool
from langchain.agents import AgentType, initialize_agent
from src.config import global_config
from src.utils import format_time, generate_unique_id, safe_json_loads
from src.vector_store import vector_store_manager
from src.document_loader import document_loader
from src.llm_client import llm_client, BaseLLMClient, LLMClientFactory, refresh_client
from src.query_intent_classifier import query_intent_classifier

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义可用工具描述
TOOLS_DESCRIPTIONS = {
    "search_knowledge": "搜索知识库获取相关信息，用于回答问题或获取背景知识",
    "load_document": "加载新文档到知识库，支持PDF、Word、Excel、PPT、图片等多种格式",
    "clear_knowledge": "清空知识库中的所有内容",
    "get_knowledge_stats": "获取知识库的统计信息，如文档数量、索引状态等",
    "create_index": "在知识库中创建新的索引",
    "switch_index": "切换知识库的活动索引",
    "optimize_index": "优化知识库索引以提高搜索性能",
    "export_index": "导出知识库索引",
    "import_index": "导入知识库索引"
}

class Agent:
    """增强版智能代理类，支持流式响应、工具使用和高级错误处理"""
    
    def __init__(self, config=None):
        """初始化代理"""
        self.config = config or global_config
        self.llm = None
        self.memory = None
        self.chain = None
        self.agent_executor = None
        self.conversation_id = generate_unique_id()
        self.tools = []
        self.is_streaming = False
        self.last_error = None
        self.conversation_stats = {
            "total_responses": 0,
            "total_tokens": 0,
            "total_search_queries": 0,
            "total_tools_used": 0,
            "last_activity_time": None
        }
        
        # 初始化LLM
        self._init_llm()
        
        # 初始化记忆
        self._init_memory()
        
        # 初始化对话链
        self._init_chain()
        
        # 初始化工具
        self._init_tools()
        
        # 初始化代理执行器
        self._init_agent_executor()
    
    def _init_llm(self):
        """初始化语言模型（支持流式响应）"""
        try:
            # 尝试使用OpenAI API
            try:
                self.llm = ChatOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name=self.config.LLM_MODEL,
                    temperature=self.config.LLM_TEMPERATURE,
                    max_tokens=self.config.LLM_MAX_TOKENS,
                    streaming=True,  # 启用流式响应
                    verbose=True
                )
                logger.info(f"成功初始化OpenAI模型: {self.config.LLM_MODEL} (支持流式)")
            except Exception as e:
                logger.warning(f"OpenAI API初始化失败: {str(e)}")
                logger.info("尝试使用本地模型作为后备方案")
                
                # 尝试使用本地模型
                try:
                    from langchain_ollama import ChatOllama
                    self.llm = ChatOllama(
                        model=self.config.LOCAL_LLM_MODEL,
                        temperature=self.config.LLM_TEMPERATURE,
                        max_tokens=self.config.LLM_MAX_TOKENS,
                        streaming=True,  # 启用流式响应
                        verbose=True
                    )
                    logger.info(f"成功初始化本地模型: {self.config.LOCAL_LLM_MODEL} (支持流式)")
                except Exception as local_e:
                    logger.error(f"本地模型初始化失败: {str(local_e)}")
                    logger.info("使用简单的基于规则的响应生成器作为最后的后备方案")
                    
                    # 创建一个简单的基于规则的响应生成器
                    from langchain_core.language_models import BaseLanguageModel
                    from langchain_core.messages import HumanMessage, AIMessage
                    
                    class SimpleRuleBasedLLM(BaseLanguageModel):
                        def _generate(self, prompts, stop=None, **kwargs):
                            # 简单的规则响应生成
                            responses = []
                            for prompt in prompts:
                                response = "感谢您的提问。我注意到您想了解一些信息，但我目前无法访问高级语言模型。" \
                                           "您可以尝试提供更明确的问题，或者稍后再试。\n\n" \
                                           "如果您需要技术支持，请提供详细的问题描述，我会尽力帮助您。"
                                responses.append(response)
                            return {"generations": [[{"text": r}] for r in responses]}
                        
                        def _llm_type(self):
                            return "simple_rule_based"
                        
                        # 实现流式生成
                        def _stream(self, prompts, stop=None, **kwargs):
                            for prompt in prompts:
                                response = "感谢您的提问。我注意到您想了解一些信息，但我目前无法访问高级语言模型。" \
                                           "您可以尝试提供更明确的问题，或者稍后再试。\n\n" \
                                           "如果您需要技术支持，请提供详细的问题描述，我会尽力帮助您。"
                                for char in response:
                                    yield {"generations": [[{"text": char}]]}
                    
                    self.llm = SimpleRuleBasedLLM()
                    logger.info("成功初始化简单规则响应生成器")
        except Exception as e:
            logger.error(f"初始化LLM失败: {str(e)}")
            self.llm = None
    
    def _init_memory(self):
        """初始化增强版对话记忆"""
        try:
            # 创建记忆存储目录
            memory_dir = os.path.join(os.path.dirname(__file__), '..', '.memory')
            os.makedirs(memory_dir, exist_ok=True)
            
            # 使用文件存储历史记录，支持持久化
            history_path = os.path.join(memory_dir, f"conversation_{self.conversation_id}.json")
            
            # 使用窗口记忆，只保留最近的对话内容
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=self.config.MEMORY_SIZE,  # 设置记忆大小
                chat_memory=FileChatMessageHistory(history_path),
                output_key="output"
            )
            logger.info(f"成功初始化对话记忆，ID: {self.conversation_id}")
        except Exception as e:
            logger.error(f"初始化对话记忆失败: {str(e)}")
            # 回退到基本内存
            try:
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    k=self.config.MEMORY_SIZE
                )
                logger.warning("已回退到基本内存模式")
            except:
                self.memory = None
    
    def _init_chain(self):
        """初始化增强版对话链"""
        try:
            # 定义提示模板
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", self.config.SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            
            # 创建对话链
            self.chain = LLMChain(
                llm=self.llm,
                prompt=prompt_template,
                memory=self.memory,
                output_parser=StrOutputParser(),
                verbose=True
            )
            logger.info("成功初始化对话链")
        except Exception as e:
            logger.error(f"初始化对话链失败: {str(e)}")
            self.chain = None
    
    def _init_tools(self):
        """初始化可用工具"""
        try:
            # 搜索知识库工具
            search_knowledge_tool = Tool(
                name="search_knowledge",
                func=self._search_knowledge,
                description=TOOLS_DESCRIPTIONS["search_knowledge"]
            )
            
            # 加载文档工具
            load_document_tool = Tool(
                name="load_document",
                func=self._load_document,
                description=TOOLS_DESCRIPTIONS["load_document"]
            )
            
            # 清空知识库工具
            clear_knowledge_tool = Tool(
                name="clear_knowledge",
                func=self._clear_knowledge,
                description=TOOLS_DESCRIPTIONS["clear_knowledge"]
            )
            
            # 获取知识库统计工具
            get_knowledge_stats_tool = Tool(
                name="get_knowledge_stats",
                func=self._get_knowledge_stats,
                description=TOOLS_DESCRIPTIONS["get_knowledge_stats"]
            )
            
            # 添加所有工具到列表
            self.tools = [
                search_knowledge_tool,
                load_document_tool,
                clear_knowledge_tool,
                get_knowledge_stats_tool
            ]
            
            # 可选工具：仅在需要时添加
            # 创建索引工具
            create_index_tool = Tool(
                name="create_index",
                func=self._create_index,
                description=TOOLS_DESCRIPTIONS["create_index"]
            )
            
            # 切换索引工具
            switch_index_tool = Tool(
                name="switch_index",
                func=self._switch_index,
                description=TOOLS_DESCRIPTIONS["switch_index"]
            )
            
            # 优化索引工具
            optimize_index_tool = Tool(
                name="optimize_index",
                func=self._optimize_index,
                description=TOOLS_DESCRIPTIONS["optimize_index"]
            )
            
            # 导出索引工具
            export_index_tool = Tool(
                name="export_index",
                func=self._export_index,
                description=TOOLS_DESCRIPTIONS["export_index"]
            )
            
            # 导入索引工具
            import_index_tool = Tool(
                name="import_index",
                func=self._import_index,
                description=TOOLS_DESCRIPTIONS["import_index"]
            )
            
            # 添加高级工具
            self.tools.extend([
                create_index_tool,
                switch_index_tool,
                optimize_index_tool,
                export_index_tool,
                import_index_tool
            ])
            
            logger.info(f"成功初始化 {len(self.tools)} 个工具")
        except Exception as e:
            logger.error(f"初始化工具失败: {str(e)}")
            self.tools = []
    
    def _init_agent_executor(self):
        """初始化代理执行器"""
        try:
            if self.llm and self.tools and self.memory:
                self.agent_executor = initialize_agent(
                    tools=self.tools,
                    llm=self.llm,
                    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                    memory=self.memory,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=3,
                    early_stopping_method="generate"
                )
                logger.info("成功初始化代理执行器")
        except Exception as e:
            logger.error(f"初始化代理执行器失败: {str(e)}")
            self.agent_executor = None
    
    def generate_response(self, user_input: str, use_knowledge: bool = True, allow_tools: bool = True) -> str:
        """生成响应（非流式）"""
        try:
            # 记录用户输入
            logger.info(f"收到用户输入: {user_input[:100]}...")

            # 路由决策（基于 YAML 配置）
            routing = self._route_query(user_input)
            action = routing.get('action')
            params = routing.get('params', {})
            preset_answer = routing.get('preset_answer')
            intent = routing.get('intent', {})
            logger.info(f"路由决策: action={action}, intent={intent.get('intent_type')}, params={params}")
            
            # 优先处理文档元数据类问题：作者/建立日期/创建时间等，直接从文档属性中给出答案
            if self._is_metadata_query(user_input):
                meta_answer = self._answer_metadata_query(user_input)
                if meta_answer:
                    return meta_answer
            
            # 检查是否需要使用工具
            if allow_tools and (action == 'tool' or self._should_use_tools(user_input)):
                return self._use_tools(user_input)
            
            # 如果需要使用知识库，先进行检索
            knowledge_content = ""
            if use_knowledge and action == 'rag':
                # 使用路由参数进行检索
                hybrid = params.get('retrieval_strategy', 'hybrid') == 'hybrid'
                top_k = params.get('top_k', 3)
                knowledge_content = self._retrieve_with_params(user_input, top_k=top_k, hybrid_search=hybrid)
                if knowledge_content:
                    logger.info(f"从知识库检索到相关内容")
                    # 使用更严格的指令化提示，避免通用拒答
                    user_input = (
                        f"用户问题: {user_input}\n\n"
                        f"请严格基于下方资料生成结构化概览，突出关键要点（背景/目的、适用范围、角色与流程、关键要点、注意事项）。"
                        f"不得输出诸如‘无法回答’、‘信息不足无法作答’等通用拒绝语；若资料确有不足，请直接说明不足，并给出可执行的后续建议。\n\n"
                        f"相关资料:\n{knowledge_content}"
                    )
            elif action in ('preset_or_llm', 'refuse_or_short_reply') and preset_answer:
                # 直接返回预设短答
                return preset_answer
            
            # 构造历史对话（转换为 role/content 字典列表）
            history = None
            if self.memory:
                chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
                from langchain_core.messages import HumanMessage, AIMessage
                history = []
                for msg in chat_history:
                    role = "assistant" if isinstance(msg, AIMessage) else "user"
                    history.append({"role": role, "content": msg.content})

            # 通过 llm_client 的高层接口生成响应（带重试/缓存）
            # 检索型回答避免写入缓存，防止命中通用拒答缓存
            cache_enabled = not (use_knowledge and action == 'rag')
            response = llm_client.generate_response(
                prompt=user_input,
                context=None,
                history=history,
                stream=False,
                cache_enabled=cache_enabled,
                structured_schema=None
            )

            # 标准化响应内容为字符串
            response_content = response if isinstance(response, str) else str(response)

            # 若命中通用拒答或过短，且存在知识上下文，则进行一次强化重试
            def _is_low_quality(text: str) -> bool:
                t = (text or "").strip()
                if len(t) < 30:
                    return True
                bad_phrases = [
                    "无法回答", "不能回答", "无法作答", "信息不足", "根据现有资料，无法回答该问题",
                    "无法提供", "无法确定", "抱歉，无法"
                ]
                return any(bp in t for bp in bad_phrases)

            if knowledge_content and _is_low_quality(response_content):
                logger.info("检测到低质量回答，进行一次强化重试以生成结构化概览。")
                strict_prompt = (
                    f"用户问题: {user_input}\n\n"
                    f"任务: 仅基于下方资料，生成中文结构化‘报告概览要点’列表。\n"
                    f"要求: 列表化、简洁准确；包含 背景/目的、适用范围、关键特性、技术范围 或 文档目的；"
                    f"若资料不足，请标注‘资料不足’并列出已确认的要点与后续建议；"
                    f"禁止输出任何‘无法回答/不能回答’等拒绝语。\n\n"
                    f"资料:\n{knowledge_content}"
                )
                retry_resp = llm_client.generate_response(
                    prompt=strict_prompt,
                    context=None,
                    history=history,
                    stream=False,
                    cache_enabled=False,
                    structured_schema=None
                )
                response_content = retry_resp if isinstance(retry_resp, str) else str(retry_resp)
            
            # 更新记忆
            if self.memory:
                from langchain_core.messages import HumanMessage, AIMessage
                self.memory.chat_memory.add_messages([
                    HumanMessage(content=user_input),
                    AIMessage(content=response_content)
                ])
            
            # 更新统计信息
            self._update_conversation_stats(has_response=True)
            
            # 记录响应
            logger.info(f"生成响应: {response_content[:100]}...")
            
            return response_content
        except Exception as e:
            logger.error(f"生成响应失败: {str(e)}")
            traceback.print_exc()
            self.last_error = str(e)
            return self._generate_error_response(e)
    
    async def generate_streaming_response(self, user_input: str, use_knowledge: bool = True) -> AsyncGenerator[str, None]:
        """生成流式响应"""
        if not self.llm:
            yield "抱歉，我当前无法处理您的请求。请稍后再试。"
            return
            
        try:
            # 记录用户输入
            logger.info(f"收到用户输入(流式): {user_input[:100]}...")
            self.is_streaming = True

            # 路由决策（基于 YAML 配置）
            routing = self._route_query(user_input)
            action = routing.get('action')
            params = routing.get('params', {})
            preset_answer = routing.get('preset_answer')
            intent = routing.get('intent', {})
            logger.info(f"流式路由决策: action={action}, intent={intent.get('intent_type')}, params={params}")
            
            # 如果需要使用知识库，先进行检索
            knowledge_content = ""
            if use_knowledge and action == 'rag':
                hybrid = params.get('retrieval_strategy', 'hybrid') == 'hybrid'
                top_k = params.get('top_k', 3)
                knowledge_content = self._retrieve_with_params(user_input, top_k=top_k, hybrid_search=hybrid)
                if knowledge_content:
                    logger.info(f"从知识库检索到相关内容(流式)")
                    # 将知识内容添加到输入中
                    user_input = f"用户问题: {user_input}\n\n相关知识:\n{knowledge_content}"
            elif action in ('preset_or_llm', 'refuse_or_short_reply') and preset_answer:
                yield preset_answer
                return
            
            # 构造历史对话（转换为 role/content 字典列表）
            history = None
            if self.memory:
                chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
                from langchain_core.messages import HumanMessage, AIMessage
                history = []
                for msg in chat_history:
                    role = "assistant" if isinstance(msg, AIMessage) else "user"
                    history.append({"role": role, "content": msg.content})

            # 使用 llm_client 的流式生成接口（同步生成器），在异步函数中迭代并 yield
            full_response = ""
            for chunk in llm_client.generate_streaming_response(
                prompt=user_input,
                context=None,
                history=history,
                cache_enabled=False
            ):
                if chunk:
                    full_response += chunk
                    yield chunk
            
            # 更新记忆
            if self.memory:
                from langchain_core.messages import HumanMessage, AIMessage
                self.memory.chat_memory.add_messages([
                    HumanMessage(content=user_input),
                    AIMessage(content=full_response)
                ])
            
            # 更新统计信息
            self._update_conversation_stats(has_response=True)
            
            logger.info(f"流式响应完成: {full_response[:100]}...")
        except Exception as e:
            logger.error(f"生成流式响应失败: {str(e)}")
            self.last_error = str(e)
            # 发送错误消息
            error_message = self._generate_error_response(e)
            yield error_message
        finally:
            self.is_streaming = False
    
    def _retrieve_from_knowledge(self, query: str) -> str:
        """从知识库检索相关内容（增强版）"""
        try:
            # 智能决定是否使用混合检索
            use_hybrid = self._should_use_hybrid_search(query)
            
            # 从向量存储中检索相关文档
            results = vector_store_manager.similarity_search(
                query, 
                k=3,
                hybrid_search=use_hybrid
            )
            
            # 更新统计信息
            self.conversation_stats["total_search_queries"] += 1
            
            # 格式化检索结果
            formatted_results = []
            for i, doc in enumerate(results):
                # 获取文档内容
                content = getattr(doc, 'page_content', '')
                # 获取文档元数据
                metadata = getattr(doc, 'metadata', {})
                
                # 格式化文档信息
                source_info = f"来源: {metadata.get('source', '未知')}"
                if 'page' in metadata:
                    source_info += f"，第 {metadata['page']} 页"
                # 优先展示文档相关元数据，便于回答“作者/建立日期”等问题
                if 'author' in metadata:
                    source_info += f"，作者: {metadata['author']}"
                # 使用 created/modified/date 作为时间线信息
                if 'created' in metadata:
                    source_info += f"，创建时间: {metadata['created']}"
                elif 'date' in metadata:
                    source_info += f"，日期: {metadata['date']}"
                if 'modified' in metadata:
                    source_info += f"，修改时间: {metadata['modified']}"
                
                # 对内容进行截断，避免过长
                max_content_length = 300
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                
                formatted_result = f"【相关资料 {i+1}】\n{source_info}\n{content}\n"
                formatted_results.append(formatted_result)
            
            return "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"从知识库检索失败: {str(e)}")
            return ""

    def _retrieve_with_params(self, query: str, top_k: int = 3, hybrid_search: bool = True) -> str:
        """根据路由参数从知识库检索内容（新增）"""
        try:
            results = vector_store_manager.similarity_search(
                query,
                k=top_k,
                hybrid_search=hybrid_search
            )
            self.conversation_stats["total_search_queries"] += 1
            formatted_results = []
            for i, doc in enumerate(results):
                content = getattr(doc, 'page_content', '')
                metadata = getattr(doc, 'metadata', {})
                source_info = f"来源: {metadata.get('source', '未知')}"
                if 'page' in metadata:
                    source_info += f"，第 {metadata['page']} 页"
                # 展示作者与时间元数据，便于回答“作者/建立日期”等问题
                if 'author' in metadata:
                    source_info += f"，作者: {metadata['author']}"
                if 'created' in metadata:
                    source_info += f"，创建时间: {metadata['created']}"
                elif 'date' in metadata:
                    source_info += f"，日期: {metadata['date']}"
                if 'modified' in metadata:
                    source_info += f"，修改时间: {metadata['modified']}"
                max_content_length = 300
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                formatted_results.append(f"【相关资料 {i+1}】\n{source_info}\n{content}\n")
            return "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"路由参数检索失败: {str(e)}")
            return ""

    # ===== 元数据直答增强 =====
    def _is_metadata_query(self, text: str) -> bool:
        """判断是否为文档元数据类问题（作者/建立日期/创建时间等）"""
        if not text:
            return False
        t = text.lower()
        keywords_cn = [
            "作者", "文件作者", "文档作者",
            "建立日期", "创建日期", "创建时间", "生成日期",
            "修改时间", "最后修改", "最后修改者",
        ]
        keywords_en = ["author", "created", "date", "modified", "last modified"]
        return any(k in text for k in keywords_cn) or any(k in t for k in keywords_en)

    def _answer_metadata_query(self, user_input: str) -> str:
        """直接基于检索到的文档元数据给出简洁直答"""
        try:
            results = vector_store_manager.similarity_search(user_input, k=5, hybrid_search=True) or []
            if not results:
                return "未找到相关文档，无法提取作者或日期。"

            best_md = None
            for doc in results:
                md = getattr(doc, 'metadata', {}) or {}
                if md.get('author') or md.get('created') or md.get('date') or md.get('modified') or md.get('last_modified_by'):
                    best_md = md
                    break
            if best_md is None:
                best_md = getattr(results[0], 'metadata', {}) or {}

            source = best_md.get('source') or best_md.get('source_file_name') or '未知来源'
            author = best_md.get('author') or '未设置'
            created = best_md.get('created') or best_md.get('date') or '未设置'
            modified = best_md.get('modified') or '未设置'
            last_modified_by = best_md.get('last_modified_by') or '未设置'

            if all(v == '未设置' for v in [author, created]) and modified == '未设置':
                return f"来源: {source}。未在该文档属性中找到作者或建立日期。请在Word文件属性中补充作者与创建时间后重新入库。"

            return (
                f"来源: {source}\n"
                f"作者: {author}\n"
                f"建立日期: {created}\n"
                f"修改时间: {modified}\n"
                f"最后修改者: {last_modified_by}"
            )
        except Exception as e:
            logger.error(f"元数据直答失败: {str(e)}")
            return "处理元数据查询时出现错误，请稍后重试。"

    def _route_query(self, user_input: str, document_type: Optional[str] = None) -> Dict[str, Any]:
        """调用查询意图分类器，生成路由决策（新增）"""
        try:
            return query_intent_classifier.get_routing_decision(user_input, document_type=document_type)
        except Exception as e:
            logger.warning(f"路由决策失败，回退到默认: {str(e)}")
            return {
                'intent': {'intent_type': 'unknown', 'confidence': 0.0},
                'action': 'rag',
                'params': {'retrieval_strategy': 'hybrid', 'top_k': 3},
                'preset_answer': None
            }
    
    def clear_chat_history(self):
        """清空对话历史"""
        try:
            if self.memory:
                self.memory.clear()
                logger.info("对话历史已清空")
                return True
            return False
        except Exception as e:
            logger.error(f"清空对话历史失败: {str(e)}")
            return False
    
    def _should_use_tools(self, user_input: str) -> bool:
        """判断是否应该使用工具"""
        # 简单的规则：检查用户输入是否包含工具相关的关键词
        tool_keywords = {
            "search_knowledge": ["搜索", "查找", "查询", "了解"],
            # 收窄加载文档的触发词，避免普通问句含“文档”误触发
            "load_document": ["加载文档", "上传文档", "导入文档"],
            "clear_knowledge": ["清空", "删除", "重置"],
            "get_knowledge_stats": ["统计", "状态", "信息"],
            "create_index": ["创建索引"],
            "switch_index": ["切换索引"],
            "optimize_index": ["优化索引"],
            "export_index": ["导出索引"],
            "import_index": ["导入索引"]
        }
        
        for keywords in tool_keywords.values():
            if any(keyword in user_input.lower() for keyword in keywords):
                return True
        
        return False
    
    def _should_use_hybrid_search(self, query: str) -> bool:
        """判断是否应该使用混合检索"""
        # 如果查询包含专业术语或关键词较多，使用混合检索
        # 简单判断：查询中包含的词语数量或特定符号
        query_lower = query.lower()
        
        # 检查是否包含技术术语
        technical_terms = ["算法", "模型", "API", "框架", "函数", "类", "方法"]
        if any(term in query_lower for term in technical_terms):
            return True
        
        # 检查是否包含代码相关符号
        code_symbols = ["=", "()", "{", "}", "[", "]", ";", ":"]
        if any(symbol in query for symbol in code_symbols):
            return True
        
        # 如果查询较短但关键词明确
        if len(query.split()) <= 5:
            return True
        
        return False
    
    def _use_tools(self, user_input: str) -> str:
        """使用工具处理请求"""
        if not self.agent_executor:
            # 如果没有代理执行器，尝试直接调用相应工具
            return self._direct_tool_call(user_input)
        
        try:
            result = self.agent_executor.run(input=user_input)
            self.conversation_stats["total_tools_used"] += 1
            return result
        except Exception as e:
            logger.error(f"使用工具失败: {str(e)}")
            return f"使用工具时出现错误: {str(e)}\n尝试使用常规回答方式...\n{self.generate_response(user_input, use_knowledge=True, allow_tools=False)}"
    
    def _direct_tool_call(self, user_input: str) -> str:
        """直接调用工具"""
        user_lower = user_input.lower()
        
        if any(keyword in user_lower for keyword in ["搜索", "查找", "查询"]):
            # 提取搜索查询
            search_query = user_input.replace("搜索", "").replace("查找", "").replace("查询", "").strip()
            knowledge = self._retrieve_from_knowledge(search_query)
            if knowledge:
                return f"根据搜索结果，我找到了以下相关信息：\n\n{knowledge}"
            else:
                return "没有找到相关信息，请尝试其他关键词。"
        elif any(keyword in user_lower for keyword in ["清空", "删除"]):
            if self._clear_knowledge():
                return "知识库已成功清空。"
            else:
                return "清空知识库失败。"
        elif any(keyword in user_lower for keyword in ["统计", "状态"]):
            stats = self._get_knowledge_stats()
            return f"知识库统计信息：\n\n{stats}"
        
        # 默认返回常规回答
        return self.generate_response(user_input, use_knowledge=True, allow_tools=False)
    
    def _generate_error_response(self, error: Exception) -> str:
        """生成错误响应"""
        error_type = type(error).__name__
        
        # 根据不同类型的错误提供不同的回复
        error_responses = {
            "APIError": "很抱歉，我在连接服务器时遇到了问题。请检查您的网络连接或稍后再试。",
            "RateLimitError": "您的请求频率过高，请稍等片刻后再尝试。",
            "TimeoutError": "请求超时，请检查您的网络连接或重试。",
            "ValidationError": "您的请求格式有误，请尝试重新表述您的问题。",
            "FileNotFoundError": "找不到指定的文件，请检查文件路径是否正确。",
            "PermissionError": "权限不足，无法执行此操作。"
        }
        
        return error_responses.get(error_type, f"抱歉，处理您的请求时出现了错误。错误类型: {error_type}\n请尝试重新表述您的问题或稍后再试。")
    
    def _update_conversation_stats(self, has_response: bool = False):
        """更新对话统计信息"""
        self.conversation_stats["last_activity_time"] = datetime.now().isoformat()
        if has_response:
            self.conversation_stats["total_responses"] += 1
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """获取对话统计信息"""
        return self.conversation_stats.copy()

    def reset_conversation(self):
        """重置对话"""
        self.conversation_id = generate_unique_id()
        self.clear_chat_history()
        self.conversation_stats = {
            "total_responses": 0,
            "total_tokens": 0,
            "total_search_queries": 0,
            "total_tools_used": 0,
            "last_activity_time": None
        }
        self.last_error = None
        logger.info(f"对话已重置，新ID: {self.conversation_id}")
        return self.conversation_id

    def generate_response_with_details(self, query: str, use_history: bool = True, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """生成带引用与细节的回答，供 /api/chat_with_references 使用"""
        start_time = time.time()
        try:
            # 构造历史对话（转换为 role/content 字典列表）
            history = None
            if use_history and self.memory:
                chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
                from langchain_core.messages import HumanMessage, AIMessage
                history = []
                for msg in chat_history:
                    role = "assistant" if isinstance(msg, AIMessage) else "user"
                    history.append({"role": role, "content": msg.content})

            # 检索知识库，获取引用材料
            retrieved_documents = []
            references = []
            try:
                results = vector_store_manager.similarity_search(query, k=5, hybrid_search=True)
                for i, doc in enumerate(results or []):
                    content = getattr(doc, 'page_content', '')
                    metadata = getattr(doc, 'metadata', {}) or {}
                    source = metadata.get('source', '未知来源')
                    # 收集用于返回的简要引用和检索详情
                    references.append({
                        'index': i + 1,
                        'source': source,
                        'snippet': (content[:500] + '...') if len(content) > 500 else content
                    })
                    retrieved_documents.append({
                        'source': source,
                        'metadata': metadata,
                        'content_preview': (content[:200] + '...') if len(content) > 200 else content
                    })
            except Exception as re_err:
                logger.error(f"检索知识库失败: {str(re_err)}")

            # 组合上下文并向 LLM 生成回答
            knowledge_context = "\n\n".join([
                f"【资料{i+1}】来源: {ref['source']}\n{ref['snippet']}" for i, ref in enumerate(references)
            ])

            prompt = (
                f"用户问题: {query}\n\n"
                f"如果有相关资料，请在保证准确性的前提下，给出结构化概述。"
                f"优先围绕以下要点：背景/目的、适用范围、角色与流程、关键要点、注意事项。"
                f"若资料不足，请明确说明不足与后续建议。\n\n"
                f"相关资料:\n{knowledge_context}"
            )

            response_text = llm_client.generate_response(
                prompt=prompt,
                context=None,
                history=history,
                stream=False,
                cache_enabled=True,
                structured_schema=None
            )

            answer = response_text if isinstance(response_text, str) else str(response_text)

            processing_time = round(time.time() - start_time, 4)
            self._update_conversation_stats(has_response=True)

            return {
                'answer': answer,
                'references': references,
                'intent_type': 'general',
                'retrieved_documents': retrieved_documents,
                'processing_time': processing_time,
                'user_id': user_id,
                'session_id': session_id
            }
        except Exception as e:
            logger.error(f"生成带引用回答失败: {str(e)}")
            traceback.print_exc()
            processing_time = round(time.time() - start_time, 4)
            return {
                'answer': self._generate_error_response(e),
                'references': [],
                'intent_type': 'error',
                'retrieved_documents': [],
                'processing_time': processing_time,
                'user_id': user_id,
                'session_id': session_id
            }

    def update_system_prompt(self, new_prompt: str):
        """更新系统提示"""
        try:
            # 更新配置中的系统提示
            self.config.SYSTEM_PROMPT = new_prompt
            # 重新初始化对话链以应用新的系统提示
            self._init_chain()
            # 重新初始化代理执行器
            self._init_agent_executor()
            logger.info("系统提示已更新")
            return True
        except Exception as e:
            logger.error(f"更新系统提示失败: {str(e)}")
            return False
    
    def update_llm_client(self, config=None):
        """更新LLM客户端配置"""
        try:
            # 使用全局的refresh_client函数更新agent模块的客户端
            success = refresh_client('agent', config)
            if success:
                logger.info("LLM客户端已更新")
                # 重新初始化LLM以使用更新后的客户端
                self._init_llm()
                return True
            return False
        except Exception as e:
            logger.error(f"更新LLM客户端失败: {str(e)}")
            return False
    
    # 工具方法实现
    def _search_knowledge(self, query: str) -> str:
        """搜索知识库工具实现"""
        try:
            results = vector_store_manager.similarity_search(query, k=5, hybrid_search=True)
            
            if not results:
                return "未找到相关信息"
            
            formatted_results = []
            for i, doc in enumerate(results):
                content = getattr(doc, 'page_content', '')[:200] + "..."
                metadata = getattr(doc, 'metadata', {})
                source = metadata.get('source', '未知来源')
                formatted_results.append(f"【结果{i+1}】来自 {source}: {content}")
            
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"搜索失败: {str(e)}"
    
    def _load_document(self, file_path: str) -> str:
        """加载文档工具实现"""
        try:
            if not os.path.exists(file_path):
                return f"文件不存在: {file_path}"
            
            # 使用文档加载器管理加载文件
            documents = document_loader_manager.load_document(file_path)
            if not documents:
                return f"无法加载文档: {file_path}"
            
            # 添加到向量存储
            success = vector_store_manager.add_documents(documents)
            if success:
                return f"成功加载并索引文档: {file_path}\n文档数量: {len(documents)}"
            else:
                return f"加载文档成功但索引失败: {file_path}"
        except Exception as e:
            return f"加载文档失败: {str(e)}"
    
    def _clear_knowledge(self) -> str:
        """清空知识库工具实现"""
        try:
            success = vector_store_manager.clear_vector_store()
            if success:
                return "知识库已成功清空"
            else:
                return "清空知识库失败"
        except Exception as e:
            return f"清空知识库失败: {str(e)}"
    
    def _get_knowledge_stats(self) -> str:
        """获取知识库统计工具实现"""
        try:
            stats = vector_store_manager.get_stats()
            
            # 格式化统计信息
            formatted_stats = []
            formatted_stats.append(f"索引数量: {len(stats.get('indices', {}))}")
            
            # 索引详情
            indices = stats.get('indices', {})
            for index_name, index_info in indices.items():
                formatted_stats.append(f"  - {index_name}: {index_info.get('vector_count', 0)} 个向量")
            
            # 其他统计信息
            formatted_stats.append(f"总查询次数: {stats.get('queries', 0)}")
            formatted_stats.append(f"缓存命中次数: {stats.get('cache_hits', 0)}")
            formatted_stats.append(f"已添加文档数: {stats.get('documents_added', 0)}")
            formatted_stats.append(f"缓存大小: {stats.get('cache_size', 0)}")
            formatted_stats.append(f"元数据键数量: {stats.get('metadata_keys', 0)}")
            
            return "\n".join(formatted_stats)
        except Exception as e:
            return f"获取统计信息失败: {str(e)}"
    
    def _create_index(self, index_name: str) -> str:
        """创建索引工具实现"""
        try:
            success = vector_store_manager.create_index(index_name)
            if success:
                return f"成功创建索引: {index_name}"
            else:
                return f"创建索引失败: {index_name} (可能已存在)"
        except Exception as e:
            return f"创建索引失败: {str(e)}"
    
    def _switch_index(self, index_name: str) -> str:
        """切换索引工具实现"""
        try:
            success = vector_store_manager.switch_index(index_name)
            if success:
                return f"已切换到索引: {index_name}"
            else:
                return f"切换索引失败: {index_name}"
        except Exception as e:
            return f"切换索引失败: {str(e)}"
    
    def _optimize_index(self, index_name: str = None) -> str:
        """优化索引工具实现"""
        try:
            success = vector_store_manager.optimize_index(index_name)
            if success:
                target_index = index_name or "当前索引"
                return f"已成功优化索引: {target_index}"
            else:
                return "优化索引失败"
        except Exception as e:
            return f"优化索引失败: {str(e)}"
    
    def _export_index(self, index_name: str = None) -> str:
        """导出索引工具实现"""
        try:
            export_path = vector_store_manager.export_index(index_name)
            if export_path:
                return f"成功导出索引到: {export_path}"
            else:
                return "导出索引失败"
        except Exception as e:
            return f"导出索引失败: {str(e)}"
    
    def _import_index(self, index_name: str, import_path: str) -> str:
        """导入索引工具实现"""
        try:
            success = vector_store_manager.import_index(index_name, import_path)
            if success:
                return f"成功从 {import_path} 导入索引: {index_name}"
            else:
                return f"导入索引失败: {import_path}"
        except Exception as e:
            return f"导入索引失败: {str(e)}"

# 创建代理实例
agent = Agent()

# 创建异步代理实例
def create_async_agent(config=None):
    """创建异步代理实例"""
    return Agent(config)