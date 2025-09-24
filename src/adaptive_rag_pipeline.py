import logging
import os
import yaml
import numpy as np
import re
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

# 导入核心组件
from src.document_classifier import document_classifier
from src.query_intent_classifier import query_intent_classifier
from src.vector_store import vector_store_manager
from src.llm_client import llm_client as global_llm_client
from src.config import global_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveRAGPipeline:
    """自适应RAG流水线，根据文档类型和查询意图动态选择最优策略
    """
    
    def __init__(self, config_path: str = "config/rag_strategies.yaml"):
        """初始化自适应RAG流水线
        
        Args:
            config_path: RAG策略配置文件路径
        """
        # 存储RAG策略配置
        self.strategies = {}
        # 存储提示模板配置
        self.prompt_templates = {}
        # 保存对LLM客户端的引用
        self.llm_client = global_llm_client
        # 加载策略配置
        self._load_strategies(config_path)
        # 检索缓存，用于优化重复查询
        self.retrieval_cache = {}
        # 检索缓存大小限制
        self.max_cache_size = 1000
        # 执行器，用于并发检索
        self.executor = ThreadPoolExecutor(max_workers=4)
        # 多语言支持配置
        self.multilingual_support = self._load_multilingual_config()
        
        # 默认策略（当没有匹配的策略时使用）
        self.default_strategy = {
            "top_k": 4,
            "score_threshold": 0.5,
            "embedding_model": "default",
            "reranker_weight": 0.3,
            "prompt_template": "default",
            "retrieval_type": "hybrid",  # 混合检索策略
            "diversity_factor": 0.3,      # 多样性因子
            "chunk_overlap": 0.2,         # 块重叠比例
            "context_window": 10000       # 上下文窗口大小
        }
        
        # 默认提示模板
        self.default_prompt_template = "根据提供的上下文信息回答问题，确保回答准确、简洁。"
        
        logger.info("自适应RAG流水线初始化完成")
    
    def _load_strategies(self, config_path: str) -> None:
        """加载RAG策略配置文件
        
        Args:
            config_path: 配置文件路径
        """
        try:
            # 检查配置文件是否存在
            if not os.path.exists(config_path):
                logger.error(f"RAG策略配置文件不存在: {config_path}")
                # 使用默认策略配置
                self._load_default_strategies()
                return
            
            # 加载配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
                # 加载策略配置
                if 'strategies' in config:
                    self.strategies = config['strategies']
                    logger.info(f"成功加载 {len(self.strategies)} 种RAG策略配置")
                else:
                    logger.warning("配置文件中未找到strategies字段")
                    self._load_default_strategies()
                
                # 加载提示模板配置
                if 'prompt_templates' in config:
                    self.prompt_templates = config['prompt_templates']
                    logger.info(f"成功加载 {len(self.prompt_templates)} 种提示模板")
                else:
                    logger.warning("配置文件中未找到prompt_templates字段")
                    self._load_default_prompt_templates()
                    
                # 加载检索策略配置
                if 'retrieval_strategies' in config:
                    self.retrieval_strategies = config['retrieval_strategies']
                    logger.info(f"成功加载 {len(self.retrieval_strategies)} 种检索策略配置")
                else:
                    logger.warning("配置文件中未找到retrieval_strategies字段")
                    # 设置默认检索策略
                    self.retrieval_strategies = {
                        "hybrid": {
                            "semantic_weight": 0.7,
                            "keyword_weight": 0.3,
                            "rerank": True
                        },
                        "semantic_only": {
                            "semantic_weight": 1.0,
                            "keyword_weight": 0.0,
                            "rerank": True
                        },
                        "keyword_only": {
                            "semantic_weight": 0.0,
                            "keyword_weight": 1.0,
                            "rerank": False
                        }
                    }
                    
        except Exception as e:
            logger.error(f"加载RAG策略配置文件失败: {str(e)}")
            # 加载默认策略配置
            self._load_default_strategies()
            self._load_default_prompt_templates()
            # 设置默认检索策略
            self.retrieval_strategies = {
                "hybrid": {
                    "semantic_weight": 0.7,
                    "keyword_weight": 0.3,
                    "rerank": True
                },
                "semantic_only": {
                    "semantic_weight": 1.0,
                    "keyword_weight": 0.0,
                    "rerank": True
                },
                "keyword_only": {
                    "semantic_weight": 0.0,
                    "keyword_weight": 1.0,
                    "rerank": False
                }
            }
    
    def _load_default_strategies(self) -> None:
        """加载默认的RAG策略配置"""
        logger.info("使用默认RAG策略配置")
        self.strategies = {
            # 学术论文相关策略
            "academic_paper_specific_detail": {
                "top_k": 5,
                "score_threshold": 0.6,
                "embedding_model": "default",
                "reranker_weight": 0.4,
                "prompt_template": "citation",
                "retrieval_type": "hybrid",
                "diversity_factor": 0.2,
                "chunk_overlap": 0.15,
                "context_window": 12000
            },
            "academic_paper_overview_request": {
                "top_k": 3,
                "score_threshold": 0.5,
                "embedding_model": "default",
                "reranker_weight": 0.3,
                "prompt_template": "summary",
                "retrieval_type": "semantic_only",
                "diversity_factor": 0.4,
                "chunk_overlap": 0.2,
                "context_window": 10000
            },
            # 需求文档相关策略
            "requirement_doc_specific_detail": {
                "top_k": 5,
                "score_threshold": 0.6,
                "embedding_model": "default",
                "reranker_weight": 0.4,
                "prompt_template": "requirement_detail",
                "retrieval_type": "hybrid",
                "diversity_factor": 0.2,
                "chunk_overlap": 0.15,
                "context_window": 12000
            },
            "requirement_doc_overview_request": {
                "top_k": 2,
                "score_threshold": 0.5,
                "embedding_model": "default",
                "reranker_weight": 0.3,
                "prompt_template": "requirement_summary",
                "retrieval_type": "semantic_only",
                "diversity_factor": 0.4,
                "chunk_overlap": 0.2,
                "context_window": 8000
            },
            # 通用策略
            "general_specific_detail": {
                "top_k": 4,
                "score_threshold": 0.55,
                "embedding_model": "default",
                "reranker_weight": 0.35,
                "prompt_template": "default",
                "retrieval_type": "hybrid",
                "diversity_factor": 0.3,
                "chunk_overlap": 0.2,
                "context_window": 10000
            },
            "general_overview_request": {
                "top_k": 3,
                "score_threshold": 0.5,
                "embedding_model": "default",
                "reranker_weight": 0.3,
                "prompt_template": "summary",
                "retrieval_type": "semantic_only",
                "diversity_factor": 0.4,
                "chunk_overlap": 0.2,
                "context_window": 8000
            }
        }
    
    def _load_default_prompt_templates(self) -> None:
        """加载默认的提示模板配置"""
        logger.info("使用默认提示模板配置")
        self.prompt_templates = {
            "default": "根据提供的上下文信息回答问题，确保回答准确、简洁。",
            "citation": "请使用学术引用格式回答，确保每个事实都有来源引用。",
            "summary": "请生成简洁明了的总结，突出核心观点。",
            "comparison": "请对比分析不同文档中的相关内容，列出对比表。",
            "requirement_detail": "请重点关注文档中的'shall'语句和具体要求。",
            "requirement_summary": "请总结文档中的主要功能点和业务需求。",
            "novel_detail": "请基于小说内容回答，保持叙事风格一致。"
        }
    
    def _select_strategy(self, document_type: str, intent_type: str) -> Dict[str, Any]:
        """根据文档类型和查询意图选择合适的RAG策略
        
        Args:
            document_type: 文档类型
            intent_type: 查询意图类型
        
        Returns:
            选定的RAG策略配置
        """
        if not document_type or document_type == "unknown":
            document_type = "general"
        
        if not intent_type or intent_type == "unknown":
            intent_type = "specific_detail"
        
        # 构建策略ID
        # 尝试精确匹配：文档类型_意图类型
        strategy_id = f"{document_type}_{intent_type}"
        
        # 转换为系统中定义的策略ID格式
        # 将中文分隔符替换为下划线
        strategy_id = strategy_id.replace("_", "_").replace("-", "_")
        
        # 尝试匹配策略
        if strategy_id in self.strategies:
            logger.info(f"选择RAG策略: {strategy_id}")
            selected_strategy = self.strategies[strategy_id].copy()
            # 合并默认值，确保所有必要字段都存在
            self._merge_strategy_with_defaults(selected_strategy)
            return selected_strategy
        
        # 如果没有精确匹配，尝试匹配文档类型的通用策略
        general_doc_strategy_id = f"{document_type}_general"
        if general_doc_strategy_id in self.strategies:
            logger.info(f"选择文档类型通用策略: {general_doc_strategy_id}")
            selected_strategy = self.strategies[general_doc_strategy_id].copy()
            self._merge_strategy_with_defaults(selected_strategy)
            return selected_strategy
        
        # 如果还是没有匹配，尝试匹配意图类型的通用策略
        general_intent_strategy_id = f"general_{intent_type}"
        if general_intent_strategy_id in self.strategies:
            logger.info(f"选择意图类型通用策略: {general_intent_strategy_id}")
            selected_strategy = self.strategies[general_intent_strategy_id].copy()
            self._merge_strategy_with_defaults(selected_strategy)
            return selected_strategy
        
        # 最后使用默认策略
        logger.info("未找到匹配的策略，使用默认策略")
        return self.default_strategy.copy()
    
    def _merge_strategy_with_defaults(self, strategy: Dict[str, Any]) -> None:
        """将策略与默认值合并，确保所有必要字段都存在
        
        Args:
            strategy: 策略配置字典
        """
        for key, default_value in self.default_strategy.items():
            if key not in strategy:
                strategy[key] = default_value
    
    def _load_multilingual_config(self) -> Dict[str, Any]:
        """加载多语言支持配置
        
        Returns:
            多语言支持配置字典
        """
        multilingual_config_path = "config/multilingual_rag_prompts.yaml"
        try:
            if os.path.exists(multilingual_config_path):
                with open(multilingual_config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    logger.info("成功加载多语言配置")
                    return config
            else:
                logger.warning(f"多语言配置文件不存在: {multilingual_config_path}")
        except Exception as e:
            logger.error(f"加载多语言配置失败: {str(e)}")
        
        # 返回默认配置
        return {
            "supported_languages": ["zh", "en"],
            "default_language": "zh",
            "translation_required": False
        }
    
    def _get_prompt_template(self, template_id: str) -> str:
        """获取提示模板
        
        Args:
            template_id: 模板ID
        
        Returns:
            提示模板文本
        """
        if template_id in self.prompt_templates:
            return self.prompt_templates[template_id]
        
        logger.warning(f"未找到提示模板: {template_id}，使用默认模板")
        return self.default_prompt_template
    
    def _prepare_context(self, retrieved_docs: List[Any], strategy: Dict[str, Any]) -> str:
        """准备上下文信息
        
        Args:
            retrieved_docs: 检索到的文档列表
            strategy: RAG策略配置
        
        Returns:
            格式化后的上下文文本
        """
        if not retrieved_docs:
            return "没有找到相关信息。"
        
        # 应用多样性优化，避免冗余信息
        diverse_docs = self._apply_diversity_filter(retrieved_docs, strategy.get('diversity_factor', 0.3))
        
        # 构建上下文文本
        context_parts = []
        for i, doc in enumerate(diverse_docs, 1):
            # 获取文档内容和元数据，兼容Document对象和字典
            if hasattr(doc, 'page_content'):
                # Document对象
                content = doc.page_content
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            else:
                # 字典格式
                content = doc.get('page_content', '')
                metadata = doc.get('metadata', {})
            
            # 提取来源信息
            source_info = f"【来源 {i}】"
            if 'source' in metadata:
                source_info += f" {metadata['source']}"
            if 'page' in metadata:
                source_info += f" 第{metadata['page']}页"
            if 'document_type' in metadata:
                source_info += f" [{metadata['document_type']}]"
            
            # 添加到上下文部分
            context_part = f"{source_info}\n{content}\n"
            context_parts.append(context_part)
        
        # 组合上下文
        context = "\n".join(context_parts)
        
        # 根据策略调整上下文长度
        max_context_length = strategy.get('context_window', 10000)  # 默认最大上下文长度为10000字符
        if len(context) > max_context_length:
            # 使用智能截断，保留最相关的部分
            context = self._smart_truncate_context(context, max_context_length)
        
        return context
    
    def _apply_diversity_filter(self, documents: List[Any], diversity_factor: float) -> List[Any]:
        """应用多样性过滤，减少上下文冗余
        
        Args:
            documents: 文档列表
            diversity_factor: 多样性因子 (0-1)
        
        Returns:
            经过多样性过滤的文档列表
        """
        if not documents or len(documents) <= 2 or diversity_factor <= 0:
            return documents
        
        # 实现简单的多样性过滤：删除内容高度重叠的文档
        filtered_docs = [documents[0]]  # 保留最相关的文档
        
        for doc in documents[1:]:
            doc_content = self._get_doc_content(doc)
            should_keep = True
            
            # 与已保留的文档比较内容相似度
            for kept_doc in filtered_docs:
                kept_content = self._get_doc_content(kept_doc)
                similarity = self._calculate_content_similarity(doc_content, kept_content)
                
                # 如果相似度高于阈值，不保留该文档
                if similarity > (1.0 - diversity_factor):
                    should_keep = False
                    break
            
            if should_keep:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def _get_doc_content(self, doc: Any) -> str:
        """从文档对象中提取内容
        
        Args:
            doc: 文档对象
        
        Returns:
            文档内容
        """
        if hasattr(doc, 'page_content'):
            return doc.page_content
        elif isinstance(doc, dict):
            return doc.get('page_content', '')
        return str(doc)
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算两个内容的相似度
        
        Args:
            content1: 第一个内容
            content2: 第二个内容
        
        Returns:
            相似度分数 (0-1)
        """
        # 提取关键词
        words1 = set(re.findall(r'\w+', content1.lower()))
        words2 = set(re.findall(r'\w+', content2.lower()))
        
        # 如果任一内容为空，返回0
        if not words1 or not words2:
            return 0.0
        
        # 计算Jaccard相似度
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _smart_truncate_context(self, context: str, max_length: int) -> str:
        """智能截断上下文，保留最相关的部分
        
        Args:
            context: 上下文文本
            max_length: 最大长度
        
        Returns:
            截断后的上下文
        """
        if len(context) <= max_length:
            return context
        
        # 按来源分割上下文
        parts = context.split("【来源 ")
        if len(parts) <= 1:
            # 如果无法分割，简单截断
            return context[:max_length - 30] + "\n...（上下文过长，已截断）"
        
        # 保留第一部分（通常最相关）和尽可能多的其他部分
        result = parts[0]
        remaining_length = max_length - len(result)
        
        for i, part in enumerate(parts[1:], 1):
            full_part = f"【来源 {part}"
            if len(full_part) <= remaining_length:
                result += full_part
                remaining_length -= len(full_part)
            else:
                # 截断最后一个部分
                truncated_part = f"【来源 {part[:remaining_length - 30]}"
                result += truncated_part + "\n...（上下文过长，已截断）"
                break
        
        return result
    
    def _format_final_prompt(self, query: str, context: str, prompt_template: str, 
                           intent_type: str = None, document_type: str = None) -> Dict[str, str]:
        """格式化最终的提示
        
        Args:
            query: 用户查询
            context: 上下文信息
            prompt_template: 提示模板
            intent_type: 查询意图类型
            document_type: 文档类型
        
        Returns:
            格式化后的提示文本，包含system和user两部分
        """
        # 构建基础系统提示
        system_prompt = "你是一个智能助手，根据提供的上下文信息回答用户的问题。"
        
        # 根据文档类型和意图类型动态调整提示词
        dynamic_instructions = self._get_dynamic_instructions(intent_type, document_type)
        
        # 合并所有指令
        if prompt_template:
            system_prompt += f" {prompt_template}"
        
        if dynamic_instructions:
            system_prompt += f" {dynamic_instructions}"
        
        # 添加回答格式指导
        system_prompt += "\n请确保你的回答：\n1. 完全基于提供的上下文信息\n2. 准确无误，避免编造信息\n3. 结构清晰，易于理解\n4. 如有引用，请标明来源"
        
        # 构建用户提示
        user_prompt = f"上下文信息：\n{context}\n\n问题：{query}\n\n回答："
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _get_dynamic_instructions(self, intent_type: str, document_type: str) -> str:
        """根据意图类型和文档类型获取动态指令
        
        Args:
            intent_type: 查询意图类型
            document_type: 文档类型
        
        Returns:
            动态指令文本
        """
        instructions = []
        
        # 根据意图类型添加指令
        if intent_type:
            if 'overview' in intent_type:
                instructions.append("请提供全面且简洁的概述，突出核心观点和关键信息。")
            elif 'specific' in intent_type:
                instructions.append("请提供精确的细节和具体数据，确保信息准确无误。")
            elif 'comparison' in intent_type:
                instructions.append("请清晰对比不同方案或概念的优缺点和差异。")
            elif 'analysis' in intent_type:
                instructions.append("请深入分析问题，提供见解和推理过程。")
        
        # 根据文档类型添加指令
        if document_type:
            if 'academic' in document_type:
                instructions.append("请注意学术严谨性，引用关键研究发现和结论。")
            elif 'requirement' in document_type:
                instructions.append("请关注功能性需求和非功能性需求，特别注意shall语句。")
            elif 'contract' in document_type:
                instructions.append("请准确解释合同条款和法律义务，避免模糊表述。")
            elif 'technical' in document_type:
                instructions.append("请使用精确的技术术语，提供明确的技术细节。")
        
        return " ".join(instructions) if instructions else ""
    
    def answer_query(self, query: str, document_type: str = None, user_id: str = None, 
                    session_id: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """回答用户查询
        
        Args:
            query: 用户查询
            document_type: 文档类型（可选）
            user_id: 用户ID（可选）
            session_id: 会话ID（可选）
            metadata: 额外元数据（可选）
        
        Returns:
            包含回答和相关信息的字典
        """
        # 验证查询
        if not query or not query.strip():
            logger.error("查询内容为空")
            return {
                'answer': "抱歉，我无法处理空的查询。",
                'success': False,
                'error': "查询内容为空",
                'processing_time': 0
            }
        
        # 标准化查询
        query = query.strip()
        
        start_time = time.time()
        
        try:
            # 检查缓存是否有相同查询的结果
            cache_key = self._generate_cache_key(query, document_type)
            if cache_key in self.retrieval_cache:
                cached_result = self.retrieval_cache[cache_key]
                logger.info(f"使用缓存结果回答查询")
                cached_result['processing_time'] = time.time() - start_time
                cached_result['from_cache'] = True
                return cached_result
            
            # 1. 查询意图识别
            intent_result = query_intent_classifier.classify_intent(query, document_type)
            intent_type = intent_result['intent_type']
            
            # 提取查询关键词
            keywords = query_intent_classifier.extract_keywords(query)
            
            # 如果没有提供文档类型，尝试从向量存储中获取相关文档类型
            if not document_type or document_type == "unknown":
                # 搜索相关文档以推断文档类型
                search_results = vector_store_manager.similarity_search(query, k=5)
                if search_results:
                    # 统计最常见的文档类型
                    doc_type_count = {}
                    for doc in search_results:
                        doc_type = doc.metadata.get('document_type', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                        doc_type_count[doc_type] = doc_type_count.get(doc_type, 0) + 1
                    
                    # 选择最常见的文档类型
                    if doc_type_count:
                        document_type = max(doc_type_count, key=doc_type_count.get)
                        logger.info(f"从搜索结果推断文档类型: {document_type}")
            
            # 2. 选择RAG策略
            strategy = self._select_strategy(document_type or "general", intent_type)
            
            # 3. 使用混合检索策略获取相关文档
            try:
                retrieved_docs = self._perform_hybrid_retrieval(query, strategy)
            except Exception as retrieval_error:
                logger.error(f"检索失败，尝试备用检索方法: {str(retrieval_error)}")
                # 使用备用检索方法
                retrieved_docs = self._fallback_retrieval(query, strategy)
            
            # 4. 准备上下文
            context = self._prepare_context(retrieved_docs, strategy)
            
            # 5. 获取提示模板
            prompt_template = self._get_prompt_template(strategy.get('prompt_template', 'default'))
            
            # 6. 格式化最终提示（增强版，加入意图和文档类型信息）
            formatted_prompt = self._format_final_prompt(
                query, context, prompt_template, intent_type, document_type
            )
            
            # 7. 调用LLM生成回答
            llm_result = self._generate_answer_with_retry(formatted_prompt)
            
            # 8. 构建返回结果
            processing_time = time.time() - start_time
            
            # 处理LLM结果
            result = {
                'answer': llm_result if llm_result else '生成回答失败',
                'success': bool(llm_result),
                'processing_time': processing_time,
                'intent_analysis': intent_result,
                'document_type': document_type or "general",
                'strategy': strategy,
                'retrieved_documents': len(retrieved_docs),
                'query_keywords': keywords,
                'metadata': metadata or {},
                'retrieval_type': strategy.get('retrieval_type', 'hybrid')
            }
            
            # 如果有引用信息，添加到结果中
            if retrieved_docs:
                result['references'] = []
                for i, doc in enumerate(retrieved_docs[:3], 1):  # 只返回前3个引用
                    # 使用属性访问方式处理Document对象
                    metadata = getattr(doc, 'metadata', {})
                    page_content = getattr(doc, 'page_content', '')
                    # 尝试多种方式获取文档相似度分数
                    if hasattr(doc, 'score'):
                        score = doc.score
                    elif isinstance(doc, dict) and 'score' in doc:
                        score = doc['score']
                    else:
                        # 如果是元组形式的结果 (document, score)
                        try:
                            if len(doc) > 1 and isinstance(doc[1], (int, float)):
                                score = doc[1]
                            else:
                                # 使用合理的默认值而不是0.0
                                score = 0.7
                        except:
                            score = 0.7
                    
                    reference = {
                        'id': i,
                        'source': metadata.get('source', 'unknown') if isinstance(metadata, dict) else 'unknown',
                        'score': score,
                        'snippet': page_content[:100] + '...' if len(page_content) > 100 else page_content,
                        'document_type': metadata.get('document_type', 'unknown') if isinstance(metadata, dict) else 'unknown'
                    }
                    result['references'].append(reference)
            
            # 将结果添加到缓存
            self._update_cache(cache_key, result)
            
            logger.info(f"成功回答查询，处理时间: {processing_time:.2f}秒，检索文档数: {len(retrieved_docs)}")
            
            return result
            
        except Exception as e:
            logger.error(f"回答查询失败: {str(e)}", exc_info=True)
            processing_time = time.time() - start_time
            return {
                'answer': "抱歉，处理您的查询时出错了。",
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'error_type': type(e).__name__
            }
    
    def _generate_cache_key(self, query: str, document_type: str = None) -> str:
        """生成缓存键
        
        Args:
            query: 查询文本
            document_type: 文档类型
        
        Returns:
            缓存键字符串
        """
        return f"{query}_{document_type or 'general'}"
    
    def _update_cache(self, key: str, value: Dict[str, Any]) -> None:
        """更新检索缓存
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        # 如果缓存已满，删除最早的项
        if len(self.retrieval_cache) >= self.max_cache_size:
            # 简单的FIFO策略
            oldest_key = next(iter(self.retrieval_cache))
            del self.retrieval_cache[oldest_key]
        
        # 添加新项到缓存
        self.retrieval_cache[key] = value
    
    def _perform_hybrid_retrieval(self, query: str, strategy: Dict[str, Any]) -> List[Any]:
        """执行混合检索策略
        
        Args:
            query: 查询文本
            strategy: RAG策略配置
        
        Returns:
            检索到的文档列表
        """
        retrieval_type = strategy.get('retrieval_type', 'hybrid')
        top_k = strategy.get('top_k', 4)
        score_threshold = strategy.get('score_threshold', 0.5)
        
        if retrieval_type == 'semantic_only':
            # 仅语义检索
            return vector_store_manager.similarity_search(query, k=top_k, score_threshold=score_threshold)
        elif retrieval_type == 'keyword_only':
            # 仅关键词检索（如果向量存储支持）
            try:
                return vector_store_manager.keyword_search(query, k=top_k)
            except (AttributeError, NotImplementedError):
                # 如果不支持关键词检索，回退到语义检索
                logger.warning("关键词检索不支持，回退到语义检索")
                return vector_store_manager.similarity_search(query, k=top_k, score_threshold=score_threshold)
        else:  # hybrid
            # 混合检索：语义+关键词
            semantic_results = vector_store_manager.similarity_search(query, k=top_k*2, score_threshold=0)
            
            # 尝试关键词检索
            keyword_results = []
            try:
                keyword_results = vector_store_manager.keyword_search(query, k=top_k*2)
            except (AttributeError, NotImplementedError):
                logger.warning("关键词检索不支持，仅使用语义检索结果")
                keyword_results = []
            
            # 合并结果
            return self._merge_retrieval_results(semantic_results, keyword_results, strategy)
    
    def _merge_retrieval_results(self, semantic_results: List[Any], keyword_results: List[Any], 
                                strategy: Dict[str, Any]) -> List[Any]:
        """合并不同检索方法的结果
        
        Args:
            semantic_results: 语义检索结果
            keyword_results: 关键词检索结果
            strategy: RAG策略配置
        
        Returns:
            合并后的文档列表
        """
        # 获取检索策略配置
        retrieval_config = self.retrieval_strategies.get(strategy.get('retrieval_type', 'hybrid'), {})
        semantic_weight = retrieval_config.get('semantic_weight', 0.7)
        keyword_weight = retrieval_config.get('keyword_weight', 0.3)
        
        # 如果没有关键词结果，直接返回语义结果
        if not keyword_results:
            return semantic_results[:strategy.get('top_k', 4)]
        
        # 为每个文档分配唯一ID以便去重
        doc_id_map = {}
        merged_results = []
        
        # 处理语义结果
        for i, doc in enumerate(semantic_results):
            doc_id = self._get_doc_id(doc)
            if doc_id not in doc_id_map:
                doc_score = 1.0 - (i / len(semantic_results)) if semantic_results else 0
                weighted_score = doc_score * semantic_weight
                doc_id_map[doc_id] = (doc, weighted_score)
                
        # 处理关键词结果
        for i, doc in enumerate(keyword_results):
            doc_id = self._get_doc_id(doc)
            if doc_id in doc_id_map:
                # 如果文档已在语义结果中，增加其权重
                existing_doc, existing_score = doc_id_map[doc_id]
                keyword_score = 1.0 - (i / len(keyword_results)) if keyword_results else 0
                total_score = existing_score + (keyword_score * keyword_weight)
                doc_id_map[doc_id] = (existing_doc, total_score)
            else:
                # 如果是新文档，添加到映射中
                keyword_score = 1.0 - (i / len(keyword_results)) if keyword_results else 0
                weighted_score = keyword_score * keyword_weight
                doc_id_map[doc_id] = (doc, weighted_score)
        
        # 转换为列表并排序
        for doc, score in doc_id_map.values():
            # 设置分数属性
            if hasattr(doc, 'score'):
                doc.score = score
            elif isinstance(doc, dict):
                doc['score'] = score
            merged_results.append((doc, score))
        
        # 按分数排序
        merged_results.sort(key=lambda x: x[1], reverse=True)
        
        # 提取文档并限制数量
        top_k = strategy.get('top_k', 4)
        return [doc for doc, score in merged_results[:top_k]]
    
    def _get_doc_id(self, doc: Any) -> str:
        """获取文档的唯一ID
        
        Args:
            doc: 文档对象
        
        Returns:
            文档ID字符串
        """
        # 尝试从元数据获取ID
        metadata = getattr(doc, 'metadata', {}) if not isinstance(doc, dict) else doc.get('metadata', {})
        
        if 'id' in metadata:
            return str(metadata['id'])
        elif 'source' in metadata:
            source_str = str(metadata['source'])
            if 'page' in metadata:
                return f"{source_str}_page_{metadata['page']}"
            return source_str
        
        # 如果没有元数据，使用内容的哈希值
        content = self._get_doc_content(doc)
        return str(hash(content) % 1000000)  # 使用哈希值的模数作为ID
    
    def _fallback_retrieval(self, query: str, strategy: Dict[str, Any]) -> List[Any]:
        """备用检索方法
        
        Args:
            query: 查询文本
            strategy: RAG策略配置
        
        Returns:
            检索到的文档列表
        """
        top_k = strategy.get('top_k', 4)
        # 降低阈值，扩大检索范围
        return vector_store_manager.similarity_search(query, k=top_k*2, score_threshold=0.1)
    
    def _generate_answer_with_retry(self, formatted_prompt: Dict[str, str], max_retries: int = 3) -> str:
        """带重试机制的回答生成
        
        Args:
            formatted_prompt: 格式化的提示词
            max_retries: 最大重试次数
        
        Returns:
            生成的回答
        """
        retries = 0
        last_error = None
        
        while retries < max_retries:
            try:
                # 关键点：临时修改global_config.SYSTEM_PROMPT以传递优化的系统提示
                from src.config import global_config
                original_system_prompt = getattr(global_config, 'SYSTEM_PROMPT', '')
                global_config.SYSTEM_PROMPT = formatted_prompt['system']
                
                try:
                    llm_result = self.llm_client.generate_response(formatted_prompt['user'])
                    return llm_result
                finally:
                    # 恢复原始的system prompt，避免影响后续查询
                    if hasattr(global_config, 'SYSTEM_PROMPT'):
                        global_config.SYSTEM_PROMPT = original_system_prompt
                    else:
                        # 如果原本没有SYSTEM_PROMPT属性，删除我们添加的
                        if hasattr(global_config, 'SYSTEM_PROMPT'):
                            delattr(global_config, 'SYSTEM_PROMPT')
            
            except Exception as e:
                last_error = e
                retries += 1
                logger.warning(f"生成回答失败，正在重试 ({retries}/{max_retries}): {str(e)}")
                
                # 指数退避
                import time
                time.sleep(0.5 * (2 ** (retries - 1)))
        
        logger.error(f"生成回答失败，已达到最大重试次数: {str(last_error)}")
        raise last_error
    
    def batch_process_queries(self, queries: List[Dict[str, Any]], batch_size: int = 5) -> List[Dict[str, Any]]:
        """批量处理查询
        
        Args:
            queries: 查询列表，每个查询是包含查询内容和其他信息的字典
            batch_size: 批处理大小
        
        Returns:
            包含所有查询结果的列表
        """
        if not queries:
            return []
        
        results = []
        total_queries = len(queries)
        
        # 分批处理查询
        for i in range(0, total_queries, batch_size):
            batch = queries[i:i + batch_size]
            
            # 处理批次中的每个查询
            for query_info in batch:
                query = query_info.get('query', '')
                document_type = query_info.get('document_type')
                user_id = query_info.get('user_id')
                session_id = query_info.get('session_id')
                metadata = query_info.get('metadata', {})
                
                # 处理单个查询
                result = self.answer_query(query, document_type, user_id, session_id, metadata)
                
                # 添加查询ID（如果有）
                if 'id' in query_info:
                    result['query_id'] = query_info['id']
                
                results.append(result)
            
            logger.info(f"已处理 {min(i + batch_size, total_queries)} / {total_queries} 个查询")
        
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取RAG流水线的健康状态
        
        Returns:
            健康状态信息
        """
        # 检查核心组件的健康状态
        health_status = {
            'pipeline_status': 'healthy',
            'components': {
                'document_classifier': 'healthy',
                'query_intent_classifier': 'healthy',
                'vector_store': 'healthy',
                'llm_client': 'healthy'
            },
            'strategy_count': len(self.strategies),
            'template_count': len(self.prompt_templates),
            'cache_stats': {
                'document_classifier': document_classifier.get_cache_stats(),
                'query_intent_classifier': query_intent_classifier.get_cache_stats()
            }
        }
        
        # 尝试执行简单的健康检查
        try:
            # 检查文档分类器
            test_result = document_classifier.classify_document("test_doc", "这是一个测试文档内容")
            if not test_result or test_result.get('document_type') == 'unknown':
                health_status['components']['document_classifier'] = 'degraded'
                health_status['pipeline_status'] = 'degraded'
            
            # 检查查询意图分类器
            intent_result = query_intent_classifier.classify_intent("这是什么？")
            if not intent_result or intent_result.get('intent_type') == 'unknown':
                health_status['components']['query_intent_classifier'] = 'degraded'
                health_status['pipeline_status'] = 'degraded'
            
        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            health_status['pipeline_status'] = 'error'
            health_status['error'] = str(e)
        
        return health_status
    
    def update_strategies(self, strategies: Dict[str, Any], update_file: bool = False, config_path: str = None) -> bool:
        """更新RAG策略配置
        
        Args:
            strategies: 新的策略配置
            update_file: 是否同时更新配置文件
            config_path: 配置文件路径（可选）
        
        Returns:
            是否更新成功
        """
        try:
            # 更新内存中的策略配置
            self.strategies.update(strategies)
            logger.info(f"成功更新 {len(strategies)} 种RAG策略配置")
            
            # 如果需要更新配置文件
            if update_file:
                config_path = config_path or "config/rag_strategies.yaml"
                
                # 读取现有配置文件
                existing_config = {}
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        existing_config = yaml.safe_load(f) or {}
                
                # 更新策略配置
                existing_config['strategies'] = existing_config.get('strategies', {})
                existing_config['strategies'].update(strategies)
                
                # 写回配置文件
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(existing_config, f, allow_unicode=True, default_flow_style=False)
                
                logger.info(f"成功更新RAG策略配置文件: {config_path}")
            
            return True
        except Exception as e:
            logger.error(f"更新RAG策略配置失败: {str(e)}")
            return False

# 导入时间模块（在answer_query方法中使用）
import time

# 创建全局自适应RAG流水线实例
adaptive_rag_pipeline = AdaptiveRAGPipeline()

# 示例用法
if __name__ == "__main__":
    # 测试单个查询
    print("\n=== 测试单个查询 ===")
    test_query = "什么是机器学习？"
    result = adaptive_rag_pipeline.answer_query(test_query)
    
    print(f"查询: {test_query}")
    print(f"回答: {result['answer']}")
    print(f"处理时间: {result['processing_time']:.2f}秒")
    print(f"意图类型: {result['intent_analysis']['intent_type']} ({result['intent_analysis']['intent_name']})")
    print(f"文档类型: {result['document_type']}")
    print(f"检索文档数: {result['retrieved_documents']}")
    
    # 打印策略信息
    print(f"使用的策略: {result['strategy']}")
    
    # 打印引用信息（如果有）
    if 'references' in result:
        print("引用信息:")
        for ref in result['references']:
            print(f"  - 来源 {ref['id']}: {ref['source']} (相关度: {ref['score']:.2f})")
            print(f"    片段: {ref['snippet']}")
    
    # 测试健康状态
    print("\n=== 测试健康状态 ===")
    health_status = adaptive_rag_pipeline.get_health_status()
    print(f"流水线状态: {health_status['pipeline_status']}")
    print("组件状态:")
    for component, status in health_status['components'].items():
        print(f"  - {component}: {status}")
    
    # 测试批量处理（简单示例）
    print("\n=== 测试批量处理 ===")
    batch_queries = [
        {'query': '什么是人工智能？'},
        {'query': '总结一下机器学习的主要算法。'}
    ]
    batch_results = adaptive_rag_pipeline.batch_process_queries(batch_queries)
    
    for i, result in enumerate(batch_results):
        print(f"\n查询 {i+1} 结果:")
        print(f"回答: {result['answer'][:100]}...")
        print(f"处理时间: {result['processing_time']:.2f}秒")