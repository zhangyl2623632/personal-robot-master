import logging
import os
import yaml
from typing import Dict, List, Tuple, Optional, Any, Union

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
        
        # 默认策略（当没有匹配的策略时使用）
        self.default_strategy = {
            "top_k": 4,
            "score_threshold": 0.5,
            "embedding_model": "default",
            "reranker_weight": 0.3,
            "prompt_template": "default"
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
                    
        except Exception as e:
            logger.error(f"加载RAG策略配置文件失败: {str(e)}")
            # 加载默认策略配置
            self._load_default_strategies()
            self._load_default_prompt_templates()
    
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
                "prompt_template": "citation"
            },
            "academic_paper_overview_request": {
                "top_k": 3,
                "score_threshold": 0.5,
                "embedding_model": "default",
                "reranker_weight": 0.3,
                "prompt_template": "summary"
            },
            # 需求文档相关策略
            "requirement_doc_specific_detail": {
                "top_k": 5,
                "score_threshold": 0.6,
                "embedding_model": "default",
                "reranker_weight": 0.4,
                "prompt_template": "requirement_detail"
            },
            "requirement_doc_overview_request": {
                "top_k": 2,
                "score_threshold": 0.5,
                "embedding_model": "default",
                "reranker_weight": 0.3,
                "prompt_template": "requirement_summary"
            },
            # 通用策略
            "general_specific_detail": {
                "top_k": 4,
                "score_threshold": 0.55,
                "embedding_model": "default",
                "reranker_weight": 0.35,
                "prompt_template": "default"
            },
            "general_overview_request": {
                "top_k": 3,
                "score_threshold": 0.5,
                "embedding_model": "default",
                "reranker_weight": 0.3,
                "prompt_template": "summary"
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
            return self.strategies[strategy_id]
        
        # 如果没有精确匹配，尝试匹配文档类型的通用策略
        general_doc_strategy_id = f"{document_type}_general"
        if general_doc_strategy_id in self.strategies:
            logger.info(f"选择文档类型通用策略: {general_doc_strategy_id}")
            return self.strategies[general_doc_strategy_id]
        
        # 如果还是没有匹配，尝试匹配意图类型的通用策略
        general_intent_strategy_id = f"general_{intent_type}"
        if general_intent_strategy_id in self.strategies:
            logger.info(f"选择意图类型通用策略: {general_intent_strategy_id}")
            return self.strategies[general_intent_strategy_id]
        
        # 最后使用默认策略
        logger.info("未找到匹配的策略，使用默认策略")
        return self.default_strategy
    
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
        
        # 构建上下文文本
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
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
            
            # 添加到上下文部分
            context_part = f"{source_info}\n{content}\n"
            context_parts.append(context_part)
        
        # 组合上下文
        context = "\n".join(context_parts)
        
        # 根据策略调整上下文长度
        max_context_length = strategy.get('max_context_length', 10000)  # 默认最大上下文长度为10000字符
        if len(context) > max_context_length:
            # 如果上下文太长，只保留前面的部分
            context = context[:max_context_length] + "\n...（上下文过长，已截断）"
        
        return context
    
    def _format_final_prompt(self, query: str, context: str, prompt_template: str) -> Dict[str, str]:
        """格式化最终的提示
        
        Args:
            query: 用户查询
            context: 上下文信息
            prompt_template: 提示模板
        
        Returns:
            格式化后的提示文本，包含system和user两部分
        """
        # 构建系统提示
        system_prompt = "你是一个智能助手，根据提供的上下文信息回答用户的问题。"
        
        # 如果有提示模板，添加到系统提示中
        if prompt_template:
            system_prompt += f" {prompt_template}"
        
        # 构建用户提示
        user_prompt = f"上下文信息：\n{context}\n\n问题：{query}\n\n回答："
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
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
            
            # 3. 根据策略检索相关文档
            search_params = {
                "k": strategy.get('top_k', 4),
                "score_threshold": strategy.get('score_threshold', 0.5)
            }
            
            search_results = vector_store_manager.similarity_search(query, **search_params)
            retrieved_docs = search_results
            
            # 4. 准备上下文
            context = self._prepare_context(retrieved_docs, strategy)
            
            # 5. 获取提示模板
            prompt_template = self._get_prompt_template(strategy.get('prompt_template', 'default'))
            
            # 6. 格式化最终提示
            formatted_prompt = self._format_final_prompt(query, context, prompt_template)
            
            # 7. 调用LLM生成回答
            # 关键点：临时修改global_config.SYSTEM_PROMPT以传递优化的系统提示
            from src.config import global_config
            original_system_prompt = getattr(global_config, 'SYSTEM_PROMPT', '')
            global_config.SYSTEM_PROMPT = formatted_prompt['system']
            
            try:
                llm_result = self.llm_client.generate_response(formatted_prompt['user'])
            finally:
                # 恢复原始的system prompt，避免影响后续查询
                if hasattr(global_config, 'SYSTEM_PROMPT'):
                    global_config.SYSTEM_PROMPT = original_system_prompt
                else:
                    # 如果原本没有SYSTEM_PROMPT属性，删除我们添加的
                    delattr(global_config, 'SYSTEM_PROMPT')
            
            # 8. 构建返回结果
            processing_time = time.time() - start_time
            
            # 处理LLM结果 - generate_response返回的是字符串
            result = {
                'answer': llm_result if llm_result else '生成回答失败',
                'success': bool(llm_result),
                'processing_time': processing_time,
                'intent_analysis': intent_result,
                'document_type': document_type or "general",
                'strategy': strategy,
                'retrieved_documents': len(retrieved_docs),
                'query_keywords': keywords,
                'metadata': metadata or {}
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
                        'snippet': page_content[:100] + '...' if len(page_content) > 100 else page_content
                    }
                    result['references'].append(reference)
            
            logger.info(f"成功回答查询，处理时间: {processing_time:.2f}秒，检索文档数: {len(retrieved_docs)}")
            
            return result
            
        except Exception as e:
            logger.error(f"回答查询失败: {str(e)}")
            processing_time = time.time() - start_time
            return {
                'answer': "抱歉，处理您的查询时出错了。",
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }
    
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