import os
import logging
from typing import List, Dict, Any, Optional
from src.document_loader import document_loader
from src.vector_store import vector_store_manager
from src.llm_client import llm_client
from src.config import global_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG（检索增强生成）流水线，整合文档加载、向量存储和LLM调用"""
    
    def __init__(self):
        """初始化RAG流水线"""
        self.document_loader = document_loader
        self.vector_store_manager = vector_store_manager
        self.llm_client = llm_client
        self.conversation_history: List[Dict[str, str]] = []
    
    def process_documents(self, directory_path=None) -> bool:
        """处理指定目录下的所有文档并添加到向量存储
        
        Args:
            directory_path: 文档目录路径，如果为None则使用配置中的路径
            
        Returns:
            处理是否成功
        """
        try:
            # 加载文档
            documents = self.document_loader.load_directory(directory_path)
            if not documents:
                logger.warning("没有找到或加载任何文档")
                return False
            
            # 分割文档
            split_docs = self.document_loader.split_documents(documents)
            if not split_docs:
                logger.warning("文档分割失败")
                return False
            
            # 添加到向量存储
            return self.vector_store_manager.add_documents(split_docs)
        except Exception as e:
            logger.error(f"处理文档失败: {str(e)}")
            return False
    
    def add_single_document(self, file_path: str) -> bool:
        """添加单个文档到向量存储
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            添加是否成功
        """
        try:
            # 加载文档
            documents = self.document_loader.load_document(file_path)
            if not documents:
                logger.warning(f"没有找到或加载文档: {file_path}")
                return False
            
            # 分割文档
            split_docs = self.document_loader.split_documents(documents)
            if not split_docs:
                logger.warning(f"文档分割失败: {file_path}")
                return False
            
            # 添加到向量存储
            return self.vector_store_manager.add_documents(split_docs)
        except Exception as e:
            logger.error(f"添加文档失败: {file_path}, 错误: {str(e)}")
            return False
    
    def preprocess_query(self, query: str) -> str:
        """预处理用户问题，识别概述类问题并进行转换
        
        Args:
            query: 用户原始查询
        
        Returns:
            预处理后的查询
        """
        # 概述类关键词列表
        overview_keywords = ["讲述", "介绍", "概述", "总结", "主要内容", "是什么", "讲什么"]
        
        # 检查是否包含概述类关键词
        if any(kw in query for kw in overview_keywords):
            # 对于概述类问题，引导模型提取关键信息
            return "请根据上下文，提取文档标题、版本、作者、发布日期等关键信息进行简要介绍。"
        
        return query
    
    def answer_query(self, query: str, use_history: bool = True, k: int = 3) -> Optional[str]:
        """回答用户查询
        
        Args:
            query: 用户查询
            use_history: 是否使用对话历史
            k: 检索的相关文档数量
        
        Returns:
            生成的回答，如果失败则返回None
        """
        try:
            # 预处理用户问题
            processed_query = self.preprocess_query(query)
            
            # 检查是否有可用的向量存储
            vector_count = self.vector_store_manager.get_vector_count()
            if vector_count == 0:
                logger.warning("向量存储为空，请先处理文档")
                # 即使没有向量存储，也尝试直接回答
                history = self.conversation_history if use_history else None
                return self.llm_client.generate_response(processed_query, context=None, history=history)
            
            # 进行相似度搜索，获取相关文档
            relevant_docs = self.vector_store_manager.similarity_search(query, k=k)
            
            # 提取相关文档的内容
            context = [doc.page_content for doc in relevant_docs]
            
            # 准备对话历史
            history = self.conversation_history if use_history else None
            
            # 调用LLM生成回答
            answer = self.llm_client.generate_response(processed_query, context=context, history=history)
            
            # 更新对话历史
            if answer and use_history:
                self.conversation_history.append({"role": "user", "content": query})
                self.conversation_history.append({"role": "assistant", "content": answer})
                
                # 限制对话历史长度
                max_history_length = 10  # 最多保留10轮对话
                if len(self.conversation_history) > max_history_length * 2:
                    self.conversation_history = self.conversation_history[-max_history_length * 2:]
            
            return answer
        except Exception as e:
            logger.error(f"回答查询失败: {str(e)}")
            return None
    
    def clear_conversation_history(self) -> None:
        """清空对话历史"""
        self.conversation_history = []
        logger.info("对话历史已清空")
    
    def clear_vector_store(self) -> bool:
        """清空向量存储"""
        result = self.vector_store_manager.clear_vector_store()
        if result:
            logger.info("向量存储已清空")
        return result
    
    def get_vector_count(self) -> int:
        """获取向量存储中的向量数量"""
        return self.vector_store_manager.get_vector_count()
    
    def validate_api_key(self) -> bool:
        """验证API密钥是否有效"""
        return self.llm_client.validate_api_key()

# 创建RAG流水线实例
rag_pipeline = RAGPipeline()