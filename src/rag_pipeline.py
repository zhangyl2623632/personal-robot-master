import os
import logging
import time
import threading
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
        self._history_lock = threading.RLock()  # 用于保护对话历史的线程锁
        self._health_status = {
            'document_loader': True,
            'vector_store': True,
            'llm_client': True
        }
        self._last_health_check = 0
        self._health_check_interval = 60  # 健康检查间隔（秒）
    
    def _health_check(self, force=False) -> Dict[str, bool]:
        """检查各组件健康状态"""
        current_time = time.time()
        
        # 定期或强制进行健康检查
        if force or current_time - self._last_health_check > self._health_check_interval:
            logger.debug("执行组件健康检查")
            
            try:
                # 检查文档加载器
                self._health_status['document_loader'] = hasattr(self.document_loader, 'load_document')
            except Exception as e:
                logger.error(f"文档加载器健康检查失败: {str(e)}")
                self._health_status['document_loader'] = False
            
            try:
                # 检查向量存储
                vector_count = self.vector_store_manager.get_vector_count()
                self._health_status['vector_store'] = vector_count >= 0  # -1 表示错误
            except Exception as e:
                logger.error(f"向量存储健康检查失败: {str(e)}")
                self._health_status['vector_store'] = False
            
            try:
                # 检查LLM客户端
                self._health_status['llm_client'] = self.llm_client.validate_api_key()
            except Exception as e:
                logger.error(f"LLM客户端健康检查失败: {str(e)}")
                self._health_status['llm_client'] = False
            
            self._last_health_check = current_time
        
        return self._health_status.copy()
    
    def get_health_status(self, force_check=False) -> Dict[str, Any]:
        """获取系统健康状态"""
        status = self._health_check(force_check)
        
        return {
            'status': 'healthy' if all(status.values()) else 'unhealthy',
            'components': status,
            'vector_count': self.get_vector_count(),
            'conversation_history_length': len(self.conversation_history)
        }
    
    def process_documents(self, directory_path=None) -> bool:
        """处理指定目录下的所有文档并添加到向量存储
        
        Args:
            directory_path: 文档目录路径，如果为None则使用配置中的路径
            
        Returns:
            处理是否成功
        """
        try:
            # 检查目录是否存在
            target_path = directory_path or global_config.DOCUMENTS_PATH
            if not target_path or not os.path.exists(target_path):
                logger.error(f"文档目录不存在: {target_path}")
                return False
            
            # 检查目录权限
            if not os.access(target_path, os.R_OK):
                logger.error(f"没有读取文档目录的权限: {target_path}")
                return False
            
            # 加载文档
            documents = self.document_loader.load_directory(target_path)
            if not documents:
                logger.warning("没有找到或加载任何文档")
                return True  # 没有文档不是错误状态
            
            # 分割文档
            split_docs = self.document_loader.split_documents(documents)
            if not split_docs:
                logger.warning("文档分割失败")
                return False
            
            # 添加到向量存储，添加重试机制
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    success = self.vector_store_manager.add_documents(split_docs)
                    if success:
                        logger.info(f"成功添加 {len(split_docs)} 个文档片段到向量存储")
                    else:
                        logger.warning(f"添加文档到向量存储失败，正在重试 ({retry_count+1}/{max_retries})")
                        retry_count += 1
                        time.sleep(1)  # 简单退避
                except Exception as e:
                    logger.error(f"添加文档到向量存储时发生异常: {str(e)}")
                    retry_count += 1
                    time.sleep(1)
            
            return success
        except Exception as e:
            logger.error(f"处理文档失败: {str(e)}", exc_info=True)
            return False
    
    def add_single_document(self, file_path: str) -> bool:
        """添加单个文档到向量存储
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            添加是否成功
        """
        try:
            # 检查文件是否存在
            if not file_path or not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return False
            
            # 检查文件权限
            if not os.access(file_path, os.R_OK):
                logger.error(f"没有读取文件的权限: {file_path}")
                return False
            
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
            logger.error(f"添加文档失败: {file_path}, 错误: {str(e)}", exc_info=True)
            return False
    
    def preprocess_query(self, query: str) -> str:
        """预处理用户问题，识别不同类型的问题并进行转换
        
        Args:
            query: 用户原始查询
        
        Returns:
            预处理后的查询
        """
        # 规范化查询
        query = query.strip()
        
        # 概述类关键词列表
        overview_keywords = ["讲述", "介绍", "概述", "总结", "主要内容", "是什么", "讲什么"]
        
        # 细节类关键词列表
        detail_keywords = ["具体", "详细", "数据", "参数", "步骤", "如何", "怎么"]
        
        # 检查是否包含概述类关键词
        if any(kw in query for kw in overview_keywords):
            # 对于概述类问题，引导模型提取关键信息
            return f"请根据上下文，详细回答关于'{query}'的问题，包括主要内容、关键点和重要细节。"
        
        # 检查是否包含细节类关键词
        if any(kw in query for kw in detail_keywords):
            # 对于细节类问题，引导模型提供具体信息
            return f"请根据上下文，准确回答关于'{query}'的具体问题，提供精确的数据、参数或步骤。"
        
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
            # 检查查询是否为空
            if not query or not query.strip():
                logger.warning("空查询，无法回答")
                return "很抱歉，您的查询不能为空。请输入有效的问题。"
            
            # 预处理用户问题
            processed_query = self.preprocess_query(query)
            
            # 检查组件健康状态
            health_status = self._health_check()
            
            # 准备对话历史
            history = self.get_conversation_history() if use_history else None
            
            # 尝试获取相关文档
            context = None
            try:
                # 检查是否有可用的向量存储
                vector_count = self.vector_store_manager.get_vector_count()
                if vector_count > 0:
                    # 进行相似度搜索，获取相关文档
                    relevant_docs = self.vector_store_manager.similarity_search(query, k=k)
                    if relevant_docs:
                        # 提取相关文档的内容
                        context = [doc.page_content for doc in relevant_docs]
                        logger.debug(f"找到了 {len(relevant_docs)} 个相关文档片段")
                    else:
                        logger.warning("没有找到相关文档")
                else:
                    logger.warning("向量存储为空，将尝试直接回答")
            except Exception as e:
                logger.error(f"搜索相关文档失败: {str(e)}")
                # 即使搜索失败，也尝试继续回答
            
            # 调用LLM生成回答，添加重试机制
            max_retries = 2
            retry_count = 0
            answer = None
            
            while retry_count < max_retries and not answer:
                try:
                    answer = self.llm_client.generate_response(processed_query, context=context, history=history)
                    if not answer:
                        logger.warning(f"LLM生成回答失败，正在重试 ({retry_count+1}/{max_retries})")
                        retry_count += 1
                        time.sleep(0.5)  # 简单退避
                except Exception as e:
                    logger.error(f"调用LLM时发生异常: {str(e)}")
                    retry_count += 1
                    time.sleep(0.5)
            
            # 如果仍然没有回答，提供默认响应
            if not answer:
                if not health_status['llm_client']:
                    answer = "很抱歉，当前AI服务不可用，请稍后再试。"
                elif not health_status['vector_store']:
                    answer = "很抱歉，文档检索服务暂时不可用，请稍后再试。"
                else:
                    answer = "很抱歉，我无法为您生成回答。请尝试重新表述您的问题。"
            
            # 更新对话历史
            if use_history:
                self.add_to_conversation_history("user", query)
                self.add_to_conversation_history("assistant", answer)
            
            return answer
        except Exception as e:
            logger.error(f"回答查询失败: {str(e)}", exc_info=True)
            return "很抱歉，处理您的请求时发生错误，请稍后再试。"
    
    def add_to_conversation_history(self, role: str, content: str) -> None:
        """添加消息到对话历史，线程安全"""
        with self._history_lock:
            self.conversation_history.append({"role": role, "content": content})
            
            # 智能限制对话历史长度，根据内容重要性保留
            max_history_length = 10  # 最多保留10轮对话
            if len(self.conversation_history) > max_history_length * 2:
                # 保留最后max_history_length轮对话
                self.conversation_history = self.conversation_history[-max_history_length * 2:]
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史，线程安全"""
        with self._history_lock:
            return self.conversation_history.copy()
    
    def clear_conversation_history(self) -> None:
        """清空对话历史"""
        with self._history_lock:
            self.conversation_history = []
        logger.info("对话历史已清空")
    
    def clear_vector_store(self) -> bool:
        """清空向量存储"""
        try:
            # 先进行备份
            backup_path = f"{global_config.VECTOR_STORE_PATH}_backup_{int(time.time())}"
            if os.path.exists(global_config.VECTOR_STORE_PATH):
                import shutil
                shutil.copytree(global_config.VECTOR_STORE_PATH, backup_path)
                logger.info(f"向量存储已备份到: {backup_path}")
            
            result = self.vector_store_manager.clear_vector_store()
            if result:
                logger.info("向量存储已清空")
            return result
        except Exception as e:
            logger.error(f"清空向量存储失败: {str(e)}", exc_info=True)
            return False
    
    def get_vector_count(self) -> int:
        """获取向量存储中的向量数量"""
        try:
            return self.vector_store_manager.get_vector_count()
        except Exception as e:
            logger.error(f"获取向量数量失败: {str(e)}")
            return -1  # 返回-1表示错误
    
    def validate_api_key(self) -> bool:
        """验证API密钥是否有效"""
        try:
            return self.llm_client.validate_api_key()
        except Exception as e:
            logger.error(f"验证API密钥失败: {str(e)}")
            return False

# 添加缺失的导入
import os

# 创建RAG流水线实例
rag_pipeline = RAGPipeline()