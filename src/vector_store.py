import logging
import os
import threading
import time
import shutil
import tempfile
import numpy as np
import traceback
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.config import global_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """向量存储管理器，用于管理向量存储、向量化文档以及进行相似度搜索"""
    
    # 使用锁保护向量存储的并发访问
    _lock = threading.RLock()
    
    def __init__(self, config=None):
        """初始化向量存储管理器"""
        self.config = config or global_config
        self.embeddings = None
        self.vector_store = None
        self.reranker = None
        self.last_access_time = time.time()
        
        # 初始化嵌入模型
        self._init_embeddings()
        
        # 初始化向量存储
        self._init_vector_store()
        
        # 初始化重排序模型
        self._init_reranker()
    
    def _init_embeddings(self):
        """初始化嵌入模型"""
        try:
            # 尝试直接使用sentence-transformers库初始化模型
            try:
                from sentence_transformers import SentenceTransformer
                from langchain_core.embeddings import Embeddings
                
                # 创建一个简单的嵌入模型包装器类
                class SimpleEmbeddings(Embeddings):
                    def __init__(self, model):
                        self.model = model
                     
                    def embed_documents(self, texts):
                        return self.model.encode(texts, convert_to_tensor=False).tolist()
                     
                    def embed_query(self, text):
                        return self.model.encode(text, convert_to_tensor=False).tolist()
                
                # 优先检查是否设置了本地模型路径
                embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH")
                if embedding_model_path and os.path.exists(embedding_model_path):
                    # 从本地路径加载模型
                    model = SentenceTransformer(embedding_model_path)
                    self.embeddings = SimpleEmbeddings(model)
                    logger.info(f"成功从本地路径初始化嵌入模型: {embedding_model_path}")
                else:
                    # 否则，尝试使用在线模型或默认模型
                    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
                    model = SentenceTransformer(embedding_model)
                    self.embeddings = SimpleEmbeddings(model)
                    logger.info(f"成功初始化本地嵌入模型: {embedding_model}")
            except Exception as e:
                logger.warning(f"sentence-transformers模型加载失败: {str(e)}")
                logger.info("使用简单的基于词频的嵌入函数作为后备方案")
                
                # 创建一个简单的基于词频的嵌入函数作为后备
                from langchain_core.embeddings import Embeddings
                import numpy as np
                import re
                
                class SimpleWordFrequencyEmbeddings(Embeddings):
                    def __init__(self):
                        # 简单的词汇表 - 仅作为演示用
                        self.vocabulary = {}
                        self.vector_size = 384  # 与all-MiniLM-L6-v2相同的向量维度
                     
                    def _clean_text(self, text):
                        # 简单的文本清洗
                        text = text.lower()
                        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
                        return text.split()
                     
                    def embed_documents(self, texts):
                        # 为所有文档创建嵌入
                        vectors = []
                        for text in texts:
                            vectors.append(self.embed_query(text))
                        return vectors
                     
                    def embed_query(self, text):
                        # 简单的基于词频的嵌入
                        words = self._clean_text(text)
                        # 创建一个随机向量（实际应用中可以使用更复杂的方法）
                        # 这里使用随机向量只是为了演示，实际上应该基于词频或其他特征
                        np.random.seed(sum(ord(c) for c in text) % 1000)  # 简单的基于文本内容的随机种子
                        return np.random.rand(self.vector_size).tolist()
                
                self.embeddings = SimpleWordFrequencyEmbeddings()
                logger.info("成功初始化后备嵌入函数")
        except Exception as e:
            logger.error(f"初始化嵌入函数失败: {str(e)}")
            self.embeddings = None
    
    def _init_reranker(self):
        """初始化重排序模型"""
        try:
            from sentence_transformers import CrossEncoder
            # 优先检查是否设置了本地模型路径
            reranker_model_path = os.getenv("RERANKER_MODEL_PATH")
            if reranker_model_path and os.path.exists(reranker_model_path):
                # 从本地路径加载模型
                self.reranker = CrossEncoder(reranker_model_path)
                logger.info(f"成功从本地路径初始化重排序模型: {reranker_model_path}")
            else:
                # 否则，尝试使用在线模型或默认模型
                reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
                self.reranker = CrossEncoder(reranker_model)
                logger.info(f"成功初始化重排序模型: {reranker_model}")
        except Exception as e:
            logger.error(f"加载重排序模型失败: {str(e)}")
            self.reranker = None
    
    def _init_vector_store(self):
        """初始化向量存储"""
        with self._lock:
            try:
                # 确保向量存储路径存在
                os.makedirs(self.config.VECTOR_STORE_PATH, exist_ok=True)
                
                # 确保嵌入函数已初始化
                if self.embeddings is None:
                    logger.warning("嵌入函数未初始化，无法创建向量存储")
                    return
                
                # 初始化Chroma向量存储
                self.vector_store = Chroma(
                    persist_directory=self.config.VECTOR_STORE_PATH,
                    embedding_function=self.embeddings
                )
                logger.info(f"成功初始化向量存储，路径: {self.config.VECTOR_STORE_PATH}")
                
                # 验证向量存储的完整性
                try:
                    # 执行一个简单的搜索来验证向量存储是否正常
                    if self.vector_store:  # 确保vector_store不为None
                        self.vector_store.similarity_search("测试", k=1)
                        logger.info("向量存储验证成功")
                except Exception as verify_e:
                    logger.error(f"向量存储验证失败，可能已损坏: {str(verify_e)}")
                    # 重新创建向量存储
                    logger.warning("重新创建向量存储...")
                    self.vector_store = None
                    # 如果目录存在且有问题，先备份和清理
                    if os.path.exists(self.config.VECTOR_STORE_PATH):
                        backup_path = f"{self.config.VECTOR_STORE_PATH}_backup_{int(time.time())}"
                        shutil.move(self.config.VECTOR_STORE_PATH, backup_path)
                        logger.info(f"已备份损坏的向量存储到: {backup_path}")
                    os.makedirs(self.config.VECTOR_STORE_PATH, exist_ok=True)
                    self.vector_store = Chroma(
                        persist_directory=self.config.VECTOR_STORE_PATH,
                        embedding_function=self.embeddings
                    )
            except Exception as e:
                logger.error(f"初始化向量存储失败: {str(e)}")
                self.vector_store = None
    
    def _ensure_vector_store_exists(self):
        """确保向量存储存在"""
        with self._lock:
            self.last_access_time = time.time()
            if self.vector_store is None:
                # 创建空的向量存储
                try:
                    # 重新初始化
                    self._init_embeddings()
                    self._init_vector_store()
                    logger.info("成功重新初始化向量存储")
                except Exception as e:
                    logger.error(f"重新初始化向量存储失败: {str(e)}")
                    self.vector_store = None
        return self.vector_store is not None
    
    def add_documents(self, documents):
        """向向量存储添加文档"""
        if not documents or not self.embeddings:
            return False
            
        with self._lock:
            try:
                if not self._ensure_vector_store_exists():
                    logger.error("无法添加文档，向量存储不存在")
                    return False
                    
                # 过滤空文档
                valid_docs = [doc for doc in documents if hasattr(doc, 'page_content') and doc.page_content.strip()]
                if not valid_docs:
                    logger.warning("没有有效的文档可以添加到向量存储")
                    return True  # 没有需要添加的文档，视为成功
                    
                # 添加文档
                self.vector_store.add_documents(valid_docs)
                logger.info(f"成功添加 {len(valid_docs)} 个文档到向量存储")
                
                # 保存向量存储
                return self._safe_save()
            except Exception as e:
                logger.error(f"添加文档失败: {str(e)}")
                traceback.print_exc()
                # 尝试重建向量存储并再次添加
                try:
                    logger.warning("尝试重建向量存储...")
                    self._rebuild_vector_store()
                    if self.vector_store:
                        self.vector_store.add_documents(valid_docs)
                        logger.info(f"重建后成功添加文档")
                        self._safe_save()
                        return True
                except Exception as rebuild_e:
                    logger.error(f"重建向量存储并添加文档失败: {str(rebuild_e)}")
                return False
    
    def _safe_save(self):
        """安全地保存向量存储"""
        if not self.vector_store:
            return False
            
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # 由于Chroma自动处理持久化，我们只需要确保操作完成
                logger.info(f"向量存储已自动保存到: {self.config.VECTOR_STORE_PATH}")
                return True
            except Exception as e:
                retry_count += 1
                logger.error(f"保存向量存储失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
                if retry_count < max_retries:
                    time.sleep(1)  # 等待一秒后重试
        return False
    
    def similarity_search(self, query, k=3, score_threshold=0.5):
        """在向量存储中进行相似度搜索，支持阈值过滤和重排序"""
        if not self.vector_store:
            if not self._ensure_vector_store_exists():
                logger.error("无法执行搜索，向量存储不存在")
                return []
                
        with self._lock:
            try:
                # 执行带分数的相似度搜索
                results_with_score = self.vector_store.similarity_search_with_score(query, k=min(k*2, 10))  # 获取更多结果用于重排序
                
                # 过滤低于阈值的结果
                filtered_results = [(doc, score) for doc, score in results_with_score if score >= score_threshold]
                logger.info(f"过滤后剩余 {len(filtered_results)} 个相关文档")
                
                if not filtered_results:
                    return []
                
                # 如果有重排序模型，进行重排序
                if self.reranker:
                    try:
                        # 准备重排序的输入对
                        pairs = [(query, doc.page_content) for doc, _ in filtered_results]
                        # 计算相关性分数
                        rerank_scores = self.reranker.predict(pairs)
                        
                        # 结合原始分数和重排序分数（可以调整权重）
                        combined_results = [(doc, 0.7 * original_score + 0.3 * rerank_score) 
                                          for (doc, original_score), rerank_score in zip(filtered_results, rerank_scores)]
                        
                        # 按分数排序
                        combined_results.sort(key=lambda x: x[1], reverse=True)
                        
                        # 只保留前k个结果
                        final_results = [doc for doc, _ in combined_results[:k]]
                        logger.info(f"重排序完成，返回前 {len(final_results)} 个结果")
                    except Exception as rerank_e:
                        logger.error(f"重排序失败: {str(rerank_e)}")
                        # 如果重排序失败，使用过滤后的结果
                        final_results = [doc for doc, _ in filtered_results[:k]]
                else:
                    # 没有重排序模型，按原始分数排序并返回
                    filtered_results.sort(key=lambda x: x[1], reverse=True)
                    final_results = [doc for doc, _ in filtered_results[:k]]
                
                logger.info(f"相似度搜索完成，返回 {len(final_results)} 个相关文档")
                return final_results
            except Exception as e:
                logger.error(f"相似度搜索失败: {str(e)}")
                # 尝试重建向量存储并再次搜索
                try:
                    logger.warning("尝试重建向量存储后再次搜索...")
                    self._rebuild_vector_store()
                    if self.vector_store:
                        results_with_score = self.vector_store.similarity_search_with_score(query, k=k)
                        final_results = [doc for doc, _ in results_with_score]
                        logger.info(f"重建后搜索完成，找到 {len(final_results)} 个结果")
                        return final_results
                except Exception as rebuild_e:
                    logger.error(f"重建后搜索也失败: {str(rebuild_e)}")
                return []
    
    def clear_vector_store(self):
        """清空向量存储"""
        with self._lock:
            try:
                # 删除向量存储目录
                if os.path.exists(self.config.VECTOR_STORE_PATH):
                    # 创建时间戳备份
                    backup_path = f"{self.config.VECTOR_STORE_PATH}_clear_{int(time.time())}"
                    if os.path.exists(backup_path):
                        shutil.rmtree(backup_path, ignore_errors=True)
                    shutil.move(self.config.VECTOR_STORE_PATH, backup_path)
                    logger.info(f"已备份并清空向量存储，备份路径: {backup_path}")
                # 重新初始化向量存储
                self._init_vector_store()
                logger.info("向量存储已清空")
                return True
            except Exception as e:
                logger.error(f"清空向量存储失败: {str(e)}")
                # 尝试直接删除文件
                try:
                    if os.path.exists(self.config.VECTOR_STORE_PATH):
                        for file in os.listdir(self.config.VECTOR_STORE_PATH):
                            file_path = os.path.join(self.config.VECTOR_STORE_PATH, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        logger.warning("已尝试直接删除向量存储文件")
                        self.vector_store = None
                        return True
                except Exception as inner_e:
                    logger.error(f"直接删除文件也失败: {str(inner_e)}")
                return False
    
    def _rebuild_vector_store(self):
        """重建向量存储"""
        logger.info("开始重建向量存储...")
        try:
            # 清理现有的向量存储
            if os.path.exists(self.config.VECTOR_STORE_PATH):
                backup_path = f"{self.config.VECTOR_STORE_PATH}_rebuild_{int(time.time())}"
                shutil.move(self.config.VECTOR_STORE_PATH, backup_path)
                logger.info(f"已备份旧的向量存储到: {backup_path}")
            # 重新初始化
            self._init_vector_store()
            logger.info("向量存储重建完成")
        except Exception as e:
            logger.error(f"重建向量存储失败: {str(e)}")
    
    def get_vector_count(self):
        """获取向量存储中的向量数量"""
        if not self.vector_store:
            return 0
            
        try:
            # 获取向量存储中的向量数量
            count = self.vector_store._collection.count()
            logger.info(f"向量存储中的向量数量: {count}")
            return count
        except Exception as e:
            logger.error(f"获取向量数量失败: {str(e)}")
            return 0

# 创建向量存储管理器实例
vector_store_manager = VectorStoreManager()

# 为了向后兼容，提供直接的vector_store访问
vector_store = vector_store_manager.vector_store if vector_store_manager.vector_store else None