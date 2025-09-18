import logging
import os
import shutil
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.config import global_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """向量存储管理器，用于管理向量存储、向量化文档以及进行相似度搜索"""
    
    def __init__(self, config=None):
        """初始化向量存储管理器"""
        self.config = config or global_config
        self.embeddings = None
        self.vector_store = None
        
        # 初始化嵌入模型
        self._init_embeddings()
        
        # 初始化向量存储
        self._init_vector_store()
        
        # 用于重排序的模型
        self.reranker = None
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
        except Exception as e:
            logger.error(f"初始化向量存储失败: {str(e)}")
            self.vector_store = None
    
    def add_documents(self, documents):
        """向向量存储添加文档"""
        # 如果向量存储未初始化，尝试重新初始化
        if not self.vector_store:
            logger.warning("向量存储未初始化，尝试重新初始化")
            self._init_embeddings()
            self._init_vector_store()
            
            # 如果仍然未初始化成功
            if not self.vector_store:
                logger.error("向量存储初始化失败，无法添加文档")
                return False
        
        if not documents:
            logger.warning("没有文档可添加")
            return False
        
        try:
            # 添加文档到向量存储
            self.vector_store.add_documents(documents)
            
            logger.info(f"成功添加 {len(documents)} 个文档到向量存储")
            return True
        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {str(e)}")
            return False
    
    def similarity_search(self, query, k=3, score_threshold=0.5):
        """在向量存储中进行相似度搜索，支持阈值过滤和重排序"""
        if not self.vector_store:
            logger.warning("向量存储未初始化")
            return []
        
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
            return []
    
    def clear_vector_store(self):
        """清空向量存储"""
        if not self.vector_store:
            logger.warning("向量存储未初始化")
            return False
        
        try:
            # 删除向量存储目录
            if os.path.exists(self.config.VECTOR_STORE_PATH):
                import shutil
                shutil.rmtree(self.config.VECTOR_STORE_PATH)
                
                # 重新初始化向量存储
                self._init_vector_store()
                
            logger.info("成功清空向量存储")
            return True
        except Exception as e:
            logger.error(f"清空向量存储失败: {str(e)}")
            return False
    
    def get_vector_count(self):
        """获取向量存储中的向量数量"""
        if not self.vector_store:
            logger.warning("向量存储未初始化")
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