import logging
import os
import threading
import time
import shutil
import tempfile
import numpy as np
import traceback
import hashlib
import json
from collections import defaultdict, OrderedDict
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.storage import LocalFileStore
from langchain.cache import SQLiteCache
from src.config import global_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 缓存配置
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '.cache')
SEARCH_CACHE_SIZE = 1000  # 搜索缓存大小
SEARCH_CACHE_TTL = 3600  # 搜索缓存有效期（秒）

class SearchCache:
    """搜索结果缓存类"""
    def __init__(self, max_size=SEARCH_CACHE_SIZE, ttl=SEARCH_CACHE_TTL):
        self.cache = OrderedDict()  # 使用OrderedDict实现LRU缓存
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.RLock()
        
    def _generate_key(self, query: str, params: Dict) -> str:
        """生成缓存键"""
        key_data = f"{query}:{json.dumps(sorted(params.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, params: Dict) -> Optional[List[Document]]:
        """获取缓存的搜索结果"""
        with self.lock:
            key = self._generate_key(query, params)
            if key in self.cache:
                timestamp, results = self.cache[key]
                # 检查是否过期
                if time.time() - timestamp < self.ttl:
                    # 将访问的项移到最后（LRU策略）
                    self.cache.move_to_end(key)
                    logger.debug(f"缓存命中: {query[:50]}...")
                    return results
                else:
                    # 删除过期项
                    del self.cache[key]
                    logger.debug(f"缓存过期: {query[:50]}...")
            return None
    
    def set(self, query: str, params: Dict, results: List[Document]) -> None:
        """设置缓存的搜索结果"""
        with self.lock:
            key = self._generate_key(query, params)
            
            # 如果缓存已满，删除最旧的项
            if len(self.cache) >= self.max_size and key not in self.cache:
                self.cache.popitem(last=False)
            
            # 存储结果和时间戳
            self.cache[key] = (time.time(), results)
            logger.debug(f"缓存设置: {query[:50]}...")
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            logger.info("搜索缓存已清空")


class VectorStoreManager:
    """增强的向量存储管理器，支持多索引、混合检索和高级功能"""
    
    # 使用锁保护向量存储的并发访问
    _lock = threading.RLock()
    
    def __init__(self, config=None):
        """初始化向量存储管理器"""
        self.config = config or global_config
        self.embeddings = None
        self.vector_stores = {}  # 多索引支持 {index_name: vector_store}
        self.current_index = "default"  # 当前活动索引
        self.reranker = None
        self.last_access_time = time.time()
        self.search_cache = SearchCache()
        self.metadata_indices = defaultdict(set)  # 元数据索引 {metadata_key: set(metadata_values)}
        self.stats = {
            "queries": 0,
            "cache_hits": 0,
            "documents_added": 0,
            "search_time": 0,
            "last_query_time": None
        }
        
        # 初始化嵌入模型
        self._init_embeddings()
        
        # 初始化默认向量存储
        self._init_vector_store(index_name=self.current_index)
        
        # 初始化重排序模型
        self._init_reranker()
        
        # 确保缓存目录存在
        os.makedirs(CACHE_DIR, exist_ok=True)
    
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
                    
                    def embed_with_metadata(self, texts, metadata_list=None):
                        """带元数据的嵌入生成"""
                        # 在实际应用中，可以根据元数据调整嵌入生成
                        return self.embed_documents(texts)
                
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
    
    def _update_metadata_index(self, documents: List[Document]) -> None:
        """更新元数据索引"""
        for doc in documents:
            if hasattr(doc, 'metadata'):
                for key, value in doc.metadata.items():
                    if value is not None:
                        # 处理列表类型的元数据值
                        if isinstance(value, list):
                            for v in value:
                                self.metadata_indices[key].add(str(v))
                        else:
                            self.metadata_indices[key].add(str(value))
    
    def _init_vector_store(self, index_name="default"):
        """初始化向量存储"""
        with self._lock:
            try:
                # 为每个索引创建单独的存储路径
                index_path = os.path.join(self.config.VECTOR_STORE_PATH, index_name)
                os.makedirs(index_path, exist_ok=True)
                
                # 确保嵌入函数已初始化
                if self.embeddings is None:
                    logger.warning("嵌入函数未初始化，无法创建向量存储")
                    return
                
                # 初始化Chroma向量存储
                vector_store = Chroma(
                    persist_directory=index_path,
                    embedding_function=self.embeddings,
                    collection_name=index_name
                )
                self.vector_stores[index_name] = vector_store
                logger.info(f"成功初始化向量存储索引 '{index_name}'，路径: {index_path}")
                
                # 验证向量存储的完整性
                try:
                    # 执行一个简单的搜索来验证向量存储是否正常
                    if vector_store:  # 确保vector_store不为None
                        vector_store.similarity_search("测试", k=1)
                        logger.info(f"向量存储索引 '{index_name}' 验证成功")
                except Exception as verify_e:
                    logger.error(f"向量存储索引 '{index_name}' 验证失败，可能已损坏: {str(verify_e)}")
                    # 重新创建向量存储
                    logger.warning(f"重新创建向量存储索引 '{index_name}'...")
                    del self.vector_stores[index_name]
                    # 如果目录存在且有问题，先备份和清理
                    if os.path.exists(index_path):
                        backup_path = f"{index_path}_backup_{int(time.time())}"
                        shutil.move(index_path, backup_path)
                        logger.info(f"已备份损坏的向量存储到: {backup_path}")
                    os.makedirs(index_path, exist_ok=True)
                    self.vector_stores[index_name] = Chroma(
                        persist_directory=index_path,
                        embedding_function=self.embeddings,
                        collection_name=index_name
                    )
            except Exception as e:
                logger.error(f"初始化向量存储索引 '{index_name}' 失败: {str(e)}")
    
    def _ensure_vector_store_exists(self, index_name=None):
        """确保向量存储存在"""
        index_name = index_name or self.current_index
        with self._lock:
            self.last_access_time = time.time()
            if index_name not in self.vector_stores or self.vector_stores[index_name] is None:
                # 创建空的向量存储
                try:
                    # 重新初始化
                    self._init_embeddings()
                    self._init_vector_store(index_name)
                    logger.info(f"成功重新初始化向量存储索引 '{index_name}'")
                except Exception as e:
                    logger.error(f"重新初始化向量存储索引 '{index_name}' 失败: {str(e)}")
                    if index_name in self.vector_stores:
                        del self.vector_stores[index_name]
        return index_name in self.vector_stores and self.vector_stores[index_name] is not None
    
    def switch_index(self, index_name="default"):
        """切换向量存储索引"""
        with self._lock:
            if index_name not in self.vector_stores:
                # 如果索引不存在，创建它
                self._init_vector_store(index_name)
            self.current_index = index_name
            logger.info(f"已切换到向量存储索引: {index_name}")
            return True
    
    def list_indices(self):
        """列出所有可用的向量存储索引"""
        return list(self.vector_stores.keys())
    
    def create_index(self, index_name):
        """创建新的向量存储索引"""
        if index_name in self.vector_stores:
            logger.warning(f"索引 '{index_name}' 已存在")
            return False
        return self._init_vector_store(index_name) is not None
    
    def delete_index(self, index_name):
        """删除向量存储索引"""
        with self._lock:
            if index_name == "default" and len(self.vector_stores) == 1:
                logger.warning("不能删除唯一的默认索引")
                return False
            if index_name in self.vector_stores:
                # 删除索引目录
                index_path = os.path.join(self.config.VECTOR_STORE_PATH, index_name)
                if os.path.exists(index_path):
                    backup_path = f"{index_path}_deleted_{int(time.time())}"
                    shutil.move(index_path, backup_path)
                    logger.info(f"已备份并删除索引 '{index_name}' 到: {backup_path}")
                del self.vector_stores[index_name]
                # 如果删除的是当前索引，切换到默认索引
                if index_name == self.current_index and "default" in self.vector_stores:
                    self.current_index = "default"
                    logger.info("已切换到默认索引")
                return True
            logger.warning(f"索引 '{index_name}' 不存在")
            return False
    
    def add_documents(self, documents: List[Document], index_name: Optional[str] = None, batch_size: int = 100):
        """向向量存储添加文档（支持批量处理）"""
        if not documents or not self.embeddings:
            return False
            
        index_name = index_name or self.current_index
        
        with self._lock:
            try:
                if not self._ensure_vector_store_exists(index_name):
                    logger.error(f"无法添加文档，向量存储索引 '{index_name}' 不存在")
                    return False
                    
                # 过滤空文档
                valid_docs = [doc for doc in documents if hasattr(doc, 'page_content') and doc.page_content.strip()]
                if not valid_docs:
                    logger.warning("没有有效的文档可以添加到向量存储")
                    return True  # 没有需要添加的文档，视为成功
                    
                vector_store = self.vector_stores[index_name]
                total_docs = len(valid_docs)
                added_docs = 0
                
                # 批量添加文档以提高性能
                for i in range(0, total_docs, batch_size):
                    batch = valid_docs[i:i + batch_size]
                    vector_store.add_documents(batch)
                    added_docs += len(batch)
                    logger.info(f"已添加批次 {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}，累计 {added_docs}/{total_docs} 个文档")
                    
                    # 更新元数据索引
                    self._update_metadata_index(batch)
                
                logger.info(f"成功添加 {added_docs} 个文档到向量存储索引 '{index_name}'")
                
                # 更新统计信息
                self.stats["documents_added"] += added_docs
                
                # 保存向量存储
                return self._safe_save(index_name)
            except Exception as e:
                logger.error(f"添加文档失败: {str(e)}")
                traceback.print_exc()
                # 尝试重建向量存储并再次添加
                try:
                    logger.warning(f"尝试重建向量存储索引 '{index_name}'...")
                    self._rebuild_vector_store(index_name)
                    if index_name in self.vector_stores:
                        vector_store = self.vector_stores[index_name]
                        vector_store.add_documents(valid_docs)
                        logger.info(f"重建后成功添加文档")
                        self._update_metadata_index(valid_docs)
                        self._safe_save(index_name)
                        return True
                except Exception as rebuild_e:
                    logger.error(f"重建向量存储并添加文档失败: {str(rebuild_e)}")
                return False
    
    def add_document_with_embedding(self, document: Document, embedding: Optional[List[float]] = None, index_name: Optional[str] = None):
        """添加带预计算嵌入的文档"""
        index_name = index_name or self.current_index
        
        with self._lock:
            try:
                if not self._ensure_vector_store_exists(index_name):
                    logger.error(f"无法添加文档，向量存储索引 '{index_name}' 不存在")
                    return False
                    
                if not hasattr(document, 'page_content') or not document.page_content.strip():
                    logger.warning("跳过空文档")
                    return True
                    
                vector_store = self.vector_stores[index_name]
                
                # 如果没有提供嵌入，使用嵌入模型计算
                if embedding is None:
                    embedding = self.embeddings.embed_query(document.page_content)
                
                # 添加单个文档
                ids = vector_store.add_documents([document], embedding=[embedding])
                
                # 更新元数据索引
                self._update_metadata_index([document])
                
                logger.info(f"成功添加单个文档到向量存储索引 '{index_name}'，ID: {ids[0] if ids else '未知'}")
                
                # 更新统计信息
                self.stats["documents_added"] += 1
                
                # 保存向量存储
                return self._safe_save(index_name)
            except Exception as e:
                logger.error(f"添加带嵌入的文档失败: {str(e)}")
                return False
    
    def batch_process_documents(self, documents: List[Document], index_name: Optional[str] = None, batch_size: int = 100, num_workers: int = 4):
        """使用多线程批量处理文档"""
        # 这里可以实现多线程处理逻辑
        # 为简化，当前使用单线程批量处理
        return self.add_documents(documents, index_name, batch_size)
    
    def _safe_save(self, index_name: Optional[str] = None):
        """安全地保存向量存储"""
        index_name = index_name or self.current_index
        if index_name not in self.vector_stores or not self.vector_stores[index_name]:
            return False
            
        vector_store = self.vector_stores[index_name]
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # 对于Chroma，我们可以尝试显式持久化
                if hasattr(vector_store, 'persist'):
                    vector_store.persist()
                logger.info(f"向量存储索引 '{index_name}' 已保存")
                return True
            except Exception as e:
                retry_count += 1
                logger.error(f"保存向量存储索引 '{index_name}' 失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
                if retry_count < max_retries:
                    time.sleep(1)  # 等待一秒后重试
        return False
    
    def optimize_index(self, index_name: Optional[str] = None):
        """优化向量存储索引（压缩、合并等）"""
        index_name = index_name or self.current_index
        with self._lock:
            if not self._ensure_vector_store_exists(index_name):
                logger.error(f"无法优化索引，向量存储索引 '{index_name}' 不存在")
                return False
            
            try:
                logger.info(f"开始优化向量存储索引 '{index_name}'...")
                # 对于Chroma，我们可以通过重建索引来优化
                # 首先获取所有文档
                vector_store = self.vector_stores[index_name]
                all_docs = vector_store.get()
                
                # 重建索引
                self._rebuild_vector_store(index_name)
                
                # 重新添加文档
                if 'documents' in all_docs and all_docs['documents']:
                    # 重新添加文档需要保持元数据和嵌入的对应关系
                    # 这里简化处理，只重新添加文档内容
                    docs = []
                    for i, content in enumerate(all_docs['documents']):
                        metadata = all_docs['metadatas'][i] if 'metadatas' in all_docs and i < len(all_docs['metadatas']) else {}
                        docs.append(Document(page_content=content, metadata=metadata))
                    
                    self.add_documents(docs, index_name)
                    logger.info(f"索引 '{index_name}' 优化完成，重新添加了 {len(docs)} 个文档")
                else:
                    logger.info(f"索引 '{index_name}' 优化完成，索引为空")
                
                return True
            except Exception as e:
                logger.error(f"优化向量存储索引 '{index_name}' 失败: {str(e)}")
                return False
    
    def get_metadata_keys(self):
        """获取所有可用的元数据键"""
        return list(self.metadata_indices.keys())
    
    def get_metadata_values(self, key: str):
        """获取指定元数据键的所有可能值"""
        return list(self.metadata_indices.get(key, set()))
    
    def similarity_search(self, query: str, k: int = 3, score_threshold: float = 0.5, 
                         metadata_filter: Optional[Dict[str, Any]] = None, index_name: Optional[str] = None,
                         use_cache: bool = True, hybrid_search: bool = False):
        """增强的相似度搜索，支持元数据过滤、缓存和混合检索"""
        start_time = time.time()
        index_name = index_name or self.current_index
        
        # 准备缓存参数
        cache_params = {
            "k": k,
            "score_threshold": score_threshold,
            "metadata_filter": metadata_filter,
            "index_name": index_name,
            "hybrid_search": hybrid_search
        }
        
        # 尝试从缓存获取结果
        if use_cache:
            cached_results = self.search_cache.get(query, cache_params)
            if cached_results is not None:
                self.stats["cache_hits"] += 1
                self.stats["last_query_time"] = time.time()
                return cached_results
        
        with self._lock:
            try:
                if not self._ensure_vector_store_exists(index_name):
                    logger.error(f"无法执行搜索，向量存储索引 '{index_name}' 不存在")
                    return []
                    
                vector_store = self.vector_stores[index_name]
                search_kwargs = {}
                
                # 添加元数据过滤
                if metadata_filter:
                    search_kwargs["filter"] = metadata_filter
                
                # 基础向量搜索
                if not hybrid_search:
                    # 执行带分数的相似度搜索
                    results_with_score = vector_store.similarity_search_with_score(
                        query, 
                        k=min(k*2, 10),  # 获取更多结果用于重排序
                        **search_kwargs
                    )
                else:
                    # 混合检索（向量+关键词）
                    results_with_score = self._hybrid_search(query, k=min(k*2, 10), 
                                                            metadata_filter=metadata_filter, 
                                                            index_name=index_name)
                
                # 过滤低于阈值的结果
                filtered_results = [(doc, score) for doc, score in results_with_score if score >= score_threshold]
                logger.info(f"过滤后剩余 {len(filtered_results)} 个相关文档")
                
                if not filtered_results:
                    return []
                
                # 移除重复文档（基于内容相似度）
                filtered_results = self._deduplicate_documents(filtered_results)
                
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
                        filtered_results.sort(key=lambda x: x[1], reverse=True)
                        final_results = [doc for doc, _ in filtered_results[:k]]
                else:
                    # 没有重排序模型，按原始分数排序并返回
                    filtered_results.sort(key=lambda x: x[1], reverse=True)
                    final_results = [doc for doc, _ in filtered_results[:k]]
                
                # 更新搜索缓存
                if use_cache and final_results:
                    self.search_cache.set(query, cache_params, final_results)
                
                # 更新统计信息
                search_time = time.time() - start_time
                self.stats["queries"] += 1
                self.stats["search_time"] += search_time
                self.stats["last_query_time"] = time.time()
                
                logger.info(f"相似度搜索完成，返回 {len(final_results)} 个相关文档，耗时: {search_time:.3f}s")
                return final_results
            except Exception as e:
                logger.error(f"相似度搜索失败: {str(e)}")
                # 尝试重建向量存储并再次搜索
                try:
                    logger.warning(f"尝试重建向量存储索引 '{index_name}' 后再次搜索...")
                    self._rebuild_vector_store(index_name)
                    if index_name in self.vector_stores:
                        vector_store = self.vector_stores[index_name]
                        search_kwargs = {}
                        if metadata_filter:
                            search_kwargs["filter"] = metadata_filter
                        results_with_score = vector_store.similarity_search_with_score(query, k=k, **search_kwargs)
                        final_results = [doc for doc, _ in results_with_score]
                        logger.info(f"重建后搜索完成，找到 {len(final_results)} 个结果")
                        return final_results
                except Exception as rebuild_e:
                    logger.error(f"重建后搜索也失败: {str(rebuild_e)}")
                return []
    
    def _hybrid_search(self, query: str, k: int = 5, metadata_filter: Optional[Dict] = None, 
                      index_name: Optional[str] = None) -> List[Tuple[Document, float]]:
        """混合检索实现（向量+关键词）"""
        index_name = index_name or self.current_index
        vector_store = self.vector_stores.get(index_name)
        if not vector_store:
            return []
        
        try:
            # 获取向量存储中的所有文档
            all_docs_result = vector_store.get()
            if not all_docs_result or 'documents' not in all_docs_result or not all_docs_result['documents']:
                return []
            
            # 构建文档列表
            docs = []
            for i, content in enumerate(all_docs_result['documents']):
                metadata = all_docs_result['metadatas'][i] if 'metadatas' in all_docs_result and i < len(all_docs_result['metadatas']) else {}
                docs.append(Document(page_content=content, metadata=metadata))
            
            # 元数据过滤
            if metadata_filter:
                filtered_docs = []
                for doc in docs:
                    match = True
                    for key, value in metadata_filter.items():
                        if key not in doc.metadata or doc.metadata[key] != value:
                            match = False
                            break
                    if match:
                        filtered_docs.append(doc)
                docs = filtered_docs
            
            if not docs:
                return []
            
            # 创建BM25检索器
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = k * 2  # 获取更多结果
            
            # BM25检索
            bm25_results = bm25_retriever.get_relevant_documents(query)
            
            # 向量检索
            vector_results_with_score = vector_store.similarity_search_with_score(query, k=k * 2, filter=metadata_filter)
            
            # 合并结果（简单加权）
            result_dict = {}
            
            # 处理BM25结果（给每个结果分配分数）
            for i, doc in enumerate(bm25_results):
                doc_id = hash(doc.page_content)  # 简单的文档标识
                # BM25分数随排名递减
                bm25_score = 1.0 / (i + 1)
                result_dict[doc_id] = (doc, bm25_score * 0.3)  # BM25权重为0.3
            
            # 处理向量结果
            for doc, vector_score in vector_results_with_score:
                doc_id = hash(doc.page_content)
                if doc_id in result_dict:
                    # 合并分数
                    existing_doc, existing_score = result_dict[doc_id]
                    result_dict[doc_id] = (existing_doc, existing_score + vector_score * 0.7)  # 向量权重为0.7
                else:
                    result_dict[doc_id] = (doc, vector_score * 0.7)
            
            # 转换为列表并排序
            combined_results = list(result_dict.values())
            combined_results.sort(key=lambda x: x[1], reverse=True)
            
            return combined_results[:k]
        
        except Exception as e:
            logger.error(f"混合检索失败: {str(e)}")
            # 失败时回退到向量检索
            return vector_store.similarity_search_with_score(query, k=k, filter=metadata_filter)
    
    def _deduplicate_documents(self, results_with_score: List[Tuple[Document, float]], 
                             similarity_threshold: float = 0.9) -> List[Tuple[Document, float]]:
        """移除相似度过高的重复文档"""
        if not results_with_score:
            return []
        
        # 简单实现：基于文档内容的哈希去重
        seen_contents = set()
        deduplicated = []
        
        for doc, score in results_with_score:
            # 计算文档内容的哈希（简化版本）
            # 在实际应用中，可以使用更复杂的相似度计算
            content_hash = hashlib.md5(doc.page_content[:1000].encode()).hexdigest()
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                deduplicated.append((doc, score))
        
        if len(deduplicated) < len(results_with_score):
            logger.info(f"已移除 {len(results_with_score) - len(deduplicated)} 个重复文档")
        
        return deduplicated
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], k: int = 10, 
                          index_name: Optional[str] = None) -> List[Document]:
        """仅根据元数据进行过滤搜索"""
        index_name = index_name or self.current_index
        
        with self._lock:
            if not self._ensure_vector_store_exists(index_name):
                logger.error(f"无法执行搜索，向量存储索引 '{index_name}' 不存在")
                return []
                
            vector_store = self.vector_stores[index_name]
            
            try:
                # 使用元数据过滤进行搜索
                results = vector_store.similarity_search(
                    "",  # 空查询
                    k=k,
                    filter=metadata_filter
                )
                
                logger.info(f"元数据搜索完成，返回 {len(results)} 个文档")
                return results
            except Exception as e:
                logger.error(f"元数据搜索失败: {str(e)}")
                return []
    
    def clear_vector_store(self, index_name: Optional[str] = None, keep_backup: bool = True):
        """清空向量存储"""
        index_name = index_name or self.current_index
        
        with self._lock:
            try:
                # 索引路径
                index_path = os.path.join(self.config.VECTOR_STORE_PATH, index_name)
                
                # 删除向量存储目录
                if os.path.exists(index_path):
                    if keep_backup:
                        # 创建时间戳备份
                        backup_path = f"{index_path}_clear_{int(time.time())}"
                        if os.path.exists(backup_path):
                            shutil.rmtree(backup_path, ignore_errors=True)
                        shutil.move(index_path, backup_path)
                        logger.info(f"已备份并清空向量存储索引 '{index_name}'，备份路径: {backup_path}")
                    else:
                        # 直接删除
                        shutil.rmtree(index_path, ignore_errors=True)
                        logger.info(f"已直接清空向量存储索引 '{index_name}'")
                
                # 重新初始化向量存储
                self._init_vector_store(index_name)
                
                # 清空相关的元数据索引
                # 注意：这会清空所有索引的元数据，在多索引场景下可能需要改进
                self.metadata_indices.clear()
                
                # 清空搜索缓存
                self.search_cache.clear()
                
                logger.info(f"向量存储索引 '{index_name}' 已清空")
                return True
            except Exception as e:
                logger.error(f"清空向量存储索引 '{index_name}' 失败: {str(e)}")
                # 尝试直接删除文件
                try:
                    index_path = os.path.join(self.config.VECTOR_STORE_PATH, index_name)
                    if os.path.exists(index_path):
                        for file in os.listdir(index_path):
                            file_path = os.path.join(index_path, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        logger.warning(f"已尝试直接删除向量存储索引 '{index_name}' 的文件")
                        if index_name in self.vector_stores:
                            del self.vector_stores[index_name]
                        return True
                except Exception as inner_e:
                    logger.error(f"直接删除文件也失败: {str(inner_e)}")
                return False
    
    def export_index(self, index_name: Optional[str] = None, export_path: Optional[str] = None) -> str:
        """导出向量存储索引"""
        index_name = index_name or self.current_index
        
        with self._lock:
            if not self._ensure_vector_store_exists(index_name):
                logger.error(f"无法导出索引，向量存储索引 '{index_name}' 不存在")
                return ""
            
            try:
                # 索引路径
                index_path = os.path.join(self.config.VECTOR_STORE_PATH, index_name)
                if not os.path.exists(index_path):
                    logger.error(f"索引路径不存在: {index_path}")
                    return ""
                
                # 导出路径
                if not export_path:
                    export_path = os.path.join(
                        CACHE_DIR, 
                        f"index_export_{index_name}_{int(time.time())}"
                    )
                
                # 创建导出目录
                os.makedirs(export_path, exist_ok=True)
                
                # 复制索引文件
                for item in os.listdir(index_path):
                    src = os.path.join(index_path, item)
                    dst = os.path.join(export_path, item)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    elif os.path.isdir(src):
                        shutil.copytree(src, dst)
                
                logger.info(f"成功导出向量存储索引 '{index_name}' 到: {export_path}")
                return export_path
            except Exception as e:
                logger.error(f"导出向量存储索引 '{index_name}' 失败: {str(e)}")
                return ""
    
    def import_index(self, index_name: str, import_path: str) -> bool:
        """导入向量存储索引"""
        with self._lock:
            try:
                if not os.path.exists(import_path):
                    logger.error(f"导入路径不存在: {import_path}")
                    return False
                
                # 目标索引路径
                target_path = os.path.join(self.config.VECTOR_STORE_PATH, index_name)
                
                # 如果索引已存在，备份它
                if os.path.exists(target_path):
                    backup_path = f"{target_path}_import_backup_{int(time.time())}"
                    shutil.move(target_path, backup_path)
                    logger.info(f"已备份现有索引到: {backup_path}")
                
                # 创建目标目录
                os.makedirs(target_path, exist_ok=True)
                
                # 复制导入文件
                for item in os.listdir(import_path):
                    src = os.path.join(import_path, item)
                    dst = os.path.join(target_path, item)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    elif os.path.isdir(src):
                        shutil.copytree(src, dst)
                
                # 重新初始化索引
                self._init_vector_store(index_name)
                
                logger.info(f"成功从 {import_path} 导入向量存储索引 '{index_name}'")
                return True
            except Exception as e:
                logger.error(f"导入向量存储索引 '{index_name}' 失败: {str(e)}")
                return False
    
    def _rebuild_vector_store(self, index_name: Optional[str] = None):
        """重建向量存储"""
        index_name = index_name or self.current_index
        logger.info(f"开始重建向量存储索引 '{index_name}'...")
        try:
            # 索引路径
            index_path = os.path.join(self.config.VECTOR_STORE_PATH, index_name)
            
            # 清理现有的向量存储
            if os.path.exists(index_path):
                backup_path = f"{index_path}_rebuild_{int(time.time())}"
                shutil.move(index_path, backup_path)
                logger.info(f"已备份旧的向量存储索引到: {backup_path}")
            
            # 重新初始化
            self._init_vector_store(index_name)
            logger.info(f"向量存储索引 '{index_name}' 重建完成")
        except Exception as e:
            logger.error(f"重建向量存储索引 '{index_name}' 失败: {str(e)}")
    
    def get_vector_count(self, index_name: Optional[str] = None) -> int:
        """获取向量存储中的向量数量"""
        index_name = index_name or self.current_index
        
        if index_name not in self.vector_stores or not self.vector_stores[index_name]:
            return 0
            
        vector_store = self.vector_stores[index_name]
        
        try:
            # 获取向量存储中的向量数量
            count = vector_store._collection.count()
            logger.info(f"向量存储索引 '{index_name}' 中的向量数量: {count}")
            return count
        except Exception as e:
            logger.error(f"获取向量数量失败: {str(e)}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取向量存储统计信息"""
        stats = self.stats.copy()
        # 添加索引相关统计
        index_stats = {}
        for index_name in self.vector_stores:
            index_stats[index_name] = {
                "vector_count": self.get_vector_count(index_name)
            }
        stats["indices"] = index_stats
        stats["cache_size"] = len(self.search_cache.cache)
        stats["metadata_keys"] = len(self.metadata_indices)
        
        return stats
    
    def refresh(self, index_name: Optional[str] = None):
        """刷新向量存储连接"""
        index_name = index_name or self.current_index
        with self._lock:
            if index_name in self.vector_stores:
                del self.vector_stores[index_name]
            self._init_vector_store(index_name)
            logger.info(f"已刷新向量存储索引 '{index_name}'")
    
    def clear_cache(self):
        """清空搜索缓存"""
        self.search_cache.clear()
        return True

# 创建向量存储管理器实例
vector_store_manager = VectorStoreManager()

# 为了向后兼容，提供直接的vector_store访问
def get_current_vector_store():
    """获取当前活动的向量存储"""
    if vector_store_manager.current_index in vector_store_manager.vector_stores:
        return vector_store_manager.vector_stores[vector_store_manager.current_index]
    return None

vector_store = get_current_vector_store()