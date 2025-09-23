import os
import yaml
import re
import logging
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentClassifier:
    """文档分类器，用于对文档进行自动分类
    支持基于规则的分类和机器学习辅助分类
    """
    
    def __init__(self, config_path: str = "config/document_types.yaml"):
        """初始化文档分类器
        
        Args:
            config_path: 文档类型配置文件路径
        """
        # 存储文档类型配置
        self.document_types = []
        # 存储分类配置
        self.classification_config = {}
        # 加载配置
        self._load_config(config_path)
        # 初始化缓存机制
        self._init_cache()
        # 初始化机器学习模型（如果启用）
        self._init_ml_model()
    
    def _load_config(self, config_path: str) -> None:
        """加载文档类型配置文件
        
        Args:
            config_path: 配置文件路径
        """
        try:
            # 检查配置文件是否存在
            if not os.path.exists(config_path):
                logger.error(f"配置文件不存在: {config_path}")
                # 使用默认配置
                self._load_default_config()
                return
            
            # 加载配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
                # 加载文档类型配置
                if 'document_types' in config:
                    self.document_types = config['document_types']
                    logger.info(f"成功加载 {len(self.document_types)} 种文档类型配置")
                else:
                    logger.warning("配置文件中未找到document_types字段")
                    self._load_default_config()
                
                # 加载分类配置
                if 'classification_config' in config:
                    self.classification_config = config['classification_config']
                else:
                    logger.warning("配置文件中未找到classification_config字段")
                    self._set_default_classification_config()
                    
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            # 加载默认配置
            self._load_default_config()
    
    def _load_default_config(self) -> None:
        """加载默认的文档类型配置"""
        logger.info("使用默认文档类型配置")
        self.document_types = [
            {
                "id": "general",
                "name": "普通文档",
                "description": "无法明确分类的通用文档",
                "file_extensions": [".txt", ".md", ".docx", ".pdf"],
                "keywords": [],
                "metadata_keys": [],
                "min_keywords_match": 0,
                "confidence_threshold": 0.5
            }
        ]
        self._set_default_classification_config()
    
    def _set_default_classification_config(self) -> None:
        """设置默认的分类配置"""
        self.classification_config = {
            "text_sampling": {
                "sample_size": 5000,
                "sample_sections": 3,
                "min_section_length": 200
            },
            "keyword_matching": {
                "case_sensitive": False,
                "exact_match_only": False,
                "allow_stemming": True
            },
            "metadata_extraction": {
                "enabled": True,
                "max_metadata_count": 10
            },
            "ml_assistance": {
                "enabled": False,
                "model_path": "models/offline/document_classifier",
                "fallback_to_rules": True
            }
        }
    
    def _init_cache(self) -> None:
        """初始化分类缓存机制"""
        # 存储文档路径到分类结果的映射
        self._cache = {}
        # 存储文档哈希到分类结果的映射
        self._hash_cache = {}
        # 缓存大小限制
        self._max_cache_size = 1000
        # 缓存过期时间（秒）
        self._cache_ttl = 3600  # 1小时
        logger.info("文档分类缓存机制初始化完成")
    
    def _init_ml_model(self) -> None:
        """初始化机器学习辅助分类模型"""
        self.ml_model = None
        
        # 检查是否启用了机器学习辅助
        if (self.classification_config.get('ml_assistance', {}).get('enabled', False) and \
            os.path.exists(self.classification_config['ml_assistance']['model_path'])):
            try:
                # 这里可以根据实际情况加载具体的机器学习模型
                # 例如使用scikit-learn、TensorFlow或PyTorch等
                logger.info(f"尝试加载机器学习分类模型: {self.classification_config['ml_assistance']['model_path']}")
                
                # 由于没有实际的模型文件，这里仅作为示例
                # 在实际应用中，这里应该加载训练好的模型
                # self.ml_model = load_model(self.classification_config['ml_assistance']['model_path'])
                
                logger.info("机器学习分类模型加载成功")
            except Exception as e:
                logger.error(f"加载机器学习分类模型失败: {str(e)}")
                if self.classification_config['ml_assistance'].get('fallback_to_rules', True):
                    logger.info("回退到基于规则的分类方法")
                else:
                    logger.warning("未启用规则分类回退，可能导致分类功能不可用")
        else:
            logger.info("机器学习辅助分类已禁用或模型文件不存在")
    
    def _generate_document_hash(self, file_path: str, content_sample: str) -> str:
        """生成文档的哈希值，用于缓存
        
        Args:
            file_path: 文档路径
            content_sample: 文档内容样本
        
        Returns:
            文档的哈希值
        """
        # 获取文件修改时间
        if os.path.exists(file_path):
            mod_time = str(os.path.getmtime(file_path))
        else:
            mod_time = "0"
        
        # 组合文件路径、修改时间和内容样本生成哈希
        hash_input = f"{file_path}:{mod_time}:{content_sample[:1000]}"
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()
    
    def _get_from_cache(self, file_path: str, content_sample: str) -> Optional[Dict[str, Any]]:
        """从缓存中获取分类结果
        
        Args:
            file_path: 文档路径
            content_sample: 文档内容样本
        
        Returns:
            分类结果，如果缓存中不存在则返回None
        """
        # 生成文档哈希
        doc_hash = self._generate_document_hash(file_path, content_sample)
        
        # 检查哈希缓存
        if doc_hash in self._hash_cache:
            cache_entry = self._hash_cache[doc_hash]
            # 检查缓存是否过期
            if time.time() < cache_entry['expiry_time']:
                logger.debug(f"从缓存中获取文档 {file_path} 的分类结果")
                return cache_entry['result']
            else:
                # 缓存已过期，移除
                del self._hash_cache[doc_hash]
                if file_path in self._cache:
                    del self._cache[file_path]
        
        return None
    
    def _store_in_cache(self, file_path: str, content_sample: str, result: Dict[str, Any]) -> None:
        """将分类结果存储到缓存中
        
        Args:
            file_path: 文档路径
            content_sample: 文档内容样本
            result: 分类结果
        """
        # 检查缓存大小
        if len(self._hash_cache) >= self._max_cache_size:
            # 移除最早的缓存项
            oldest_key = next(iter(self._hash_cache))
            oldest_path = self._hash_cache[oldest_key]['file_path']
            del self._hash_cache[oldest_key]
            if oldest_path in self._cache:
                del self._cache[oldest_path]
        
        # 生成文档哈希
        doc_hash = self._generate_document_hash(file_path, content_sample)
        
        # 计算过期时间
        expiry_time = time.time() + self._cache_ttl
        
        # 存储到缓存
        cache_entry = {
            'result': result,
            'expiry_time': expiry_time,
            'file_path': file_path
        }
        
        self._hash_cache[doc_hash] = cache_entry
        self._cache[file_path] = doc_hash
        
        logger.debug(f"将文档 {file_path} 的分类结果存储到缓存")
    
    def _sample_document_content(self, document_content: str) -> str:
        """采样文档内容，用于分类
        
        Args:
            document_content: 完整的文档内容
        
        Returns:
            采样后的文档内容
        """
        # 获取采样配置
        sample_config = self.classification_config.get('text_sampling', {})
        sample_size = sample_config.get('sample_size', 5000)
        sample_sections = sample_config.get('sample_sections', 3)
        min_section_length = sample_config.get('min_section_length', 200)
        
        # 如果文档内容较短，直接返回全部内容
        if len(document_content) <= sample_size:
            return document_content
        
        # 否则，进行分段采样
        sampled_content = []
        section_size = len(document_content) // (sample_sections + 1)
        
        for i in range(1, sample_sections + 1):
            start_pos = i * section_size
            # 寻找合适的起始位置（句子边界）
            while start_pos > 0 and document_content[start_pos] not in ['.', '。', '!', '！', '?', '？', '\n', '\r']:
                start_pos -= 1
            
            # 截取样本
            sample = document_content[start_pos:start_pos + sample_size // sample_sections]
            
            # 如果样本太短，扩展到最小长度
            if len(sample) < min_section_length:
                end_pos = min(start_pos + min_section_length, len(document_content))
                sample = document_content[start_pos:end_pos]
            
            sampled_content.append(sample)
        
        # 组合样本
        return '\n\n'.join(sampled_content)
    
    def _match_keywords(self, content: str, keywords: List[str]) -> int:
        """在文档内容中匹配关键词
        
        Args:
            content: 文档内容
            keywords: 关键词列表
        
        Returns:
            匹配的关键词数量
        """
        if not keywords:
            return 0
        
        # 获取关键词匹配配置
        keyword_config = self.classification_config.get('keyword_matching', {})
        case_sensitive = keyword_config.get('case_sensitive', False)
        exact_match_only = keyword_config.get('exact_match_only', False)
        
        # 如果不区分大小写，转换内容为小写
        if not case_sensitive:
            content = content.lower()
            keywords = [kw.lower() for kw in keywords]
        
        # 统计匹配的关键词数量
        matched_count = 0
        
        for keyword in keywords:
            if exact_match_only:
                # 精确匹配（作为完整的词）
                # 使用正则表达式确保匹配完整的词
                pattern = r'\\b' + re.escape(keyword) + r'\\b'
                if re.search(pattern, content):
                    matched_count += 1
            else:
                # 模糊匹配（作为子字符串）
                if keyword in content:
                    matched_count += 1
        
        return matched_count
    
    def _extract_metadata(self, file_path: str, content: str) -> Dict[str, str]:
        """从文档中提取元数据
        
        Args:
            file_path: 文档路径
            content: 文档内容
        
        Returns:
            提取的元数据字典
        """
        metadata = {}
        
        # 检查是否启用元数据提取
        if not self.classification_config.get('metadata_extraction', {}).get('enabled', True):
            return metadata
        
        # 从文件路径提取基本信息
        file_name = os.path.basename(file_path)
        metadata['file_name'] = file_name
        
        # 尝试提取文件创建时间和修改时间
        if os.path.exists(file_path):
            metadata['file_size'] = str(os.path.getsize(file_path)) + ' bytes'
            metadata['last_modified'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(file_path)))
        
        # 从内容中提取标题（尝试前几行）
        lines = content.strip().split('\n')[:10]
        for line in lines:
            stripped_line = line.strip()
            if stripped_line and len(stripped_line) > 5 and len(stripped_line) < 200:
                # 简单启发式：查找可能的标题行
                # 假设标题行没有缩进，并且不包含太多特殊字符
                if not re.search(r'^\\s', stripped_line) and len(re.findall(r'[^\w\s，。,\.]', stripped_line)) < len(stripped_line) * 0.2:
                    metadata['detected_title'] = stripped_line
                    break
        
        # 限制元数据数量
        max_metadata_count = self.classification_config.get('metadata_extraction', {}).get('max_metadata_count', 10)
        if len(metadata) > max_metadata_count:
            # 保留最重要的元数据
            important_keys = ['file_name', 'detected_title', 'file_size', 'last_modified']
            filtered_metadata = {k: v for k, v in metadata.items() if k in important_keys}
            # 如果重要元数据不足，添加其他元数据
            remaining_keys = [k for k in metadata.keys() if k not in important_keys]
            for k in remaining_keys[:max_metadata_count - len(filtered_metadata)]:
                filtered_metadata[k] = metadata[k]
            metadata = filtered_metadata
        
        return metadata
    
    def classify_document(self, file_path: str, document_content: str = None) -> Dict[str, Any]:
        """对文档进行分类
        
        Args:
            file_path: 文档路径
            document_content: 文档内容，如果为None则尝试从文件中读取
        
        Returns:
            分类结果，包含文档类型、置信度等信息
        """
        # 验证文件路径
        if not file_path:
            logger.error("文档路径不能为空")
            return {
                'document_type': 'unknown',
                'confidence': 0.0,
                'metadata': {},
                'classification_method': 'error',
                'error': '文档路径不能为空'
            }
        
        # 如果没有提供文档内容，尝试从文件中读取
        if document_content is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    document_content = f.read()
            except UnicodeDecodeError:
                # 尝试其他编码
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        document_content = f.read()
                except Exception as e:
                    logger.error(f"读取文件内容失败: {str(e)}")
                    return {
                        'document_type': 'unknown',
                        'confidence': 0.0,
                        'metadata': {},
                        'classification_method': 'error',
                        'error': f'读取文件内容失败: {str(e)}'
                    }
            except Exception as e:
                logger.error(f"读取文件内容失败: {str(e)}")
                return {
                    'document_type': 'unknown',
                    'confidence': 0.0,
                    'metadata': {},
                    'classification_method': 'error',
                    'error': f'读取文件内容失败: {str(e)}'
                }
        
        # 检查缓存
        content_sample = document_content[:2000]  # 使用前2000个字符作为样本用于缓存键
        cached_result = self._get_from_cache(file_path, content_sample)
        if cached_result:
            return cached_result
        
        # 采样文档内容用于分类
        sampled_content = self._sample_document_content(document_content)
        
        # 提取元数据
        metadata = self._extract_metadata(file_path, document_content)
        
        # 尝试使用机器学习模型分类
        if self.ml_model:
            try:
                ml_result = self._classify_with_ml(sampled_content)
                if ml_result and ml_result['confidence'] > 0.7:
                    # 如果机器学习结果置信度足够高，直接返回
                    result = {
                        'document_type': ml_result['document_type'],
                        'confidence': ml_result['confidence'],
                        'metadata': metadata,
                        'classification_method': 'machine_learning'
                    }
                    # 存储到缓存
                    self._store_in_cache(file_path, content_sample, result)
                    return result
            except Exception as e:
                logger.error(f"机器学习分类失败: {str(e)}")
                # 如果配置了回退到规则分类，则继续进行规则分类
                if not self.classification_config.get('ml_assistance', {}).get('fallback_to_rules', True):
                    return {
                        'document_type': 'unknown',
                        'confidence': 0.0,
                        'metadata': metadata,
                        'classification_method': 'error',
                        'error': f'机器学习分类失败且未启用回退: {str(e)}'
                    }
        
        # 基于规则的分类
        result = self._classify_with_rules(file_path, sampled_content, metadata)
        
        # 存储到缓存
        self._store_in_cache(file_path, content_sample, result)
        
        return result
    
    def _classify_with_ml(self, content: str) -> Optional[Dict[str, Any]]:
        """使用机器学习模型进行文档分类
        
        Args:
            content: 文档内容
        
        Returns:
            分类结果，如果分类失败则返回None
        """
        # 注意：这是一个示例实现，实际应用中需要根据加载的模型进行相应的处理
        # 由于没有实际加载的模型，这里返回None
        logger.warning("机器学习分类模型未实际加载，返回None")
        return None
    
    def _classify_with_rules(self, file_path: str, content: str, metadata: Dict[str, str]) -> Dict[str, Any]:
        """使用基于规则的方法进行文档分类
        
        Args:
            file_path: 文档路径
            content: 文档内容
            metadata: 文档元数据
        
        Returns:
            分类结果
        """
        # 获取文件扩展名
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # 存储每个文档类型的匹配得分
        type_scores = {}
        
        # 对每种文档类型进行评分
        for doc_type in self.document_types:
            score = 0
            confidence = 0.0
            
            # 检查文件扩展名
            if file_extension in doc_type.get('file_extensions', []):
                score += 20
            
            # 匹配关键词
            keywords = doc_type.get('keywords', [])
            matched_keywords = self._match_keywords(content, keywords)
            
            # 计算关键词匹配得分
            if keywords:
                keyword_score = (matched_keywords / len(keywords)) * 80  # 关键词匹配最高得80分
                score += keyword_score
                
                # 检查是否满足最小关键词匹配数量
                min_keywords_match = doc_type.get('min_keywords_match', 0)
                if matched_keywords < min_keywords_match:
                    # 如果不满足最小关键词匹配数量，跳过这种类型
                    continue
            
            # 计算置信度
            confidence_threshold = doc_type.get('confidence_threshold', 0.5)
            confidence = min(score / 100.0, 1.0)  # 归一化到0-1之间
            
            # 如果置信度达到阈值，添加到候选类型
            if confidence >= confidence_threshold:
                type_scores[doc_type['id']] = {
                    'name': doc_type['name'],
                    'confidence': confidence,
                    'matched_keywords': matched_keywords,
                    'total_keywords': len(keywords)
                }
        
        # 如果没有匹配的类型，使用通用类型
        if not type_scores:
            return {
                'document_type': 'general',
                'document_type_name': '普通文档',
                'confidence': 0.5,
                'metadata': metadata,
                'classification_method': 'rule_based',
                'details': '未找到匹配的文档类型，使用通用类型'
            }
        
        # 选择置信度最高的类型
        best_type_id = max(type_scores, key=lambda x: type_scores[x]['confidence'])
        best_type = type_scores[best_type_id]
        
        return {
            'document_type': best_type_id,
            'document_type_name': best_type['name'],
            'confidence': best_type['confidence'],
            'metadata': metadata,
            'classification_method': 'rule_based',
            'details': {
                'matched_keywords': best_type['matched_keywords'],
                'total_keywords': best_type['total_keywords'],
                'file_extension': file_extension
            }
        }
    
    def clear_cache(self) -> None:
        """清除分类缓存"""
        self._cache.clear()
        self._hash_cache.clear()
        logger.info("文档分类缓存已清除")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息
        
        Returns:
            缓存统计信息，包含缓存项数量和最大缓存大小
        """
        return {
            'current_size': len(self._hash_cache),
            'max_size': self._max_cache_size,
            'expiry_time_seconds': self._cache_ttl
        }

# 创建全局文档分类器实例
document_classifier = DocumentClassifier()

# 示例用法
if __name__ == "__main__":
    # 创建一个测试文档
    test_doc_path = "test_document.txt"
    with open(test_doc_path, 'w', encoding='utf-8') as f:
        f.write("# 机器学习在自然语言处理中的应用\n\n")
        f.write("## 摘要\n")
        f.write("本文探讨了机器学习技术在自然语言处理领域的最新应用，包括文本分类、情感分析、机器翻译等任务。\n\n")
        f.write("## 引言\n")
        f.write("自然语言处理是人工智能的一个重要分支，近年来随着深度学习技术的发展，取得了显著进展。\n")
        
    try:
        # 测试分类功能
        result = document_classifier.classify_document(test_doc_path)
        print("分类结果:")
        print(yaml.dump(result, allow_unicode=True, default_flow_style=False))
        
        # 测试缓存功能
        print("\n测试缓存功能:")
        start_time = time.time()
        result_from_cache = document_classifier.classify_document(test_doc_path)
        cache_time = time.time() - start_time
        print(f"从缓存获取结果耗时: {cache_time:.6f} 秒")
        
        # 打印缓存统计信息
        print("\n缓存统计信息:")
        print(yaml.dump(document_classifier.get_cache_stats(), allow_unicode=True, default_flow_style=False))
        
    finally:
        # 清理测试文件
        if os.path.exists(test_doc_path):
            os.remove(test_doc_path)