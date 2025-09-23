import logging
import re
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import hashlib
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryIntentClassifier:
    """查询意图分类器，用于理解用户查询的意图
    支持基于规则和关键词的意图识别，并可与RAG Pipeline集成
    """
    
    def __init__(self, config_path: str = None):
        """初始化查询意图分类器
        
        Args:
            config_path: 配置文件路径
        """
        # 定义查询意图类型
        self.intent_types = {
            "specific_detail": {  # 具体细节查询
                "name": "具体细节查询",
                "description": "用户希望获取文档中的具体信息或细节",
                "keywords": ["什么是", "什么是", "具体内容", "详细说明", "具体数字", "具体时间", "具体地点", "具体人物", "具体步骤", 
                              "如何", "怎样", "教程", "步骤", "方法", "如何操作", "怎样使用", "配置方法", "设置步骤",
                              "参数", "属性", "特性", "特征", "功能", "作用", "用途", "效果", "影响",
                              "定义", "概念", "含义", "意思", "解释", "说明", "理解"],
                "confidence_threshold": 0.6
            },
            "overview_request": {  # 概述请求
                "name": "概述请求",
                "description": "用户希望获取文档的总结或概述",
                "keywords": ["总结", "概述", "概括", "简述", "概述一下", "总结一下", "大致内容", "主要内容",
                              "要点", "重点", "核心", "中心思想", "主要观点", "主要结论", "关键内容"],
                "confidence_threshold": 0.5
            },
            "comparison_analysis": {  # 比较分析
                "name": "比较分析",
                "description": "用户希望对不同内容进行比较或分析",
                "keywords": ["比较", "对比", "区别", "不同", "差异", "优缺点", "优势", "劣势", "好处", "坏处",
                              "vs", "VS", "对比分析", "比较分析", "优缺点分析", "优势分析", "劣势分析"],
                "confidence_threshold": 0.5
            },
            "reasoning": {  # 推理分析
                "name": "推理分析",
                "description": "用户希望基于文档内容进行推理或分析",
                "keywords": ["为什么", "原因", "理由", "为什么是", "为什么会", "分析", "推理", "推断", "推测",
                              "原因分析", "结果分析", "影响分析", "趋势分析", "问题分析"],
                "confidence_threshold": 0.55
            },
            "application_advice": {  # 应用建议
                "name": "应用建议",
                "description": "用户希望获取应用或实施建议",
                "keywords": ["建议", "建议", "意见", "推荐", "应用", "实施", "执行", "应用场景", "使用场景",
                              "最佳实践", "实施步骤", "应用方法", "使用建议", "推荐方案", "解决方案"],
                "confidence_threshold": 0.5
            },
            "list_collection": {  # 列表收集
                "name": "列表收集",
                "description": "用户希望获取列表形式的信息",
                "keywords": ["有哪些", "哪些", "包括哪些", "包含哪些", "列举", "列出", "清单", "目录", "列表",
                              "汇总", "集合", "所有", "全部", "一共", "总共", "总数", "数量"],
                "confidence_threshold": 0.5
            },
            "yes_no_question": {  # 是非问题
                "name": "是非问题",
                "description": "用户提出的是或否的问题",
                "keywords": [],  # 是非问题通常通过句式识别
                "patterns": [r"^[是否].*[？吗]$", r"^.*是不是.*[？吗]$", r"^.*有没有.*[？吗]$", r"^.*对吗[？吗]?$", r"^.*对吧[？吗]?$"]
            }
        }
        
        # 初始化缓存机制
        self._init_cache()
        
        # 加载自定义配置（如果提供）
        if config_path and os.path.exists(config_path):
            self._load_custom_config(config_path)
        
        logger.info("查询意图分类器初始化完成")
    
    def _init_cache(self) -> None:
        """初始化意图分类缓存机制"""
        # 存储查询到意图分类结果的映射
        self._cache = {}
        # 缓存大小限制
        self._max_cache_size = 5000
        # 缓存过期时间（秒）
        self._cache_ttl = 3600  # 1小时
        logger.info("查询意图分类缓存机制初始化完成")
    
    def _load_custom_config(self, config_path: str) -> None:
        """加载自定义配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                custom_config = json.load(f)
                
                # 更新意图类型配置
                if 'intent_types' in custom_config:
                    for intent_id, intent_config in custom_config['intent_types'].items():
                        if intent_id in self.intent_types:
                            # 更新现有意图类型
                            self.intent_types[intent_id].update(intent_config)
                        else:
                            # 添加新的意图类型
                            self.intent_types[intent_id] = intent_config
                    
                    logger.info(f"成功加载自定义意图类型配置，当前共 {len(self.intent_types)} 种意图类型")
                
                # 更新缓存配置
                if 'cache_config' in custom_config:
                    cache_config = custom_config['cache_config']
                    if 'max_size' in cache_config:
                        self._max_cache_size = cache_config['max_size']
                    if 'ttl' in cache_config:
                        self._cache_ttl = cache_config['ttl']
                    
                    logger.info(f"更新缓存配置：最大大小={self._max_cache_size}, TTL={self._cache_ttl}秒")
                    
        except Exception as e:
            logger.error(f"加载自定义配置失败: {str(e)}")
    
    def _generate_query_hash(self, query: str) -> str:
        """生成查询的哈希值，用于缓存
        
        Args:
            query: 用户查询
        
        Returns:
            查询的哈希值
        """
        # 对查询进行标准化处理
        normalized_query = query.lower().strip()
        # 生成哈希
        return hashlib.md5(normalized_query.encode('utf-8')).hexdigest()
    
    def _get_from_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """从缓存中获取意图分类结果
        
        Args:
            query: 用户查询
        
        Returns:
            意图分类结果，如果缓存中不存在则返回None
        """
        # 生成查询哈希
        query_hash = self._generate_query_hash(query)
        
        # 检查缓存
        if query_hash in self._cache:
            cache_entry = self._cache[query_hash]
            # 检查缓存是否过期
            if time.time() < cache_entry['expiry_time']:
                logger.debug(f"从缓存中获取查询 '{query}' 的意图分类结果")
                return cache_entry['result']
            else:
                # 缓存已过期，移除
                del self._cache[query_hash]
        
        return None
    
    def _store_in_cache(self, query: str, result: Dict[str, Any]) -> None:
        """将意图分类结果存储到缓存中
        
        Args:
            query: 用户查询
            result: 意图分类结果
        """
        # 检查缓存大小
        if len(self._cache) >= self._max_cache_size:
            # 移除最早的缓存项
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        # 生成查询哈希
        query_hash = self._generate_query_hash(query)
        
        # 计算过期时间
        expiry_time = time.time() + self._cache_ttl
        
        # 存储到缓存
        self._cache[query_hash] = {
            'result': result,
            'expiry_time': expiry_time,
            'query': query
        }
        
        logger.debug(f"将查询 '{query}' 的意图分类结果存储到缓存")
    
    def _match_keywords(self, query: str, keywords: List[str]) -> int:
        """在查询中匹配关键词
        
        Args:
            query: 用户查询
            keywords: 关键词列表
        
        Returns:
            匹配的关键词数量
        """
        if not keywords:
            return 0
        
        # 转换查询为小写以进行不区分大小写的匹配
        query_lower = query.lower()
        
        # 统计匹配的关键词数量
        matched_count = 0
        for keyword in keywords:
            if keyword.lower() in query_lower:
                matched_count += 1
        
        return matched_count
    
    def _match_patterns(self, query: str, patterns: List[str]) -> bool:
        """使用正则表达式匹配查询模式
        
        Args:
            query: 用户查询
            patterns: 正则表达式模式列表
        
        Returns:
            是否匹配任何模式
        """
        if not patterns:
            return False
        
        # 转换查询为小写以进行不区分大小写的匹配
        query_lower = query.lower()
        
        # 检查是否匹配任何模式
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def _detect_yes_no_question(self, query: str) -> bool:
        """检测查询是否为是非问题
        
        Args:
            query: 用户查询
        
        Returns:
            是否为是非问题
        """
        # 使用正则表达式检测是非问题的典型句式
        yes_no_patterns = [
            r"^[是否].*[？吗]$",  # 以"是"或"否"开头，以问号或"吗"结尾
            r"^.*是不是.*[？吗]$",  # 包含"是不是"，以问号或"吗"结尾
            r"^.*有没有.*[？吗]$",  # 包含"有没有"，以问号或"吗"结尾
            r"^.*对吗[？吗]?$",  # 以"对吗"结尾，可选问号
            r"^.*对吧[？吗]?$",  # 以"对吧"结尾，可选问号
            r"^.*可以吗[？吗]?$",  # 以"可以吗"结尾，可选问号
            r"^.*行吗[？吗]?$",  # 以"行吗"结尾，可选问号
            r"^.*好不好[？吗]?$",  # 以"好不好"结尾，可选问号
        ]
        
        return self._match_patterns(query, yes_no_patterns)
    
    def classify_intent(self, query: str, document_type: str = None) -> Dict[str, Any]:
        """对用户查询进行意图分类
        
        Args:
            query: 用户查询
            document_type: 文档类型（可选），用于更精确的意图识别
        
        Returns:
            意图分类结果，包含意图类型、置信度等信息
        """
        # 验证查询
        if not query or not query.strip():
            logger.error("查询内容为空")
            return {
                'intent_type': 'unknown',
                'intent_name': '未知意图',
                'confidence': 0.0,
                'reason': '查询内容为空'
            }
        
        # 标准化查询
        query = query.strip()
        
        # 检查缓存
        cached_result = self._get_from_cache(query)
        if cached_result:
            return cached_result
        
        # 存储每个意图类型的匹配得分
        intent_scores = {}
        total_keywords = 0
        
        # 对每种意图类型进行评分
        for intent_id, intent_config in self.intent_types.items():
            score = 0
            confidence = 0.0
            matched_keywords = 0
            
            # 检查关键词匹配
            keywords = intent_config.get('keywords', [])
            if keywords:
                matched_keywords = self._match_keywords(query, keywords)
                # 计算关键词匹配得分（最高60分）
                keyword_score = (matched_keywords / len(keywords)) * 60 if keywords else 0
                score += keyword_score
                
                # 累加总关键词数用于归一化
                total_keywords += len(keywords)
            
            # 检查模式匹配（如果有）
            patterns = intent_config.get('patterns', [])
            if patterns and self._match_patterns(query, patterns):
                # 模式匹配加20分
                score += 20
            
            # 特殊处理：是非问题检测
            if intent_id == 'yes_no_question':
                if self._detect_yes_no_question(query):
                    # 如果是是非问题，直接给高分
                    score = 80
                    matched_keywords = 1
            
            # 归一化得分到0-1之间
            if score > 0:
                confidence = min(score / 100.0, 1.0)
            
            # 应用文档类型特定的调整（如果提供）
            if document_type:
                # 这里可以根据文档类型对置信度进行调整
                # 例如，对于学术论文，推理分析类问题的置信度可以适当提高
                doc_type_adjustments = {
                    'academic_paper': {'reasoning': 1.1, 'specific_detail': 1.05},
                    'requirement_doc': {'specific_detail': 1.1, 'list_collection': 1.05},
                    'novel': {'overview_request': 1.05, 'reasoning': 1.05}
                }
                
                if document_type in doc_type_adjustments and intent_id in doc_type_adjustments[document_type]:
                    confidence *= doc_type_adjustments[document_type][intent_id]
                    confidence = min(confidence, 1.0)  # 确保不超过1
            
            # 检查是否满足置信度阈值
            confidence_threshold = intent_config.get('confidence_threshold', 0.5)
            if confidence >= confidence_threshold:
                intent_scores[intent_id] = {
                    'name': intent_config['name'],
                    'confidence': confidence,
                    'matched_keywords': matched_keywords,
                    'total_keywords': len(keywords)
                }
        
        # 如果没有匹配的意图类型，设置为未知意图
        if not intent_scores:
            result = {
                'intent_type': 'unknown',
                'intent_name': '未知意图',
                'confidence': 0.5,
                'reason': '未匹配到已知意图类型'
            }
        else:
            # 选择置信度最高的意图类型
            best_intent_id = max(intent_scores, key=lambda x: intent_scores[x]['confidence'])
            best_intent = intent_scores[best_intent_id]
            
            # 生成详细的理由
            reason = f"匹配意图类型 '{best_intent['name']}'，"
            if best_intent['total_keywords'] > 0:
                reason += f"匹配 {best_intent['matched_keywords']} 个关键词（共 {best_intent['total_keywords']} 个），"
            reason += f"置信度：{best_intent['confidence']:.2f}"
            
            result = {
                'intent_type': best_intent_id,
                'intent_name': best_intent['name'],
                'confidence': best_intent['confidence'],
                'reason': reason,
                'details': {
                    'matched_keywords': best_intent['matched_keywords'],
                    'total_keywords': best_intent['total_keywords'],
                    'all_matched_intents': {k: v['confidence'] for k, v in intent_scores.items()}
                }
            }
        
        # 存储到缓存
        self._store_in_cache(query, result)
        
        return result
    
    def clear_cache(self) -> None:
        """清除意图分类缓存"""
        self._cache.clear()
        logger.info("查询意图分类缓存已清除")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息
        
        Returns:
            缓存统计信息，包含缓存项数量和最大缓存大小
        """
        return {
            'current_size': len(self._cache),
            'max_size': self._max_cache_size,
            'expiry_time_seconds': self._cache_ttl
        }
    
    def extract_keywords(self, query: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """从查询中提取关键词
        
        Args:
            query: 用户查询
            top_n: 返回的关键词数量
        
        Returns:
            关键词及其权重的列表
        """
        if not query or not query.strip():
            return []
        
        # 简单的关键词提取实现
        # 在实际应用中，可以考虑使用更复杂的NLP技术如TF-IDF或TextRank
        
        # 移除常见停用词
        stopwords = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', 
            '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', 
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when', 'where', 
            'how', 'who', 'which', 'this', 'that', 'these', 'those'
        ])
        
        # 使用正则表达式提取单词
        words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', query)
        
        # 过滤停用词
        filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
        
        # 计算词频
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 对关键词进行排序
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 计算权重（归一化词频）
        total_freq = sum(word_freq.values()) if word_freq else 1
        weighted_keywords = [(word, freq / total_freq) for word, freq in sorted_keywords]
        
        # 返回前N个关键词
        return weighted_keywords[:top_n]

# 创建全局查询意图分类器实例
query_intent_classifier = QueryIntentClassifier()

# 示例用法
if __name__ == "__main__":
    # 测试不同类型的查询
    test_queries = [
        "什么是人工智能的定义？",
        "总结一下这篇文章的主要内容",
        "比较深度学习和传统机器学习的区别",
        "为什么会出现这种情况？请分析原因",
        "对于这个问题，你有什么建议吗？",
        "列举出文档中提到的所有技术点",
        "这个解决方案可行吗？"
    ]
    
    print("\n=== 测试查询意图分类 ===")
    
    for query in test_queries:
        result = query_intent_classifier.classify_intent(query)
        print(f"\n查询: {query}")
        print(f"意图类型: {result['intent_type']} ({result['intent_name']})")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"理由: {result['reason']}")
        
        # 提取关键词
        keywords = query_intent_classifier.extract_keywords(query)
        print("提取的关键词:")
        for word, weight in keywords:
            print(f"  - {word}: {weight:.2f}")
    
    # 测试缓存功能
    print("\n=== 测试缓存功能 ===")
    start_time = time.time()
    result_from_cache = query_intent_classifier.classify_intent("什么是人工智能的定义？")
    cache_time = time.time() - start_time
    print(f"从缓存获取结果耗时: {cache_time:.6f} 秒")
    
    # 打印缓存统计信息
    print("\n缓存统计信息:")
    print(query_intent_classifier.get_cache_stats())