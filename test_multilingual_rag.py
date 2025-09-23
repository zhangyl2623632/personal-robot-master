import os
import sys
import time
import logging
import os
from typing import Dict, Any, List

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from src.adaptive_rag_pipeline import AdaptiveRAGPipeline
from src.vector_store import VectorStoreManager
from src.llm_client import llm_client
from src.config import global_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMultilingualRAGTester:
    """增强版多语言RAG流程测试器"""
    
    def __init__(self):
        """初始化测试器"""
        # 初始化核心组件，使用优化的配置文件
        self.rag_pipeline = AdaptiveRAGPipeline(config_path="config/rag_strategies_optimized.yaml")
        self.vector_store_manager = VectorStoreManager()
        
        # 测试配置
        self.expected_document = "Merchant Quick loan TRS.docx"
        
        # 添加颜色编码以增强终端输出
        self.colors = {
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'reset': '\033[0m'
        }
    
    def colorize(self, text: str, color: str) -> str:
        """为文本添加颜色"""
        if os.name == 'nt':  # Windows系统不支持颜色
            return text
        return self.colors.get(color, '') + text + self.colors['reset']
    
    def contains_english_content(self, text: str) -> bool:
        """检查文本是否包含英文内容"""
        return any(char.isalpha() and ord(char) < 128 for char in text)
    
    def is_chinese_text(self, text: str) -> bool:
        """检查文本是否主要为中文内容"""
        chinese_chars_count = 0
        total_chars_count = 0
        
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符范围
                chinese_chars_count += 1
            if char.strip():
                total_chars_count += 1
        
        return total_chars_count > 0 and (chinese_chars_count / total_chars_count) > 0.3
    
    def extract_english_keywords(self, text: str) -> List[str]:
        """从文本中提取英文关键词"""
        import re
        # 提取连续的英文单词
        english_words = re.findall(r'[A-Za-z]+(?:[\s-][A-Za-z]+)*', text)
        return [word.strip() for word in english_words if len(word.strip()) > 2]
    
    def test_multilingual_rag_pipeline(self, query: str) -> Dict[str, Any]:
        """完整测试多语言RAG流程的每一个步骤"""
        print(f"\n{self.colorize('='*80, 'cyan')}")
        print(f"{self.colorize('开始多语言RAG流程测试', 'cyan')}")
        print(f"{self.colorize('='*80, 'cyan')}")
        
        # 步骤1: 理解中文问题
        print(f"\n{self.colorize('【步骤1】理解中文问题', 'blue')}")
        print(f"中文查询: {query}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 步骤2-4: 定位英文原文、理解英文内容、提取整合信息
        print(f"\n{self.colorize('【步骤2-4】定位英文原文 -> 理解英文内容 -> 提取整合信息', 'blue')}")
        
        # 调用RAG流水线处理查询
        result = self.rag_pipeline.answer_query(query)
        
        # 分析检索到的文档
        retrieved_docs = result.get('retrieved_documents', 0)
        references = result.get('references', [])
        
        print(f"检索到相关文档数: {retrieved_docs}")
        
        # 检查是否找到目标文档并提取英文内容
        target_document_found = False
        english_content_found = False
        extracted_english_keywords = []
        
        if references:
            for i, ref in enumerate(references, 1):
                source = ref.get('source', '')
                snippet = ref.get('snippet', '')
                score = ref.get('score', '未知')
                
                # 检查是否包含目标文档
                if self.expected_document in source:
                    target_document_found = True
                    print(f"{self.colorize(f'✓ 成功定位到目标文档: {self.expected_document}', 'green')}")
                
                # 检查是否包含英文内容
                if self.contains_english_content(snippet):
                    english_content_found = True
                    # 提取英文关键词
                    keywords = self.extract_english_keywords(snippet)
                    extracted_english_keywords.extend(keywords)
                    
                    print(f"\n引用 {i}:")
                    print(f"  来源: {source}")
                    print(f"  相似度: {score}")
                    print(f"  内容片段: {snippet[:200]}...")
                    if keywords:
                        print(f"  提取的英文关键词: {', '.join(set(keywords))}")
        
        if not target_document_found:
            print(f"{self.colorize(f'✗ 未找到目标文档: {self.expected_document}', 'red')}")
        
        if not english_content_found:
            print(f"{self.colorize('✗ 未在检索到的文档中发现英文内容', 'red')}")
        
        # 步骤5-6: 翻译转换为地道中文、组织生成中文回答
        print(f"\n{self.colorize('【步骤5-6】翻译转换为地道中文 -> 组织生成中文回答', 'blue')}")
        
        answer = result.get('answer', '')
        is_chinese_answer = self.is_chinese_text(answer)
        
        print(f"生成的回答:")
        print(f"{self.colorize(answer, 'yellow')}")
        print(f"回答是否为中文: {self.colorize('是', 'green') if is_chinese_answer else self.colorize('否', 'red')}")
        
        # 评估回答质量
        answer_quality = ""
        if "根据现有资料，无法回答该问题" in answer:
            answer_quality = self.colorize("需要优化 - 未提供有效答案", 'red')
        elif len(answer) < 20:
            answer_quality = self.colorize("需要改进 - 回答过于简短", 'yellow')
        else:
            answer_quality = self.colorize("良好 - 提供了有意义的回答", 'green')
        
        print(f"回答质量评估: {answer_quality}")
        
        # 计算处理时间
        processing_time = time.time() - start_time
        print(f"\n总处理时间: {processing_time:.2f}秒")
        
        # 综合评估流程完成情况
        print(f"\n{self.colorize('【多语言RAG流程综合评估】', 'cyan')}")
        
        # 构建详细的测试结果
        test_result = {
            'query': query,
            'answer': answer,
            'is_chinese_answer': is_chinese_answer,
            'target_document_found': target_document_found,
            'english_content_found': english_content_found,
            'extracted_english_keywords': list(set(extracted_english_keywords)),
            'retrieved_documents': retrieved_docs,
            'processing_time': processing_time,
            'references': references,
            'success': result.get('success', False)
        }
        
        # 打印流程评估结果
        success_metrics = [
            ("理解中文问题", True),
            ("定位英文原文", target_document_found),
            ("理解英文内容", english_content_found),
            ("提取整合信息", len(extracted_english_keywords) > 0),
            ("翻译转换为地道中文", is_chinese_answer),
            ("组织生成有意义的中文回答", "根据现有资料，无法回答该问题" not in answer and len(answer) > 20)
        ]
        
        print("流程完成情况:")
        overall_success = True
        for step, success in success_metrics:
            status = self.colorize("✓ 成功", 'green') if success else self.colorize("✗ 失败", 'red')
            print(f"  {step}: {status}")
            if not success:
                overall_success = False
        
        print(f"\n{self.colorize('='*80, 'cyan')}")
        
        return test_result
    
    def run_comprehensive_tests(self):
        """运行全面的多语言RAG测试"""
        # 准备针对Merchant Quick loan文档的专门测试问题
        specific_test_queries = [
            "Merchant Quick loan的版本号和发布日期是什么？",
            "Merchant Quick Loan的技术规格文档包含哪些主要内容？",
            "Merchant Quick Loan的英文全称是什么？",
            "请提取Merchant Quick Loan文档中的关键技术信息？",
            "Merchant Quick loan请讲述下此文档"
        ]
        
        print(f"\n{self.colorize('开始全面多语言RAG测试', 'cyan')}")
        print(f"测试文档: {self.expected_document}")
        print(f"测试问题数量: {len(specific_test_queries)}")
        
        all_results = []
        
        # 逐个测试问题
        for query in specific_test_queries:
            result = self.test_multilingual_rag_pipeline(query)
            all_results.append(result)
            
        # 生成综合报告
        self.generate_comprehensive_report(all_results)
        
    def generate_comprehensive_report(self, all_results: List[Dict[str, Any]]):
        """生成综合测试报告"""
        print(f"\n{self.colorize('='*80, 'cyan')}")
        print(f"{self.colorize('多语言RAG流程综合测试报告', 'cyan')}")
        print(f"{self.colorize('='*80, 'cyan')}")
        
        # 统计各项指标
        total_queries = len(all_results)
        success_count = sum(1 for r in all_results if r['success']) 
        target_found_count = sum(1 for r in all_results if r['target_document_found'])
        english_content_count = sum(1 for r in all_results if r['english_content_found'])
        chinese_answer_count = sum(1 for r in all_results if r['is_chinese_answer'])
        meaningful_answer_count = sum(1 for r in all_results if "根据现有资料，无法回答该问题" not in r['answer'] and len(r['answer']) > 20)
        
        avg_processing_time = sum(r['processing_time'] for r in all_results) / total_queries if total_queries > 0 else 0
        
        # 打印统计结果
        print(f"测试查询总数: {total_queries}")
        print(f"成功处理的查询: {self.colorize(f'{success_count}/{total_queries}', 'green' if success_count == total_queries else 'yellow')}")
        print(f"成功定位目标文档: {self.colorize(f'{target_found_count}/{total_queries}', 'green' if target_found_count > 0 else 'red')}")
        print(f"成功识别英文内容: {self.colorize(f'{english_content_count}/{total_queries}', 'green' if english_content_count > 0 else 'red')}")
        print(f"生成中文回答: {self.colorize(f'{chinese_answer_count}/{total_queries}', 'green' if chinese_answer_count == total_queries else 'yellow')}")
        print(f"生成有意义的回答: {self.colorize(f'{meaningful_answer_count}/{total_queries}', 'green' if meaningful_answer_count > 0 else 'red')}")
        print(f"平均处理时间: {avg_processing_time:.2f}秒")
        
        # 提取所有英文关键词
        all_keywords = set()
        for result in all_results:
            all_keywords.update(result['extracted_english_keywords'])
        
        if all_keywords:
            print(f"\n从文档中提取的主要英文关键词: {', '.join(sorted(all_keywords))[:200]}...")
        
        # 分析和建议
        print(f"\n{self.colorize('【分析与建议】', 'blue')}")
        
        if target_found_count == 0:
            print(f"{self.colorize('✗ 问题: 未找到目标文档', 'red')}")
            print("  建议: 检查向量存储是否正确加载了目标文档")
        
        if english_content_count == 0:
            print(f"{self.colorize('✗ 问题: 未识别到英文内容', 'red')}")
            print("  建议: 检查文档内容是否确实包含英文，或优化英文检测算法")
        
        if meaningful_answer_count == 0:
            print(f"{self.colorize('✗ 问题: 未能生成有意义的回答', 'red')}")
            print("  建议: 检查LLM配置、提示模板或文档内容是否足够丰富")
            print("  建议: 考虑调整RAG策略参数，如增加检索文档数量或调整相似度阈值")
        
        # 总结
        print(f"\n{self.colorize('【总结】', 'cyan')}")
        if meaningful_answer_count > 0:
            print(f"{self.colorize('✓ 多语言RAG流程基本验证成功！', 'green')}")
            print("  系统能够理解中文问题，定位英文文档，提取信息并生成中文回答")
        else:
            print(f"{self.colorize('✗ 多语言RAG流程需要优化', 'red')}")
            print("  虽然系统能够完成部分流程，但在生成有意义回答方面仍有改进空间")
        
        print(f"\n{self.colorize('='*80, 'cyan')}")
        print(f"{self.colorize('测试完成', 'cyan')}")

if __name__ == "__main__":
    # 创建增强版测试器实例
    tester = EnhancedMultilingualRAGTester()
    
    # 运行全面测试
    tester.run_comprehensive_tests()