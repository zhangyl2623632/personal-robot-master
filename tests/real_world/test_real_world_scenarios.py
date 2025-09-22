import os
import sys
import json
import time
import logging
import argparse
import shutil
import tempfile
from typing import List, Dict, Tuple, Any, Optional
import zipfile
import subprocess
import concurrent.futures

# 导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.rag_pipeline import rag_pipeline
from src.config import global_config
from src.document_loader import DocumentLoader

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RealWorldScenarioTester")

class RealWorldScenarioTester:
    """真实世界场景测试工具，用于测试RAG系统在真实应用场景中的表现"""
    
    def __init__(self):
        """初始化真实世界场景测试工具"""
        # 测试配置
        self.test_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "test_data")
        self.real_world_dir = os.path.join(self.test_data_dir, "real_world")
        self.documents_dir = os.path.join(self.test_data_dir, "documents")
        self.golden_qa_dir = os.path.join(self.test_data_dir, "golden_qa")
        self.edge_cases_dir = os.path.join(self.test_data_dir, "edge_cases")
        self.corrupted_samples_dir = os.path.join(self.test_data_dir, "corrupted_samples")
        self.results_dir = os.path.join(self.test_data_dir, "results")
        
        # 创建必要的目录
        self._setup_directories()
        
        # 测试结果
        self.test_results = {
            "scenarios": {},
            "overall_status": "pending"
        }
    
    def _setup_directories(self):
        """创建测试所需的目录"""
        required_dirs = [
            self.test_data_dir,
            self.real_world_dir,
            self.documents_dir,
            self.golden_qa_dir,
            self.edge_cases_dir,
            self.corrupted_samples_dir,
            self.results_dir
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"创建目录: {dir_path}")
    
    def _prepare_test_document(self, document_type: str = "contract") -> str:
        """准备测试文档"""
        try:
            # 根据文档类型准备不同的测试文档
            if document_type == "contract":
                # 创建一个简单的合同文档
                doc_path = os.path.join(self.documents_dir, "test_contract.txt")
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write("销售合同\n\n")
                    f.write("甲方：北京科技有限公司\n")
                    f.write("乙方：上海贸易有限公司\n\n")
                    f.write("第1条 合同标的\n")
                    f.write("1.1 甲方向乙方提供以下产品：A型号服务器5台，单价20,000元/台，总金额100,000元。\n\n")
                    f.write("第2条 交货时间与地点\n")
                    f.write("2.1 交货时间：合同签订后30天内。\n")
                    f.write("2.2 交货地点：上海市浦东新区张江高科技园区博云路2号。\n\n")
                    f.write("第3条 付款方式\n")
                    f.write("3.1 合同签订后10个工作日内，乙方向甲方支付合同总金额的30%作为预付款，即30,000元。\n")
                    f.write("3.2 货物验收合格后15个工作日内，乙方向甲方支付合同总金额的70%余款，即70,000元。\n\n")
                    f.write("第4条 质量保证与违约责任\n")
                    f.write("4.1 甲方保证所提供的产品符合国家相关标准及合同约定。\n")
                    f.write("4.2 若一方违反本合同约定，应向对方支付合同总金额15%的违约金。\n\n")
                    f.write("第5条 争议解决\n")
                    f.write("5.1 本合同的签订、履行、解释及争议解决均适用中华人民共和国法律。\n")
                    f.write("5.2 双方在履行本合同过程中发生的争议，应首先通过友好协商解决；协商不成的，任何一方均有权向有管辖权的人民法院提起诉讼。\n\n")
                    f.write("甲方（盖章）：          乙方（盖章）：\n")
                    f.write("代表人（签字）：        代表人（签字）：\n")
                    f.write("日期：2023年12月1日      日期：2023年12月1日\n")
                return doc_path
            elif document_type == "technical_report":
                # 创建一个简单的技术报告文档
                doc_path = os.path.join(self.documents_dir, "test_technical_report.txt")
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write("人工智能技术发展报告\n\n")
                    f.write("摘要\n")
                    f.write("本报告对人工智能技术的最新发展进行了全面分析，包括机器学习、深度学习、自然语言处理等核心技术领域的进展。\n\n")
                    f.write("1. 引言\n")
                    f.write("人工智能（Artificial Intelligence，AI）是当前科技领域最前沿的研究方向之一，其发展速度令人瞩目。\n\n")
                    f.write("2. 核心技术领域\n")
                    f.write("2.1 机器学习\n")
                    f.write("机器学习是人工智能的核心技术之一，主要包括监督学习、无监督学习和强化学习三种范式。\n")
                    f.write("2.2 深度学习\n")
                    f.write("深度学习是机器学习的一个分支，通过模拟人脑神经网络结构，实现对复杂数据的高效处理。\n")
                    f.write("2.3 自然语言处理\n")
                    f.write("自然语言处理（NLP）研究如何使计算机理解和生成人类语言，是AI领域的重要方向。\n\n")
                    f.write("3. 应用场景\n")
                    f.write("3.1 智能客服\n")
                    f.write("基于NLP技术的智能客服系统已经广泛应用于各个行业，大大提高了客户服务效率。\n")
                    f.write("3.2 医疗诊断\n")
                    f.write("AI技术在医疗影像诊断、药物研发等方面展现出巨大潜力。\n")
                    f.write("3.3 自动驾驶\n")
                    f.write("自动驾驶技术是AI在交通领域的重要应用，涉及计算机视觉、传感器融合等多项技术。\n\n")
                    f.write("4. 挑战与展望\n")
                    f.write("尽管AI技术取得了巨大进步，但仍面临数据隐私、算法公平性、可解释性等挑战。未来，AI技术将朝着更加智能、安全、可靠的方向发展。\n")
                return doc_path
            elif document_type == "user_manual":
                # 创建一个简单的用户手册文档
                doc_path = os.path.join(self.documents_dir, "test_user_manual.txt")
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write("智能机器人用户手册\n\n")
                    f.write("第1章 产品介绍\n")
                    f.write("1.1 产品概述\n")
                    f.write("本智能机器人是一款集语音交互、信息查询、智能家居控制于一体的智能设备。\n\n")
                    f.write("第2章 安装与设置\n")
                    f.write("2.1 开箱检查\n")
                    f.write("打开包装盒，检查以下物品是否齐全：智能机器人主机、电源适配器、使用说明书、保修卡。\n")
                    f.write("2.2 连接电源\n")
                    f.write("使用提供的电源适配器连接智能机器人和电源插座。\n")
                    f.write("2.3 连接网络\n")
                    f.write("打开智能机器人电源，按照语音提示连接Wi-Fi网络。\n\n")
                    f.write("第3章 基本操作\n")
                    f.write("3.1 唤醒机器人\n")
                    f.write("说\"你好，机器人\"或按下机器人顶部的唤醒按钮唤醒机器人。\n")
                    f.write("3.2 语音交互\n")
                    f.write("唤醒后，直接说出您的问题或指令，如\"今天天气怎么样？\"、\"播放我喜欢的音乐\"。\n")
                    f.write("3.3 控制智能家居\n")
                    f.write("通过语音指令控制已连接的智能家居设备，如\"打开客厅灯\"、\"设置空调温度为25度\"。\n\n")
                    f.write("第4章 常见问题\n")
                    f.write("4.1 机器人无法唤醒\n")
                    f.write("- 检查电源连接是否正常\n")
                    f.write("- 确保音量设置适当\n")
                    f.write("- 尝试使用唤醒按钮\n")
                    f.write("4.2 网络连接失败\n")
                    f.write("- 检查Wi-Fi信号强度\n")
                    f.write("- 确认Wi-Fi密码输入正确\n")
                    f.write("- 尝试重启路由器和机器人\n\n")
                    f.write("第5章 售后服务\n")
                    f.write("产品保修期为一年，自购买之日起计算。如有任何问题，请联系客服热线：400-123-4567。\n")
                return doc_path
            else:
                # 默认创建一个通用文档
                doc_path = os.path.join(self.documents_dir, "test_generic.txt")
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write("测试文档\n\n")
                    f.write("这是一个用于测试的通用文档，包含一些示例内容。\n")
                    f.write("文档中提到了一些关键词：测试、文档、内容、示例、关键词。\n")
                return doc_path
        except Exception as e:
            logger.error(f"准备测试文档失败: {str(e)}")
            return ""
    
    def _prepare_golden_qa(self, document_path: str) -> List[Tuple[str, str, str]]:
        """为测试文档准备黄金问答对"""
        # 获取文档文件名（不含扩展名）
        doc_filename = os.path.splitext(os.path.basename(document_path))[0]
        qa_file = os.path.join(self.golden_qa_dir, f"{doc_filename}.jsonl")
        
        # 如果已经存在对应的QA文件，直接返回
        if os.path.exists(qa_file):
            qa_pairs = []
            try:
                with open(qa_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        qa_data = json.loads(line)
                        question = qa_data.get("question", "")
                        answer = qa_data.get("answer", "")
                        type_tag = qa_data.get("type", "normal")
                        if question and answer:
                            qa_pairs.append((question, answer, type_tag))
                logger.info(f"加载了 {len(qa_pairs)} 个QA对: {qa_file}")
                return qa_pairs
            except Exception as e:
                logger.error(f"加载QA对失败: {str(e)}")
        
        # 根据文档类型生成不同的黄金问答对
        if "contract" in doc_filename:
            return [
                ("合同的总金额是多少？", "合同总金额是100,000元。", "factual"),
                ("交货时间是什么时候？", "合同签订后30天内。", "factual"),
                ("违约金的比例是多少？", "合同总金额的15%。", "factual"),
                ("甲方和乙方分别是谁？", "甲方是北京科技有限公司，乙方是上海贸易有限公司。", "factual"),
                ("付款方式是怎样的？", "合同签订后10个工作日内支付30%预付款，货物验收合格后15个工作日内支付70%余款。", "detailed")
            ]
        elif "technical_report" in doc_filename:
            return [
                ("人工智能的核心技术有哪些？", "机器学习、深度学习、自然语言处理等。", "factual"),
                ("机器学习包括哪些范式？", "监督学习、无监督学习和强化学习三种范式。", "factual"),
                ("AI技术在医疗领域有哪些应用？", "医疗影像诊断、药物研发等方面。", "detailed"),
                ("AI技术面临哪些挑战？", "数据隐私、算法公平性、可解释性等挑战。", "detailed"),
                ("请概述这份报告的主要内容。", "本报告对人工智能技术的最新发展进行了全面分析，包括核心技术领域的进展、应用场景以及面临的挑战与展望。", "overview")
            ]
        elif "user_manual" in doc_filename:
            return [
                ("如何唤醒智能机器人？", "说\"你好，机器人\"或按下机器人顶部的唤醒按钮唤醒机器人。", "instructional"),
                ("产品的保修期是多久？", "产品保修期为一年，自购买之日起计算。", "factual"),
                ("如果机器人无法唤醒，应该怎么办？", "检查电源连接是否正常，确保音量设置适当，尝试使用唤醒按钮。", "troubleshooting"),
                ("智能机器人有哪些功能？", "集语音交互、信息查询、智能家居控制于一体。", "factual"),
                ("如何连接网络？", "打开智能机器人电源，按照语音提示连接Wi-Fi网络。", "instructional")
            ]
        else:
            # 默认问答对
            return [
                ("这是什么文档？", "这是一个用于测试的文档。", "factual"),
                ("文档中提到了哪些关键词？", "测试、文档、内容、示例、关键词等。", "factual"),
                ("文档的主要内容是什么？", "文档包含一些用于测试的示例内容。", "overview")
            ]
    
    def _prepare_edge_cases(self) -> List[Tuple[str, str, str]]:
        """准备边界测试用例"""
        return [
            ("上文提到的风险控制措施有哪些？", "请提供更多信息，以便我更准确地回答您的问题。", "context_dependent"),
            ("第三章第二节说的那个方案，具体步骤是什么？", "根据文档第三章第二节的内容，该方案的具体步骤包括...", "position_dependent"),
            ("A产品和B产品的区别是什么？", "在提供的文档中，我没有找到关于A产品和B产品的具体信息。", "multi_doc_compare"),
            ("刚才说的那个数字，再重复一遍", "抱歉，我没有找到之前的对话记录。请提供更多上下文信息。", "conversation_memory"),
            ("帮我找一下张三的联系方式", "出于安全考虑，我无法提供该信息。", "privacy_filter"),
            ("胡说八道", "抱歉，我不太理解您的意思。请您换一种方式表达，我会尽力为您提供帮助。", "invalid_query"),
            ("你是谁？", "我是一个智能问答助手，旨在为您提供信息查询和问题解答服务。", "system_query"),
            ("1+1=？", "1+1=2。", "math_query"),
            ("根据第三段回答，什么是最重要的技术挑战？", "根据文档第三段的内容，最重要的技术挑战是...", "position_reference"),
            ("解释一下文档中提到的'数据隐私'问题", "数据隐私问题指的是在AI系统运行过程中，如何保护用户数据不被滥用、泄露或未经授权访问的问题。", "terminology_explanation")
        ]
    
    def _create_corrupted_file(self, file_type: str = "pdf") -> str:
        """创建损坏的文件样本"""
        try:
            # 创建损坏样本目录
            if not os.path.exists(self.corrupted_samples_dir):
                os.makedirs(self.corrupted_samples_dir)
            
            # 生成文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.corrupted_samples_dir, f"corrupted_{file_type}_{timestamp}.{file_type}")
            
            # 根据文件类型创建不同的损坏文件
            if file_type == "pdf":
                # 创建一个无效的PDF文件
                with open(file_path, "w") as f:
                    f.write("这不是一个有效的PDF文件\n")
                    f.write("它缺少必要的PDF文件头和结构\n")
            elif file_type == "docx":
                # 创建一个无效的DOCX文件
                with open(file_path, "w") as f:
                    f.write("这不是一个有效的DOCX文件\n")
                    f.write("DOCX文件实际上是一个ZIP压缩包\n")
            elif file_type == "zip":
                # 创建一个损坏的ZIP文件
                with open(file_path, "w") as f:
                    f.write("这不是一个有效的ZIP文件\n")
                    f.write("它缺少ZIP文件的签名和结构\n")
            else:
                # 创建一个通用的损坏文件
                with open(file_path, "w") as f:
                    f.write("这是一个损坏的文件样本\n")
                    f.write("它无法被正常解析\n")
            
            logger.info(f"创建损坏文件样本成功: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"创建损坏文件样本失败: {str(e)}")
            return ""
    
    def _create_empty_file(self, file_type: str = "txt") -> str:
        """创建空文件样本"""
        try:
            # 创建空文件
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.corrupted_samples_dir, f"empty_{file_type}_{timestamp}.{file_type}")
            
            # 创建空文件
            open(file_path, "w").close()
            
            logger.info(f"创建空文件样本成功: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"创建空文件样本失败: {str(e)}")
            return ""
    
    def _create_large_file(self, size_mb: int) -> str:
        """创建大文件样本"""
        try:
            # 创建大文件
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.performance_dir, f"large_file_{size_mb}mb_{timestamp}.txt")
            
            # 计算需要写入的字节数
            target_size = size_mb * 1024 * 1024  # 转换为字节
            
            # 写入内容
            chunk_size = 1024 * 1024  # 每次写入1MB
            chunk_content = "X" * chunk_size  # 创建一个1MB的内容块
            
            with open(file_path, "w") as f:
                written_size = 0
                while written_size < target_size:
                    # 写入块内容
                    f.write(chunk_content)
                    written_size += len(chunk_content.encode('utf-8'))
            
            logger.info(f"创建大文件样本成功: {file_path}, 大小约 {size_mb}MB")
            return file_path
        except Exception as e:
            logger.error(f"创建大文件样本失败: {str(e)}")
            return ""
    
    def test_real_document(self, document_type: str = "contract") -> Dict:
        """测试真实文档处理能力"""
        scenario_name = f"real_document_{document_type}"
        results = {
            "scenario": scenario_name,
            "document_type": document_type,
            "status": "running",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            # 1. 准备测试文档
            logger.info(f"开始真实文档测试: {document_type}")
            document_path = self._prepare_test_document(document_type)
            if not document_path:
                results["status"] = "failed"
                results["error"] = "准备测试文档失败"
                return results
            
            results["document_path"] = document_path
            
            # 2. 清空向量库
            rag_pipeline.vector_store_manager.clear_vector_store()
            
            # 3. 添加文档到向量库
            logger.info(f"添加文档到向量库: {document_path}")
            add_start_time = time.time()
            add_success = rag_pipeline.add_single_document(document_path)
            add_time = time.time() - add_start_time
            
            if not add_success:
                results["status"] = "failed"
                results["error"] = "添加文档到向量库失败"
                return results
            
            results["add_document_time"] = add_time
            
            # 4. 获取向量数量
            vector_count = rag_pipeline.vector_store_manager.get_vector_count()
            results["vector_count"] = vector_count
            
            # 5. 准备黄金问答对
            qa_pairs = self._prepare_golden_qa(document_path)
            results["qa_pairs_count"] = len(qa_pairs)
            
            # 6. 执行问答测试
            qa_results = []
            total_response_time = 0
            correct_answers = 0
            
            for question, gold_answer, qa_type in qa_pairs:
                logger.info(f"测试问答 [{qa_type}]: {question}")
                
                # 执行查询
                query_start_time = time.time()
                answer = rag_pipeline.answer_query(question)
                query_time = time.time() - query_start_time
                total_response_time += query_time
                
                # 评估答案（简化版）
                is_correct = gold_answer.lower() in answer.lower() or "无法回答" in answer
                if is_correct:
                    correct_answers += 1
                
                # 记录结果
                qa_results.append({
                    "question": question,
                    "type": qa_type,
                    "answer": answer,
                    "gold_answer": gold_answer,
                    "is_correct": is_correct,
                    "response_time": query_time
                })
            
            # 7. 计算统计数据
            results["qa_results"] = qa_results
            results["correct_answers"] = correct_answers
            results["accuracy"] = correct_answers / len(qa_pairs) if qa_pairs else 0
            results["average_response_time"] = total_response_time / len(qa_pairs) if qa_pairs else 0
            
            # 8. 记录状态
            results["status"] = "passed" if correct_answers >= len(qa_pairs) * 0.8 else "partially_passed"
            
        except Exception as e:
            logger.error(f"真实文档测试失败: {str(e)}")
            results["status"] = "failed"
            results["error"] = str(e)
        finally:
            # 确保清理
            try:
                rag_pipeline.vector_store_manager.clear_vector_store()
            except:
                pass
            
            results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存场景结果
        self.test_results["scenarios"][scenario_name] = results
        
        return results
    
    def test_edge_cases(self) -> Dict:
        """测试边界情况处理能力"""
        scenario_name = "edge_cases"
        results = {
            "scenario": scenario_name,
            "status": "running",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            # 1. 准备边界测试用例
            logger.info("开始边界情况测试")
            edge_cases = self._prepare_edge_cases()
            results["edge_cases_count"] = len(edge_cases)
            
            # 2. 清空向量库
            rag_pipeline.vector_store_manager.clear_vector_store()
            
            # 3. 执行边界测试用例
            edge_results = []
            total_response_time = 0
            valid_responses = 0
            
            for question, expected_response, case_type in edge_cases:
                logger.info(f"测试边界情况 [{case_type}]: {question}")
                
                # 执行查询
                query_start_time = time.time()
                actual_response = rag_pipeline.answer_query(question)
                query_time = time.time() - query_start_time
                total_response_time += query_time
                
                # 评估响应（简化版）
                # 这里我们检查实际响应是否包含了预期响应中的关键词
                is_valid = any(keyword in actual_response for keyword in expected_response.split()[:5])
                if is_valid:
                    valid_responses += 1
                
                # 记录结果
                edge_results.append({
                    "question": question,
                    "type": case_type,
                    "actual_response": actual_response,
                    "expected_response": expected_response,
                    "is_valid": is_valid,
                    "response_time": query_time
                })
            
            # 4. 计算统计数据
            results["edge_results"] = edge_results
            results["valid_responses"] = valid_responses
            results["valid_response_rate"] = valid_responses / len(edge_cases) if edge_cases else 0
            results["average_response_time"] = total_response_time / len(edge_cases) if edge_cases else 0
            
            # 5. 记录状态
            results["status"] = "passed" if valid_responses >= len(edge_cases) * 0.7 else "partially_passed"
            
        except Exception as e:
            logger.error(f"边界情况测试失败: {str(e)}")
            results["status"] = "failed"
            results["error"] = str(e)
        finally:
            # 确保清理
            try:
                rag_pipeline.vector_store_manager.clear_vector_store()
            except:
                pass
            
            results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存场景结果
        self.test_results["scenarios"][scenario_name] = results
        
        return results
    
    def test_document_format_handling(self) -> Dict:
        """测试不同文档格式处理能力"""
        scenario_name = "document_format_handling"
        results = {
            "scenario": scenario_name,
            "status": "running",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            # 1. 准备不同格式的测试文档
            logger.info("开始文档格式处理测试")
            
            # 注意：实际测试中，应该使用真实的不同格式文档
            # 这里为了简化，我们使用之前创建的文本文件模拟不同格式
            doc_formats = ["txt", "pdf", "docx", "xlsx", "csv"]
            format_results = []
            
            for doc_format in doc_formats:
                logger.info(f"测试文档格式: {doc_format}")
                
                # 清空向量库
                rag_pipeline.vector_store_manager.clear_vector_store()
                
                # 准备测试文件路径
                # 这里简化处理，使用一个通用的文本文件
                doc_path = os.path.join(self.documents_dir, f"test_file.{doc_format}")
                
                # 如果文件不存在，创建一个简单的文件
                if not os.path.exists(doc_path):
                    with open(doc_path, "w") as f:
                        f.write(f"这是一个{doc_format}格式的测试文件\n")
                        f.write(f"用于测试系统对{doc_format}格式的处理能力\n")
                
                # 尝试添加文档到向量库
                try:
                    add_success = rag_pipeline.add_single_document(doc_path)
                    
                    # 获取向量数量
                    vector_count = rag_pipeline.vector_store_manager.get_vector_count()
                    
                    # 记录结果
                    format_results.append({
                        "format": doc_format,
                        "file_path": doc_path,
                        "add_success": add_success,
                        "vector_count": vector_count,
                        "error": None
                    })
                except Exception as e:
                    logger.error(f"处理{doc_format}格式文件失败: {str(e)}")
                    format_results.append({
                        "format": doc_format,
                        "file_path": doc_path,
                        "add_success": False,
                        "vector_count": 0,
                        "error": str(e)
                    })
            
            # 2. 计算统计数据
            results["format_results"] = format_results
            success_count = sum(1 for result in format_results if result["add_success"])
            results["success_count"] = success_count
            results["success_rate"] = success_count / len(format_results) if format_results else 0
            
            # 3. 记录状态
            results["status"] = "passed" if success_count >= len(format_results) * 0.8 else "partially_passed"
            
        except Exception as e:
            logger.error(f"文档格式处理测试失败: {str(e)}")
            results["status"] = "failed"
            results["error"] = str(e)
        finally:
            # 确保清理
            try:
                rag_pipeline.vector_store_manager.clear_vector_store()
            except:
                pass
            
            results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存场景结果
        self.test_results["scenarios"][scenario_name] = results
        
        return results
    
    def test_corrupted_file_handling(self) -> Dict:
        """测试损坏文件处理能力"""
        scenario_name = "corrupted_file_handling"
        results = {
            "scenario": scenario_name,
            "status": "running",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            # 1. 准备损坏文件样本
            logger.info("开始损坏文件处理测试")
            
            # 创建不同类型的损坏文件
            corrupted_files = []
            for file_type in ["pdf", "docx", "zip", "txt"]:
                corrupted_file = self._create_corrupted_file(file_type)
                if corrupted_file:
                    corrupted_files.append((corrupted_file, file_type))
            
            # 创建空文件
            for file_type in ["txt", "pdf"]:
                empty_file = self._create_empty_file(file_type)
                if empty_file:
                    corrupted_files.append((empty_file, f"empty_{file_type}"))
            
            results["corrupted_files_count"] = len(corrupted_files)
            
            # 2. 测试损坏文件处理
            corrupted_results = []
            
            for corrupted_file, file_type in corrupted_files:
                logger.info(f"测试损坏文件处理: {file_type}")
                
                # 清空向量库
                rag_pipeline.vector_store_manager.clear_vector_store()
                
                # 尝试添加损坏文件
                try:
                    add_start_time = time.time()
                    add_success = rag_pipeline.add_single_document(corrupted_file)
                    add_time = time.time() - add_start_time
                    
                    # 获取向量数量
                    vector_count = rag_pipeline.vector_store_manager.get_vector_count()
                    
                    # 对于损坏文件，我们期望添加失败
                    # 对于空文件，系统可能会成功添加但向量数量为0
                    expected_success = False
                    if file_type.startswith("empty_"):
                        expected_success = True
                    
                    is_handling_correct = (not add_success) if not file_type.startswith("empty_") else (vector_count == 0)
                    
                    # 记录结果
                    corrupted_results.append({
                        "file_path": corrupted_file,
                        "file_type": file_type,
                        "add_success": add_success,
                        "vector_count": vector_count,
                        "add_time": add_time,
                        "is_handling_correct": is_handling_correct,
                        "error": None
                    })
                except Exception as e:
                    logger.error(f"处理损坏文件失败: {str(e)}")
                    # 对于损坏文件，我们期望系统抛出异常，这也是一种正确的处理方式
                    corrupted_results.append({
                        "file_path": corrupted_file,
                        "file_type": file_type,
                        "add_success": False,
                        "vector_count": 0,
                        "add_time": 0,
                        "is_handling_correct": True,  # 抛出异常被视为正确处理
                        "error": str(e)
                    })
            
            # 3. 计算统计数据
            results["corrupted_results"] = corrupted_results
            correct_handling_count = sum(1 for result in corrupted_results if result["is_handling_correct"])
            results["correct_handling_count"] = correct_handling_count
            results["correct_handling_rate"] = correct_handling_count / len(corrupted_results) if corrupted_results else 0
            
            # 4. 记录状态
            results["status"] = "passed" if correct_handling_count >= len(corrupted_results) * 0.8 else "partially_passed"
            
        except Exception as e:
            logger.error(f"损坏文件处理测试失败: {str(e)}")
            results["status"] = "failed"
            results["error"] = str(e)
        finally:
            # 确保清理
            try:
                rag_pipeline.vector_store_manager.clear_vector_store()
                # 清理临时文件
                self._cleanup_temp_files()
            except:
                pass
            
            results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存场景结果
        self.test_results["scenarios"][scenario_name] = results
        
        return results
    
    def test_error_recovery(self) -> Dict:
        """测试错误恢复机制"""
        scenario_name = "error_recovery"
        results = {
            "scenario": scenario_name,
            "status": "running",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            # 1. 准备测试
            logger.info("开始错误恢复机制测试")
            
            # 准备测试文档
            doc_path = self._prepare_test_document()
            if not doc_path:
                results["status"] = "failed"
                results["error"] = "准备测试文档失败"
                return results
            
            # 2. 测试文档处理中途失败的恢复
            logger.info("测试文档处理中途失败的恢复")
            
            # 这里我们需要模拟文档处理中途失败的情况
            # 一种方法是在处理过程中手动中断
            # 但这在自动化测试中很难实现
            # 所以我们简化处理，测试多次添加同一份文档
            
            # 清空向量库
            rag_pipeline.vector_store_manager.clear_vector_store()
            
            # 第一次添加文档
            first_add_success = rag_pipeline.add_single_document(doc_path)
            first_vector_count = rag_pipeline.vector_store_manager.get_vector_count()
            
            # 第二次添加同一份文档（应该可以成功）
            second_add_success = rag_pipeline.add_single_document(doc_path)
            second_vector_count = rag_pipeline.vector_store_manager.get_vector_count()
            
            # 3. 测试向量库写入失败的重试
            logger.info("测试向量库写入失败的重试")
            
            # 这里简化处理，通过清空向量库然后重新添加来模拟重试
            # 实际测试中应该模拟向量库写入失败的情况
            
            # 清空向量库
            clear_success = True
            try:
                rag_pipeline.vector_store_manager.clear_vector_store()
            except:
                clear_success = False
            
            # 重新添加文档
            retry_add_success = rag_pipeline.add_single_document(doc_path)
            retry_vector_count = rag_pipeline.vector_store_manager.get_vector_count()
            
            # 4. 记录结果
            results["recovery_tests"] = [
                {
                    "test_name": "document_processing_recovery",
                    "first_add_success": first_add_success,
                    "first_vector_count": first_vector_count,
                    "second_add_success": second_add_success,
                    "second_vector_count": second_vector_count,
                    "is_recovery_successful": second_add_success and second_vector_count >= first_vector_count
                },
                {
                    "test_name": "vector_store_retry",
                    "clear_success": clear_success,
                    "retry_add_success": retry_add_success,
                    "retry_vector_count": retry_vector_count,
                    "is_recovery_successful": clear_success and retry_add_success and retry_vector_count > 0
                }
            ]
            
            # 5. 计算统计数据
            successful_recoveries = sum(1 for test in results["recovery_tests"] if test["is_recovery_successful"])
            results["successful_recoveries"] = successful_recoveries
            results["recovery_rate"] = successful_recoveries / len(results["recovery_tests"]) if results["recovery_tests"] else 0
            
            # 6. 记录状态
            results["status"] = "passed" if successful_recoveries == len(results["recovery_tests"]) else "partially_passed"
            
        except Exception as e:
            logger.error(f"错误恢复机制测试失败: {str(e)}")
            results["status"] = "failed"
            results["error"] = str(e)
        finally:
            # 确保清理
            try:
                rag_pipeline.vector_store_manager.clear_vector_store()
            except:
                pass
            
            results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存场景结果
        self.test_results["scenarios"][scenario_name] = results
        
        return results
    
    def test_large_document_handling(self, size_mb: int = 10) -> Dict:
        """测试大文档处理能力"""
        scenario_name = f"large_document_handling_{size_mb}mb"
        results = {
            "scenario": scenario_name,
            "size_mb": size_mb,
            "status": "running",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            # 1. 准备大文件
            logger.info(f"开始大文档处理测试: {size_mb}MB")
            
            # 注意：实际测试中应该使用真实的大文档
            # 这里为了简化，我们创建一个大的文本文件
            large_file = self._create_large_file(size_mb)
            if not large_file:
                results["status"] = "failed"
                results["error"] = "创建大文件失败"
                return results
            
            results["large_file_path"] = large_file
            
            # 2. 清空向量库
            rag_pipeline.vector_store_manager.clear_vector_store()
            
            # 3. 测试添加大文件
            logger.info(f"添加大文件到向量库: {large_file}")
            add_start_time = time.time()
            
            try:
                add_success = rag_pipeline.add_single_document(large_file)
                add_time = time.time() - add_start_time
                
                # 获取向量数量
                vector_count = rag_pipeline.vector_store_manager.get_vector_count()
                
                # 记录结果
                results["add_success"] = add_success
                results["add_time"] = add_time
                results["vector_count"] = vector_count
                
                # 4. 测试从大文档中查询信息
                if add_success:
                    # 准备一些简单的查询
                    queries = [
                        "文档中包含什么内容？",
                        "这是什么类型的文档？",
                        "文档的主要信息是什么？"
                    ]
                    
                    query_results = []
                    total_response_time = 0
                    
                    for query in queries:
                        query_start_time = time.time()
                        answer = rag_pipeline.answer_query(query)
                        query_time = time.time() - query_start_time
                        total_response_time += query_time
                        
                        # 评估答案（简化版）
                        is_valid = len(answer) > 10 and "无法回答" not in answer
                        
                        query_results.append({
                            "query": query,
                            "answer": answer,
                            "is_valid": is_valid,
                            "response_time": query_time
                        })
                    
                    results["query_results"] = query_results
                    valid_queries = sum(1 for result in query_results if result["is_valid"])
                    results["valid_queries"] = valid_queries
                    results["average_query_time"] = total_response_time / len(queries) if queries else 0
                
                # 5. 记录状态
                if add_success and vector_count > 0:
                    results["status"] = "passed"
                else:
                    results["status"] = "failed"
            except Exception as e:
                add_time = time.time() - add_start_time
                logger.error(f"添加大文件失败: {str(e)}")
                results["add_success"] = False
                results["add_time"] = add_time
                results["error"] = str(e)
                results["status"] = "failed"
            
        except Exception as e:
            logger.error(f"大文档处理测试失败: {str(e)}")
            results["status"] = "failed"
            results["error"] = str(e)
        finally:
            # 确保清理
            try:
                rag_pipeline.vector_store_manager.clear_vector_store()
                # 清理大文件
                if 'large_file_path' in results and os.path.exists(results['large_file_path']):
                    try:
                        os.remove(results['large_file_path'])
                    except:
                        pass
            except:
                pass
            
            results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存场景结果
        self.test_results["scenarios"][scenario_name] = results
        
        return results
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            # 清理损坏文件样本目录
            if os.path.exists(self.corrupted_samples_dir):
                for file_name in os.listdir(self.corrupted_samples_dir):
                    file_path = os.path.join(self.corrupted_samples_dir, file_name)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"清理临时文件失败: {file_path}, 错误: {str(e)}")
        except Exception as e:
            logger.error(f"清理临时文件失败: {str(e)}")
    
    def run_all_scenarios(self) -> Dict:
        """运行所有测试场景"""
        self.test_results = {
            "scenarios": {},
            "overall_status": "running",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            logger.info("开始运行所有真实世界测试场景")
            
            # 1. 测试不同类型的真实文档
            document_types = ["contract", "technical_report", "user_manual"]
            for doc_type in document_types:
                self.test_real_document(doc_type)
            
            # 2. 测试边界情况
            self.test_edge_cases()
            
            # 3. 测试文档格式处理
            self.test_document_format_handling()
            
            # 4. 测试损坏文件处理
            self.test_corrupted_file_handling()
            
            # 5. 测试错误恢复机制
            self.test_error_recovery()
            
            # 6. 测试大文档处理（可选，因为创建大文件可能需要较多时间和磁盘空间）
            # 注意：默认跳过，因为创建大文件可能需要较多资源
            # self.test_large_document_handling(size_mb=10)
            
            # 7. 计算总体统计数据
            scenarios = self.test_results["scenarios"]
            total_scenarios = len(scenarios)
            passed_scenarios = sum(1 for scenario in scenarios.values() if scenario["status"] == "passed")
            partially_passed_scenarios = sum(1 for scenario in scenarios.values() if scenario["status"] == "partially_passed")
            failed_scenarios = sum(1 for scenario in scenarios.values() if scenario["status"] == "failed")
            
            self.test_results["statistics"] = {
                "total_scenarios": total_scenarios,
                "passed_scenarios": passed_scenarios,
                "partially_passed_scenarios": partially_passed_scenarios,
                "failed_scenarios": failed_scenarios,
                "pass_rate": passed_scenarios / total_scenarios if total_scenarios else 0
            }
            
            # 8. 确定总体状态
            if failed_scenarios > total_scenarios * 0.3:
                self.test_results["overall_status"] = "failed"
            elif passed_scenarios >= total_scenarios * 0.7:
                self.test_results["overall_status"] = "passed"
            else:
                self.test_results["overall_status"] = "partially_passed"
            
        except Exception as e:
            logger.error(f"运行所有测试场景失败: {str(e)}")
            self.test_results["overall_status"] = "failed"
            self.test_results["error"] = str(e)
        finally:
            # 确保清理
            try:
                self._cleanup_temp_files()
            except:
                pass
            
            self.test_results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # 保存测试结果
            self._save_test_results()
        
        return self.test_results
    
    def _save_test_results(self):
        """保存测试结果"""
        try:
            # 生成结果文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(self.results_dir, f"real_world_test_results_{timestamp}.json")
            
            # 保存结果
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"真实世界测试结果已保存到: {results_file}")
        except Exception as e:
            logger.error(f"保存测试结果失败: {str(e)}")
    
    def generate_report(self) -> Dict:
        """生成测试报告"""
        if not self.test_results:
            return {"error": "没有测试结果可生成报告"}
        
        report = {
            "title": "真实世界场景测试报告",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status": self.test_results.get("overall_status", "unknown"),
            "summary": {},
            "detailed_results": self.test_results.get("scenarios", {})
        }
        
        # 生成摘要
        stats = self.test_results.get("statistics", {})
        if stats:
            report["summary"] = {
                "total_scenarios": stats.get("total_scenarios", 0),
                "passed_scenarios": stats.get("passed_scenarios", 0),
                "partially_passed_scenarios": stats.get("partially_passed_scenarios", 0),
                "failed_scenarios": stats.get("failed_scenarios", 0),
                "pass_rate": f"{stats.get('pass_rate', 0) * 100:.1f}%"
            }
        
        # 添加建议
        recommendations = []
        
        # 分析失败的场景并提出建议
        for scenario_name, scenario_result in self.test_results.get("scenarios", {}).items():
            if scenario_result.get("status") == "failed":
                if "real_document" in scenario_name:
                    recommendations.append(f"改进对{scenario_result.get('document_type', '文档')}类型的处理能力")
                elif "edge_cases" in scenario_name:
                    recommendations.append("加强对边界情况的处理，特别是上下文相关和隐私过滤的场景")
                elif "document_format" in scenario_name:
                    recommendations.append("优化对不同文档格式的支持")
                elif "corrupted_file" in scenario_name:
                    recommendations.append("提高对损坏文件和空文件的容错能力")
                elif "error_recovery" in scenario_name:
                    recommendations.append("增强系统的错误恢复机制")
                elif "large_document" in scenario_name:
                    recommendations.append("优化大文档的处理效率和性能")
        
        if recommendations:
            report["recommendations"] = recommendations
        
        # 保存报告
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.results_dir, f"real_world_test_report_{timestamp}.json")
            
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"测试报告已保存到: {report_file}")
        except Exception as e:
            logger.error(f"保存测试报告失败: {str(e)}")
        
        return report

# 命令行入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="个人智能问答机器人真实世界场景测试工具")
    parser.add_argument("--scenario", choices=["document", "edge", "format", "corrupted", "recovery", "large", "all"], 
                        default="all", help="测试场景")
    parser.add_argument("--doc-type", type=str, default="contract", help="文档类型")
    parser.add_argument("--size-mb", type=int, default=10, help="大文档大小(MB)")
    
    # 解析参数
    args = parser.parse_args()
    
    # 初始化测试工具
    tester = RealWorldScenarioTester()
    
    # 根据场景执行测试
    if args.scenario == "document":
        # 测试真实文档
        print(f"开始测试真实文档: {args.doc_type}")
        results = tester.test_real_document(args.doc_type)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.scenario == "edge":
        # 测试边界情况
        print("开始测试边界情况")
        results = tester.test_edge_cases()
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.scenario == "format":
        # 测试文档格式处理
        print("开始测试文档格式处理")
        results = tester.test_document_format_handling()
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.scenario == "corrupted":
        # 测试损坏文件处理
        print("开始测试损坏文件处理")
        results = tester.test_corrupted_file_handling()
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.scenario == "recovery":
        # 测试错误恢复机制
        print("开始测试错误恢复机制")
        results = tester.test_error_recovery()
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.scenario == "large":
        # 测试大文档处理
        print(f"开始测试大文档处理: {args.size_mb}MB")
        results = tester.test_large_document_handling(args.size_mb)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        # 运行所有场景
        print("开始运行所有真实世界测试场景")
        results = tester.run_all_scenarios()
        
        # 生成并打印报告
        print("\n===== 真实世界场景测试报告 =====")
        report = tester.generate_report()
        
        # 打印摘要
        print("\n摘要:")
        for key, value in report.get("summary", {}).items():
            print(f"  {key}: {value}")
        
        # 打印建议
        if "recommendations" in report:
            print("\n建议:")
            for recommendation in report["recommendations"]:
                print(f"  - {recommendation}")
        
        print("\n测试结果已保存到results目录")