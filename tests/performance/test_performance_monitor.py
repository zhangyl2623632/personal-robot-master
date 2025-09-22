import os
import sys
import time
import json
import logging
import argparse
import concurrent.futures
from typing import List, Dict, Tuple, Optional
import psutil
import numpy as np
from datetime import datetime, timedelta

# 导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.rag_pipeline import rag_pipeline
from src.config import global_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PerformanceMonitor")

class PerformanceMonitor:
    """性能监控工具，用于测试RAG系统的性能指标"""
    
    def __init__(self):
        """初始化性能监控工具"""
        # 测试配置
        self.test_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "test_data")
        self.performance_dir = os.path.join(self.test_data_dir, "performance")
        self.results_dir = os.path.join(self.test_data_dir, "results")
        self.performance_log_file = os.path.join(self.results_dir, "performance_log.jsonl")
        
        # 创建必要的目录
        self._setup_directories()
        
        # 性能测试配置
        self.target_response_time = 3.0  # 目标响应时间（秒）
        self.max_concurrent_users = 50   # 最大并发用户数
        self.default_chunk_size = 512    # 默认分块大小
        self.performance_metrics = {
            "response_times": [],
            "cpu_usage": [],
            "memory_usage": [],
            "vector_count": [],
            "throughput": []
        }
        
        # 资源监控状态
        self.monitoring = False
        self.monitor_thread = None
    
    def _setup_directories(self):
        """创建测试所需的目录"""
        required_dirs = [
            self.test_data_dir,
            self.performance_dir,
            self.results_dir
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"创建目录: {dir_path}")
    
    def _log_performance_metric(self, metrics: Dict):
        """记录性能指标到日志文件"""
        try:
            with open(self.performance_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"记录性能指标失败: {str(e)}")
    
    def _get_current_resource_usage(self) -> Dict:
        """获取当前系统资源使用情况"""
        try:
            # 获取CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 获取内存使用率
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            # 获取向量库中的向量数量
            vector_count = 0
            if hasattr(rag_pipeline, "vector_store_manager"):
                try:
                    vector_count = rag_pipeline.vector_store_manager.get_vector_count()
                except Exception as e:
                    logger.error(f"获取向量数量失败: {str(e)}")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "vector_count": vector_count
            }
        except Exception as e:
            logger.error(f"获取资源使用情况失败: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": 0,
                "memory_percent": 0,
                "vector_count": 0
            }
    
    def _generate_test_query(self, query_type: str = "normal") -> str:
        """生成测试查询"""
        query_types = {
            "normal": [
                "请概述本文档的主要内容",
                "系统的核心功能是什么",
                "请解释文档中提到的关键概念",
                "文档中提到了哪些重要的数据点",
                "如何使用系统进行文档分析"
            ],
            "complex": [
                "请详细比较文档中提到的三种不同方法的优缺点",
                "根据文档中的数据，预测未来三个月的发展趋势并说明理由",
                "总结文档的五个主要观点，并分别提供支持这些观点的证据",
                "文档中提到的技术挑战是什么，有哪些潜在的解决方案",
                "分析文档中的案例研究，提取关键成功因素和失败教训"
            ],
            "factual": [
                "文档中提到的截止日期是什么时候",
                "系统支持哪些文件格式",
                "文档中引用了多少个数据源",
                "项目的预算是多少",
                "团队成员的职责分工是什么"
            ]
        }
        
        queries = query_types.get(query_type, query_types["normal"])
        # 简单的轮询选择
        current_index = len(self.performance_metrics["response_times"]) % len(queries)
        return queries[current_index]
    
    def _execute_query(self, query: str, use_history: bool = False) -> Tuple[str, float, bool]:
        """执行查询并返回结果、响应时间和成功状态"""
        start_time = time.time()
        success = True
        answer = ""
        
        try:
            answer = rag_pipeline.answer_query(query, use_history=use_history)
        except Exception as e:
            logger.error(f"查询执行失败: {str(e)}")
            success = False
            answer = f"查询执行失败: {str(e)}"
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # 记录性能指标
        resource_usage = self._get_current_resource_usage()
        metric = {
            "query": query,
            "response_time": response_time,
            "success": success,
            **resource_usage
        }
        self._log_performance_metric(metric)
        
        # 更新内部指标
        self.performance_metrics["response_times"].append(response_time)
        self.performance_metrics["cpu_usage"].append(resource_usage["cpu_percent"])
        self.performance_metrics["memory_usage"].append(resource_usage["memory_percent"])
        self.performance_metrics["vector_count"].append(resource_usage["vector_count"])
        
        return answer, response_time, success
    
    def test_single_query_response_time(self, query: str = None) -> Dict:
        """测试单个查询的响应时间"""
        results = {
            "test_type": "single_query_response_time",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 如果没有提供查询，生成一个
            if query is None:
                query = self._generate_test_query()
            
            logger.info(f"测试单个查询响应时间: {query}")
            
            # 执行查询
            answer, response_time, success = self._execute_query(query)
            
            # 记录结果
            results["query"] = query
            results["response_time"] = response_time
            results["success"] = success
            results["answer"] = answer
            results["meets_target"] = response_time < self.target_response_time
            
            # 获取资源使用情况
            resource_usage = self._get_current_resource_usage()
            results.update(resource_usage)
            
        except Exception as e:
            logger.error(f"单查询响应时间测试失败: {str(e)}")
            results["error"] = str(e)
            results["success"] = False
        
        return results
    
    def test_concurrent_requests(self, num_queries: int = 10, concurrent_users: int = 5, query_type: str = "normal") -> Dict:
        """测试并发请求下的性能"""
        results = {
            "test_type": "concurrent_requests",
            "num_queries": num_queries,
            "concurrent_users": concurrent_users,
            "query_type": query_type,
            "timestamp": datetime.now().isoformat(),
            "successful_queries": 0,
            "failed_queries": 0,
            "response_times": [],
            "total_execution_time": 0
        }
        
        try:
            # 准备测试查询
            queries = [self._generate_test_query(query_type) for _ in range(num_queries)]
            
            logger.info(f"开始并发请求测试: {num_queries}个查询, {concurrent_users}个并发用户")
            
            # 记录开始时间
            start_time = time.time()
            
            # 使用线程池模拟并发用户
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                # 提交所有查询任务
                future_to_query = {executor.submit(self._execute_query, query): query for query in queries}
                
                # 收集结果
                for future in concurrent.futures.as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        answer, response_time, success = future.result()
                        results["response_times"].append(response_time)
                        if success:
                            results["successful_queries"] += 1
                        else:
                            results["failed_queries"] += 1
                    except Exception as e:
                        logger.error(f"并发查询执行失败: {query}, 错误: {str(e)}")
                        results["failed_queries"] += 1
            
            # 记录结束时间
            end_time = time.time()
            results["total_execution_time"] = end_time - start_time
            
            # 计算统计数据
            if results["response_times"]:
                results["average_response_time"] = sum(results["response_times"]) / len(results["response_times"])
                results["min_response_time"] = min(results["response_times"])
                results["max_response_time"] = max(results["response_times"])
                results["median_response_time"] = sorted(results["response_times"])[len(results["response_times"]) // 2]
                
                # 计算响应时间分布
                response_time_array = np.array(results["response_times"])
                results["p90_response_time"] = np.percentile(response_time_array, 90)
                results["p95_response_time"] = np.percentile(response_time_array, 95)
                results["p99_response_time"] = np.percentile(response_time_array, 99)
                
                # 计算满足目标响应时间的比例
                results["percentage_meeting_target"] = (
                    sum(1 for rt in results["response_times"] if rt < self.target_response_time) / 
                    len(results["response_times"]) * 100
                )
            
            # 计算吞吐量 (QPS - Queries Per Second)
            if results["total_execution_time"] > 0:
                results["throughput"] = results["successful_queries"] / results["total_execution_time"]
            
            # 获取最终资源使用情况
            resource_usage = self._get_current_resource_usage()
            results.update(resource_usage)
            
        except Exception as e:
            logger.error(f"并发请求测试失败: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def test_performance_scalability(self, max_concurrent: int = 50, step: int = 5, num_queries_per_user: int = 3) -> Dict:
        """测试系统的性能可扩展性（负载阶梯测试）"""
        results = {
            "test_type": "performance_scalability",
            "max_concurrent": max_concurrent,
            "step": step,
            "num_queries_per_user": num_queries_per_user,
            "timestamp": datetime.now().isoformat(),
            "scalability_results": {}
        }
        
        try:
            # 逐步增加并发用户数
            for users in range(step, max_concurrent + 1, step):
                logger.info(f"开始负载阶梯测试：{users} 并发用户")
                
                # 每个阶梯执行的查询数量
                num_queries = users * num_queries_per_user
                
                # 执行性能测试
                step_result = self.test_concurrent_requests(
                    num_queries=num_queries,
                    concurrent_users=users,
                    query_type="normal"
                )
                
                # 记录结果
                results["scalability_results"][f"{users}_users"] = step_result
            
            # 分析可扩展性
            self._analyze_scalability(results)
            
            # 获取最终资源使用情况
            resource_usage = self._get_current_resource_usage()
            results.update(resource_usage)
            
        except Exception as e:
            logger.error(f"可扩展性测试失败: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def _analyze_scalability(self, results: Dict):
        """分析系统的可扩展性"""
        scalability_results = results.get("scalability_results", {})
        
        # 提取关键指标用于分析
        user_counts = []
        throughputs = []
        avg_response_times = []
        resource_usages = []
        
        for user_count_str, result in scalability_results.items():
            if "error" in result:
                continue
            
            # 提取用户数
            try:
                user_count = int(user_count_str.split('_')[0])
            except:
                continue
            
            # 提取指标
            user_counts.append(user_count)
            
            throughput = result.get("throughput", 0)
            throughputs.append(throughput)
            
            avg_response_time = result.get("average_response_time", 0)
            avg_response_times.append(avg_response_time)
            
            # 计算平均资源使用率
            avg_cpu = result.get("cpu_percent", 0)
            avg_memory = result.get("memory_percent", 0)
            resource_usages.append((avg_cpu + avg_memory) / 2)
        
        # 添加分析结果
        if user_counts:
            results["analysis"] = {
                "user_counts": user_counts,
                "throughputs": throughputs,
                "avg_response_times": avg_response_times,
                "resource_usages": resource_usages,
                "max_throughput": max(throughputs) if throughputs else 0,
                "bottleneck_users": user_counts[np.argmax(throughputs)] if throughputs else 0
            }
    
    def _generate_test_document(self, size_mb: int, chunk_size: int = 512) -> str:
        """生成指定大小的测试文档"""
        try:
            # 创建性能测试文档目录
            perf_docs_dir = os.path.join(self.performance_dir, "docs")
            if not os.path.exists(perf_docs_dir):
                os.makedirs(perf_docs_dir)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            doc_path = os.path.join(perf_docs_dir, f"test_doc_{size_mb}mb_{timestamp}.txt")
            
            # 生成文档内容
            # 我们使用随机文本填充文档
            # 为了简化，这里使用重复的文本模式
            mb_size = 1024 * 1024  # 1MB = 1024KB
            chunk_text = "这是一段用于测试的文本。" * 100  # 生成一个基础文本块
            
            with open(doc_path, "w", encoding="utf-8") as f:
                current_size = 0
                while current_size < size_mb * mb_size:
                    # 写入文本块
                    f.write(chunk_text)
                    current_size += len(chunk_text.encode('utf-8'))
            
            logger.info(f"生成测试文档成功: {doc_path}, 大小约 {size_mb}MB")
            return doc_path
        except Exception as e:
            logger.error(f"生成测试文档失败: {str(e)}")
            return ""
    
    def test_vector_database_efficiency(self, num_chunks: int = 10000, chunk_size: int = 512) -> Dict:
        """测试向量数据库在不同规模下的检索效率"""
        results = {
            "test_type": "vector_database_efficiency",
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
            "timestamp": datetime.now().isoformat(),
            "efficiency_results": {}
        }
        
        try:
            # 确保向量库为空
            rag_pipeline.vector_store_manager.clear_vector_store()
            
            # 计算需要生成的文档大小
            # 假设每个token平均占2字节，每个chunk大约包含chunk_size/2个token
            # 则每个chunk大约占用 chunk_size 字节
            total_size_mb = (num_chunks * chunk_size) / (1024 * 1024)  # 转换为MB
            
            # 为了测试不同规模，我们分阶段添加文档
            stages = [0.1, 0.25, 0.5, 0.75, 1.0]  # 总规模的百分比
            current_chunks = 0
            
            for stage in stages:
                target_chunks = int(num_chunks * stage)
                chunks_to_add = target_chunks - current_chunks
                
                if chunks_to_add <= 0:
                    continue
                
                # 计算需要生成的文档大小
                stage_size_mb = (chunks_to_add * chunk_size) / (1024 * 1024)  # 转换为MB
                stage_size_mb = max(1, stage_size_mb)  # 至少生成1MB
                
                logger.info(f"测试向量数据库效率 - 阶段 {stage*100}%: 目标 {target_chunks} chunks")
                
                # 生成并添加测试文档
                doc_path = self._generate_test_document(stage_size_mb, chunk_size)
                if not doc_path:
                    logger.error(f"无法生成测试文档，跳过阶段 {stage*100}%")
                    continue
                
                # 添加文档到向量库
                start_time = time.time()
                success = rag_pipeline.add_single_document(doc_path)
                add_time = time.time() - start_time
                
                if not success:
                    logger.error(f"添加文档失败，跳过阶段 {stage*100}%")
                    continue
                
                # 更新当前chunk数（这里简化处理，实际应该从向量库获取）
                current_chunks = target_chunks
                
                # 执行检索测试
                retrieval_results = self._test_retrieval_efficiency(chunk_size)
                
                # 记录阶段结果
                results["efficiency_results"][f"{target_chunks}_chunks"] = {
                    "add_time": add_time,
                    "retrieval_results": retrieval_results,
                    "resource_usage": self._get_current_resource_usage()
                }
            
            # 分析向量数据库效率
            self._analyze_vector_database_efficiency(results)
            
        except Exception as e:
            logger.error(f"向量数据库效率测试失败: {str(e)}")
            results["error"] = str(e)
        finally:
            # 清理向量库
            try:
                rag_pipeline.vector_store_manager.clear_vector_store()
            except:
                pass
        
        return results
    
    def _test_retrieval_efficiency(self, chunk_size: int) -> Dict:
        """测试检索效率"""
        results = {
            "queries": [],
            "average_retrieval_time": 0
        }
        
        try:
            # 执行多个检索查询
            num_queries = 5
            total_retrieval_time = 0
            
            for i in range(num_queries):
                query = f"测试检索效率查询 {i+1}"
                
                # 只测量检索部分的时间（如果可能）
                start_time = time.time()
                
                # 尝试直接调用检索方法
                if hasattr(rag_pipeline.vector_store_manager, "search_similar_documents"):
                    try:
                        documents = rag_pipeline.vector_store_manager.search_similar_documents(query, top_k=5)
                        retrieval_time = time.time() - start_time
                        
                        # 记录结果
                        results["queries"].append({
                            "query": query,
                            "retrieval_time": retrieval_time,
                            "num_documents": len(documents)
                        })
                        
                        total_retrieval_time += retrieval_time
                    except Exception as e:
                        logger.error(f"直接检索测试失败: {str(e)}")
                        # 降级到完整查询
                        answer, response_time, success = self._execute_query(query)
                        results["queries"].append({
                            "query": query,
                            "retrieval_time": response_time,  # 这里包含了生成时间
                            "full_response_time": response_time,
                            "success": success
                        })
                        total_retrieval_time += response_time
                else:
                    # 直接执行完整查询
                    answer, response_time, success = self._execute_query(query)
                    results["queries"].append({
                        "query": query,
                        "retrieval_time": response_time,  # 这里包含了生成时间
                        "full_response_time": response_time,
                        "success": success
                    })
                    total_retrieval_time += response_time
            
            # 计算平均检索时间
            if results["queries"]:
                results["average_retrieval_time"] = total_retrieval_time / len(results["queries"])
            
        except Exception as e:
            logger.error(f"检索效率测试失败: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def _analyze_vector_database_efficiency(self, results: Dict):
        """分析向量数据库效率"""
        efficiency_results = results.get("efficiency_results", {})
        
        # 提取关键指标用于分析
        chunk_counts = []
        add_times = []
        retrieval_times = []
        
        for chunk_count_str, result in efficiency_results.items():
            if "error" in result:
                continue
            
            # 提取chunk数
            try:
                chunk_count = int(chunk_count_str.split('_')[0])
            except:
                continue
            
            # 提取指标
            chunk_counts.append(chunk_count)
            add_times.append(result.get("add_time", 0))
            
            retrieval_time = result.get("retrieval_results", {}).get("average_retrieval_time", 0)
            retrieval_times.append(retrieval_time)
        
        # 添加分析结果
        if chunk_counts:
            results["analysis"] = {
                "chunk_counts": chunk_counts,
                "add_times": add_times,
                "retrieval_times": retrieval_times,
                "linearity_score": self._calculate_linearity_score(chunk_counts, retrieval_times)
            }
    
    def _calculate_linearity_score(self, x_values: List[int], y_values: List[float]) -> float:
        """计算线性度得分，衡量指标随规模增长的线性程度"""
        if len(x_values) < 2 or len(y_values) < 2:
            return 0.0
        
        try:
            # 使用简单线性回归计算理论值
            x_array = np.array(x_values)
            y_array = np.array(y_values)
            
            # 计算线性回归系数
            coefficients = np.polyfit(x_array, y_array, 1)
            polynomial = np.poly1d(coefficients)
            
            # 计算理论值
            y_pred = polynomial(x_array)
            
            # 计算决定系数 R²
            ss_total = np.sum((y_array - np.mean(y_array))**2)
            ss_residual = np.sum((y_array - y_pred)** 2)
            r_squared = 1 - (ss_residual / ss_total)
            
            # 线性度得分在0-1之间，越高表示线性度越好
            return float(r_squared)
        except Exception as e:
            logger.error(f"计算线性度得分失败: {str(e)}")
            return 0.0
    
    def start_resource_monitoring(self, interval: int = 5):
        """开始持续资源监控"""
        import threading
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._resource_monitoring_thread, args=(interval,))
        self.monitor_thread.daemon = True  # 设置为守护线程，主线程结束时自动终止
        self.monitor_thread.start()
        
        logger.info(f"开始持续资源监控，间隔 {interval} 秒")
    
    def stop_resource_monitoring(self):
        """停止持续资源监控"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)  # 等待监控线程结束，最多等待5秒
            
            logger.info("停止持续资源监控")
    
    def _resource_monitoring_thread(self, interval: int):
        """资源监控线程函数"""
        while self.monitoring:
            try:
                # 获取资源使用情况
                resource_usage = self._get_current_resource_usage()
                
                # 记录资源使用情况
                self._log_performance_metric({
                    "monitoring": True,
                    **resource_usage
                })
                
                # 等待指定间隔
                time.sleep(interval)
            except Exception as e:
                logger.error(f"资源监控线程出错: {str(e)}")
                # 发生错误时，短暂暂停后继续
                time.sleep(1)
    
    def test_long_term_stability(self, duration_hours: float = 1, query_interval_seconds: int = 30) -> Dict:
        """测试系统的长期运行稳定性"""
        results = {
            "test_type": "long_term_stability",
            "duration_hours": duration_hours,
            "query_interval_seconds": query_interval_seconds,
            "timestamp": datetime.now().isoformat(),
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "memory_leak_detected": False,
            "performance_degradation_detected": False
        }
        
        try:
            # 计算测试结束时间
            end_time = datetime.now() + timedelta(hours=duration_hours)
            
            # 记录初始内存使用情况
            initial_memory = psutil.virtual_memory().used
            initial_response_times = []
            
            # 开始资源监控
            self.start_resource_monitoring(interval=60)  # 每分钟记录一次资源使用情况
            
            logger.info(f"开始长期稳定性测试: 持续 {duration_hours} 小时")
            
            # 持续执行测试直到达到指定时长
            while datetime.now() < end_time:
                # 执行查询
                query = self._generate_test_query()
                answer, response_time, success = self._execute_query(query)
                
                # 更新统计
                results["total_queries"] += 1
                if success:
                    results["successful_queries"] += 1
                    # 记录前5个查询的响应时间作为基准
                    if len(initial_response_times) < 5:
                        initial_response_times.append(response_time)
                else:
                    results["failed_queries"] += 1
                
                # 检查内存泄漏
                current_memory = psutil.virtual_memory().used
                memory_increase = (current_memory - initial_memory) / (1024 * 1024)  # 转换为MB
                
                # 如果内存增加超过100MB，标记为可能存在内存泄漏
                if memory_increase > 100:
                    results["memory_leak_detected"] = True
                    logger.warning(f"可能存在内存泄漏: 内存增加 {memory_increase:.2f} MB")
                
                # 检查性能衰减
                if initial_response_times and success:
                    avg_initial_response_time = sum(initial_response_times) / len(initial_response_times)
                    # 如果当前响应时间是初始平均响应时间的2倍以上，标记为性能衰减
                    if response_time > avg_initial_response_time * 2:
                        results["performance_degradation_detected"] = True
                        logger.warning(f"检测到性能衰减: 当前响应时间 {response_time:.2f}s, 初始平均 {avg_initial_response_time:.2f}s")
                
                # 等待指定间隔
                time.sleep(query_interval_seconds)
            
            # 停止资源监控
            self.stop_resource_monitoring()
            
            # 记录最终资源使用情况
            results["final_resource_usage"] = self._get_current_resource_usage()
            
        except Exception as e:
            logger.error(f"长期稳定性测试失败: {str(e)}")
            results["error"] = str(e)
            # 确保停止资源监控
            self.stop_resource_monitoring()
        
        return results
    
    def run_performance_test_suite(self, skip_long_term: bool = True) -> Dict:
        """运行完整的性能测试套件"""
        results = {
            "test_type": "performance_test_suite",
            "timestamp": datetime.now().isoformat(),
            "suite_results": {}
        }
        
        try:
            # 1. 测试单查询响应时间
            logger.info("开始单查询响应时间测试")
            single_query_results = self.test_single_query_response_time()
            results["suite_results"]["single_query"] = single_query_results
            
            # 2. 测试不同并发用户下的性能
            logger.info("开始多并发用户性能测试")
            concurrent_results = {}
            
            # 测试5、10、50并发用户
            for concurrent_users in [5, 10, 50]:
                result = self.test_concurrent_requests(
                    num_queries=concurrent_users * 3,  # 每个用户执行3个查询
                    concurrent_users=concurrent_users
                )
                concurrent_results[f"{concurrent_users}_users"] = result
            
            results["suite_results"]["concurrent_tests"] = concurrent_results
            
            # 3. 测试性能可扩展性
            logger.info("开始性能可扩展性测试")
            scalability_results = self.test_performance_scalability(
                max_concurrent=20,  # 简化测试，只到20并发
                step=5
            )
            results["suite_results"]["scalability"] = scalability_results
            
            # 4. 测试向量数据库效率
            logger.info("开始向量数据库效率测试")
            vector_efficiency_results = self.test_vector_database_efficiency(
                num_chunks=1000,  # 简化测试，只到1000个chunk
                chunk_size=self.default_chunk_size
            )
            results["suite_results"]["vector_efficiency"] = vector_efficiency_results
            
            # 5. 测试长期稳定性（可选）
            if not skip_long_term:
                logger.info("开始长期稳定性测试")
                # 注意：这个测试默认跳过，因为它可能需要很长时间
                # 实际使用时可以设置skip_long_term=False来运行
                stability_results = self.test_long_term_stability(
                    duration_hours=0.1,  # 简化测试，只运行6分钟
                    query_interval_seconds=30
                )
                results["suite_results"]["stability"] = stability_results
            else:
                logger.info("跳过长期稳定性测试")
            
            # 生成综合性能报告
            results = self._generate_comprehensive_performance_report(results)
            
            # 保存测试结果
            self._save_performance_results(results)
            
        except Exception as e:
            logger.error(f"运行性能测试套件失败: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def _generate_comprehensive_performance_report(self, results: Dict) -> Dict:
        """生成综合性能报告"""
        # 从各项测试结果中提取关键指标
        key_metrics = {
            "single_query_response_time": results.get("suite_results", {}).get("single_query", {}).get("response_time", 0),
            "average_response_time_5_users": 0,
            "average_response_time_10_users": 0,
            "average_response_time_50_users": 0,
            "max_throughput": 0,
            "vector_retrieval_time": 0,
            "meets_performance_target": False
        }
        
        # 提取并发测试结果
        concurrent_tests = results.get("suite_results", {}).get("concurrent_tests", {})
        for users_str, result in concurrent_tests.items():
            if users_str == "5_users":
                key_metrics["average_response_time_5_users"] = result.get("average_response_time", 0)
                key_metrics["max_throughput"] = max(key_metrics["max_throughput"], result.get("throughput", 0))
            elif users_str == "10_users":
                key_metrics["average_response_time_10_users"] = result.get("average_response_time", 0)
                key_metrics["max_throughput"] = max(key_metrics["max_throughput"], result.get("throughput", 0))
            elif users_str == "50_users":
                key_metrics["average_response_time_50_users"] = result.get("average_response_time", 0)
                key_metrics["max_throughput"] = max(key_metrics["max_throughput"], result.get("throughput", 0))
        
        # 提取向量数据库效率结果
        vector_efficiency = results.get("suite_results", {}).get("vector_efficiency", {})
        efficiency_results = vector_efficiency.get("efficiency_results", {})
        for chunk_count_str, result in efficiency_results.items():
            retrieval_time = result.get("retrieval_results", {}).get("average_retrieval_time", 0)
            key_metrics["vector_retrieval_time"] = max(key_metrics["vector_retrieval_time"], retrieval_time)
        
        # 判断是否满足性能目标
        # 简化判断：单查询响应时间<3秒，5用户下平均响应时间<5秒
        key_metrics["meets_performance_target"] = (
            key_metrics["single_query_response_time"] < self.target_response_time and 
            key_metrics["average_response_time_5_users"] < 5.0
        )
        
        # 添加综合评估
        results["key_performance_metrics"] = key_metrics
        
        # 添加性能建议
        recommendations = []
        if key_metrics["single_query_response_time"] >= self.target_response_time:
            recommendations.append("优化单查询响应时间，目标<3秒")
        if key_metrics["average_response_time_10_users"] >= 8.0:
            recommendations.append("增强系统在10并发用户下的性能")
        if key_metrics["max_throughput"] < 5.0:
            recommendations.append("提升系统吞吐量，当前QPS较低")
        
        if recommendations:
            results["performance_recommendations"] = recommendations
        else:
            results["performance_recommendations"] = ["系统性能良好，满足当前需求"]
        
        return results
    
    def _save_performance_results(self, results: Dict):
        """保存性能测试结果"""
        try:
            # 生成结果文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(self.results_dir, f"performance_results_{timestamp}.json")
            
            # 保存结果
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"性能测试结果已保存到: {results_file}")
        except Exception as e:
            logger.error(f"保存性能测试结果失败: {str(e)}")

# 命令行入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="个人智能问答机器人性能监控工具")
    parser.add_argument("--test-type", choices=["single", "concurrent", "scalability", "vector", "stability", "suite"], 
                        default="suite", help="测试类型")
    parser.add_argument("--num-queries", type=int, default=10, help="并发测试的查询数量")
    parser.add_argument("--concurrent-users", type=int, default=5, help="并发用户数")
    parser.add_argument("--max-concurrent", type=int, default=20, help="可扩展性测试的最大并发用户数")
    parser.add_argument("--num-chunks", type=int, default=1000, help="向量数据库测试的chunk数量")
    parser.add_argument("--duration", type=float, default=0.1, help="稳定性测试的持续时间（小时）")
    parser.add_argument("--skip-long-term", action="store_true", default=True, help="跳过长期稳定性测试")
    
    # 解析参数
    args = parser.parse_args()
    
    # 初始化性能监控工具
    monitor = PerformanceMonitor()
    
    # 根据测试类型执行测试
    if args.test_type == "single":
        # 测试单查询响应时间
        print("开始单查询响应时间测试")
        results = monitor.test_single_query_response_time()
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.test_type == "concurrent":
        # 测试并发请求
        print(f"开始并发请求测试: {args.num_queries}个查询, {args.concurrent_users}个并发用户")
        results = monitor.test_concurrent_requests(
            num_queries=args.num_queries,
            concurrent_users=args.concurrent_users
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.test_type == "scalability":
        # 测试可扩展性
        print(f"开始可扩展性测试: 最大{args.max_concurrent}并发用户")
        results = monitor.test_performance_scalability(max_concurrent=args.max_concurrent)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.test_type == "vector":
        # 测试向量数据库效率
        print(f"开始向量数据库效率测试: {args.num_chunks}个chunk")
        results = monitor.test_vector_database_efficiency(num_chunks=args.num_chunks)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.test_type == "stability":
        # 测试长期稳定性
        print(f"开始长期稳定性测试: 持续{args.duration}小时")
        results = monitor.test_long_term_stability(duration_hours=args.duration)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        # 运行完整性能测试套件
        print("开始运行完整性能测试套件")
        results = monitor.run_performance_test_suite(skip_long_term=args.skip_long_term)
        print("\n===== 性能测试报告 =====")
        print("关键性能指标:")
        for metric, value in results.get('key_performance_metrics', {}).items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")
        print("\n性能建议:")
        for recommendation in results.get('performance_recommendations', []):
            print(f"  - {recommendation}")
        print("\n性能测试结果已保存到results目录")