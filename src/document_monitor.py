import os
import time
import logging
import threading
import json
import shutil
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import traceback
from pathlib import Path
from src.rag_pipeline import rag_pipeline
from src.config import global_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentChangeHandler:
    """文档变更事件处理器基类"""
    
    def on_document_added(self, file_path, metadata=None):
        """当文档被添加时调用"""
        pass
    
    def on_document_updated(self, file_path, metadata=None):
        """当文档被更新时调用"""
        pass
    
    def on_document_deleted(self, file_path, metadata=None):
        """当文档被删除时调用"""
        pass
    
    def on_document_processing_failed(self, file_path, error=None):
        """当文档处理失败时调用"""
        pass
    
    def on_batch_complete(self, processed_files, failed_files):
        """当批次处理完成时调用"""
        pass


class DocumentMonitor:
    """增强的文档监控器，支持事件驱动、错误重试、版本控制和性能优化"""
    
    def __init__(self, max_workers=2, retry_attempts=3, retry_delay=5):
        """初始化文档监控器
        
        Args:
            max_workers: 并发处理的最大线程数
            retry_attempts: 处理失败时的重试次数
            retry_delay: 重试间隔（秒）
        """
        self.running = False
        self.monitor_thread = None
        self.last_checked = {}  # 记录每个文件的最后修改时间
        self.update_interval = global_config.DOCUMENT_UPDATE_INTERVAL
        # 初始化时强制使用空字典，避免旧的set类型数据结构问题
        self.vector_store_metadata = {}
        # 事件处理器列表
        self.event_handlers = []
        # 并发处理配置
        self.max_workers = max_workers
        self.thread_pool = None
        # 错误重试配置
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        # 处理队列
        self.processing_queue = Queue()
        # 版本控制
        self.version_history = {}
        # 处理统计
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'retries': 0
        }
        # 文档哈希值存储
        self.document_hashes = {}
        # 启动处理线程
        self.worker_threads = []
        # 然后尝试加载元数据
        try:
            loaded_metadata = self._load_vector_store_metadata()
            # 确保转换为字典格式
            if isinstance(loaded_metadata, set):
                # 转换set为字典
                for file_path in loaded_metadata:
                    self.vector_store_metadata[file_path] = {
                        'path': file_path,
                        'status': 'unknown',
                        'timestamp': datetime.now().isoformat(),
                        'version': 1,
                        'priority': 'medium'
                    }
            elif isinstance(loaded_metadata, list):
                # 从列表格式加载
                for item in loaded_metadata:
                    if isinstance(item, dict) and 'path' in item:
                        # 确保有必要的字段
                        item.setdefault('status', 'unknown')
                        item.setdefault('timestamp', datetime.now().isoformat())
                        item.setdefault('version', 1)
                        item.setdefault('priority', 'medium')
                        self.vector_store_metadata[item['path']] = item
                        # 从元数据加载文件哈希值
                        if 'hash' in item and item.get('status') == 'success':
                            self.document_hashes[item['path']] = item['hash']
                    elif isinstance(item, str):
                        # 旧格式的路径字符串
                        self.vector_store_metadata[item] = {
                            'path': item,
                            'status': 'unknown',
                            'timestamp': datetime.now().isoformat(),
                            'version': 1,
                            'priority': 'medium'
                        }
            elif isinstance(loaded_metadata, dict):
                # 确保每个元数据项都有必要的字段
                for path, metadata in loaded_metadata.items():
                    # 从元数据加载文件哈希值
                    if 'hash' in metadata and metadata.get('status') == 'success':
                        self.document_hashes[path] = metadata['hash']
                    if isinstance(metadata, dict):
                        metadata.setdefault('status', 'unknown')
                        metadata.setdefault('timestamp', datetime.now().isoformat())
                        metadata.setdefault('version', 1)
                        metadata.setdefault('priority', 'medium')
                self.vector_store_metadata = loaded_metadata
        except Exception as e:
            logger.error(f"加载元数据时出错，使用空字典: {str(e)}")
    
    def register_event_handler(self, handler):
        """注册文档变更事件处理器
        
        Args:
            handler: 实现DocumentChangeHandler接口的处理器实例
        """
        if handler not in self.event_handlers:
            self.event_handlers.append(handler)
            logger.info(f"已注册文档变更事件处理器: {handler.__class__.__name__}")
    
    def unregister_event_handler(self, handler):
        """注销文档变更事件处理器
        
        Args:
            handler: 要注销的处理器实例
        """
        if handler in self.event_handlers:
            self.event_handlers.remove(handler)
            logger.info(f"已注销文档变更事件处理器: {handler.__class__.__name__}")
    
    def _notify_event(self, event_name, *args, **kwargs):
        """通知所有事件处理器
        
        Args:
            event_name: 事件名称
            *args, **kwargs: 事件参数
        """
        for handler in self.event_handlers:
            try:
                method = getattr(handler, event_name, None)
                if method and callable(method):
                    method(*args, **kwargs)
            except Exception as e:
                logger.error(f"事件处理器执行出错: {handler.__class__.__name__}.{event_name}, 错误: {str(e)}")
    
    def _load_vector_store_metadata(self):
        """加载向量存储中的文档元数据"""
        # 尝试从文件加载元数据
        metadata_path = os.path.join(global_config.VECTOR_STORE_PATH, 'metadata.json')
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    # 处理不同格式的元数据
                    result = {}
                    
                    if isinstance(metadata, list):
                        # 检查是否是旧格式（简单的文件路径列表）
                        if len(metadata) > 0 and isinstance(metadata[0], str):
                            logger.info(f"检测到旧格式的元数据，包含 {len(metadata)} 个文件路径")
                            # 转换旧格式到新格式
                            for file_path in metadata:
                                result[file_path] = {
                                    'path': file_path,
                                    'status': 'unknown',  # 无法确定旧数据的状态，标记为unknown
                                    'timestamp': datetime.now().isoformat(),
                                    'version': 1,
                                    'priority': 'medium'
                                }
                        elif len(metadata) > 0 and isinstance(metadata[0], dict):
                            # 新格式（包含状态等信息的对象列表）
                            logger.info(f"加载新格式的元数据，包含 {len(metadata)} 个文件")
                            for item in metadata:
                                if 'path' in item:
                                    # 确保有必要的字段
                                    item.setdefault('status', 'unknown')
                                    item.setdefault('timestamp', datetime.now().isoformat())
                                    item.setdefault('version', 1)
                                    item.setdefault('priority', 'medium')
                                    result[item['path']] = item
                    elif isinstance(metadata, dict):
                        # 直接使用字典格式
                        logger.info(f"加载字典格式的元数据，包含 {len(metadata)} 个文件")
                        result = metadata
                    elif isinstance(metadata, set):
                        # 处理set类型
                        logger.info(f"加载set格式的元数据，包含 {len(metadata)} 个文件")
                        for file_path in metadata:
                            result[file_path] = {
                                'path': file_path,
                                'status': 'unknown',
                                'timestamp': datetime.now().isoformat(),
                                'version': 1,
                                'priority': 'medium'
                            }
                    
                    logger.info(f"成功加载向量存储元数据，共 {len(result)} 个文件")
                    return result
        except Exception as e:
            logger.error(f"加载向量存储元数据失败: {str(e)}")
        
        # 默认返回空字典
        return {}
    
    def _calculate_file_hash(self, file_path):
        """计算文件内容的哈希值，用于文件内容去重
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件内容的SHA-256哈希值
        """
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                # 分块读取文件内容，避免大文件占用过多内存
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希值失败: {file_path}, 错误: {str(e)}")
            return None
    
    def _save_vector_store_metadata(self):
        """保存向量存储元数据到文件"""
        metadata_path = os.path.join(global_config.VECTOR_STORE_PATH, 'metadata.json')
        try:
            # 确保向量存储目录存在
            os.makedirs(global_config.VECTOR_STORE_PATH, exist_ok=True)
            
            # 保存字典的值集合（包含状态等信息的对象列表）
            metadata_list = list(self.vector_store_metadata.values())
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=2)
            
            logger.info(f"成功保存向量存储元数据，共 {len(metadata_list)} 个文件，包含状态信息")
        except Exception as e:
            logger.error(f"保存向量存储元数据失败: {str(e)}")
    
    def _save_version_history(self):
        """保存版本历史记录"""
        history_path = os.path.join(global_config.VECTOR_STORE_PATH, 'version_history.json')
        try:
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.version_history, f, ensure_ascii=False, indent=2)
            logger.info(f"成功保存版本历史，共 {len(self.version_history)} 个文件的版本信息")
        except Exception as e:
            logger.error(f"保存版本历史失败: {str(e)}")
    
    def _load_version_history(self):
        """加载版本历史记录"""
        history_path = os.path.join(global_config.VECTOR_STORE_PATH, 'version_history.json')
        try:
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    self.version_history = json.load(f)
                logger.info(f"成功加载版本历史，共 {len(self.version_history)} 个文件的版本信息")
        except Exception as e:
            logger.error(f"加载版本历史失败: {str(e)}")
            self.version_history = {}
    
    def start_monitoring(self):
        """启动文档监控"""
        if self.running:
            logger.info("文档监控已经在运行中")
            return
        
        self.running = True
        # 创建线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True  # 设置为守护线程，主线程结束时自动退出
        self.monitor_thread.start()
        # 启动工作线程
        for _ in range(self.max_workers):
            worker_thread = threading.Thread(target=self._worker_loop)
            worker_thread.daemon = True
            worker_thread.start()
            self.worker_threads.append(worker_thread)
        # 加载版本历史
        self._load_version_history()
        
        logger.info(f"文档监控已启动，检查间隔: {self.update_interval}秒，最大工作线程: {self.max_workers}")
    
    def stop_monitoring(self):
        """停止文档监控"""
        if not self.running:
            logger.info("文档监控未运行")
            return
        
        self.running = False
        # 等待监控线程结束
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        # 关闭线程池
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        # 清空处理队列
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
                self.processing_queue.task_done()
            except:
                break
        # 保存元数据和版本历史
        self._save_vector_store_metadata()
        self._save_version_history()
        
        logger.info("文档监控已停止，资源已释放")
    
    def _worker_loop(self):
        """工作线程循环，处理队列中的任务"""
        while self.running:
            try:
                # 尝试从队列获取任务，超时时间为1秒
                task = self.processing_queue.get(timeout=1.0)
                if task:
                    task_type = task.get('type')
                    file_path = task.get('path')
                    priority = task.get('priority', 'medium')
                    
                    logger.debug(f"处理任务: {task_type} - {file_path} (优先级: {priority})" )
                    
                    try:
                        if task_type == 'new':
                            success = self._process_new_file_with_retry(file_path)
                        elif task_type == 'update':
                            success = self._process_updated_file_with_retry(file_path)
                        elif task_type == 'delete':
                            success = self._process_deleted_file(file_path)
                        else:
                            logger.warning(f"未知任务类型: {task_type}")
                            success = False
                    finally:
                        # 确保任务被标记为完成，避免队列阻塞
                        self.processing_queue.task_done()
            except Empty:
                # 队列超时是正常的预期行为，不记录为错误
                pass
            except Exception as e:
                logger.error(f"工作线程处理出错: {str(e)}")
                logger.debug(traceback.format_exc())
                # 确保任务被标记为完成，避免队列阻塞
                try:
                    if 'task' in locals():
                        self.processing_queue.task_done()
                except Exception:
                    pass
            # 短暂休眠，避免CPU占用过高
            time.sleep(0.01)
    
    def _monitor_loop(self):
        """监控循环，定时检查文档变化"""
        while self.running:
            try:
                self._check_documents()
            except Exception as e:
                logger.error(f"文档检查过程中出错: {str(e)}")
                logger.debug(traceback.format_exc())
            
            # 直接等待指定的时间间隔，提高效率
            if self.running:
                logger.debug(f"等待下一个检查周期: {self.update_interval}秒")
                time.sleep(self.update_interval)
    
    def _check_documents(self):
        """检查文档变化并更新向量存储"""
        documents_path = global_config.DOCUMENTS_PATH
        
        if not os.path.exists(documents_path):
            logger.warning(f"文档目录不存在: {documents_path}")
            return
        
        # 记录当前检查周期
        # 避免过于频繁的日志输出，只在需要时记录
        # logger.info(f"开始文档检查周期 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        
        # 收集目录下的所有文件
        files_to_check = []
        for root, _, files in os.walk(documents_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file_path)[1].lower()
                
                # 只检查支持的文件类型
                if file_extension in global_config.SUPPORTED_FILE_TYPES:
                    # 过滤临时文件和隐藏文件
                    if not self._should_ignore_file(file):
                        files_to_check.append(file_path)
        
        logger.info(f"本次检查共扫描 {len(files_to_check)} 个文件")
        
        # 检查文件变化
        updated_files = []
        new_files = []
        deleted_files = []
        
        # 检查新增和更新的文件
        for file_path in files_to_check:
            try:
                # 获取文件的最后修改时间和大小（作为更准确的变更检测）
                file_stats = os.stat(file_path)
                last_modified = file_stats.st_mtime
                file_size = file_stats.st_size
                file_signature = (last_modified, file_size)
                
                # 检查文件是否为空
                if file_size == 0:
                    logger.warning(f"跳过空文件: {file_path}")
                    continue
                
                # 检查文件是否是新文件或已更新
                if file_path not in self.last_checked:
                    # 分类文件并确定优先级
                    priority = self._determine_file_priority(file_path)
                    # 检查文件是否已经在向量存储中
                    if file_path not in self.vector_store_metadata:
                        new_files.append((file_path, priority))
                    else:
                        # 检查文件在元数据中的状态
                        metadata = self.vector_store_metadata.get(file_path, {})
                        # 如果状态不是success，或者状态是unknown，都需要重新处理
                        if metadata.get('status') != 'success':
                            logger.info(f"文件存在于向量存储中但状态非成功({metadata.get('status', 'unknown')})，重新处理: {file_path}")
                            new_files.append((file_path, priority))
                        else:
                            logger.info(f"文件已存在于向量存储中且处理成功，跳过: {file_path}")
                elif self.last_checked[file_path] != file_signature:
                    priority = self._determine_file_priority(file_path)
                    updated_files.append((file_path, priority))
                else:
                    # 文件未变化，检查其在向量存储中的状态
                    metadata = self.vector_store_metadata.get(file_path, {})
                    if metadata.get('status') == 'success':
                        logger.debug(f"文件未变化且处理成功，跳过: {file_path}")
                
                # 更新记录的最后修改时间和大小
                self.last_checked[file_path] = file_signature
            except Exception as e:
                logger.error(f"检查文件时出错: {file_path}, 错误: {str(e)}")
        
        # 检查删除的文件
        current_files_set = set(files_to_check)
        for stored_file in list(self.vector_store_metadata.keys()):
            if stored_file not in current_files_set:
                deleted_files.append(stored_file)
        
        # 按优先级排序文件（高优先级先处理）
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        new_files.sort(key=lambda x: priority_order.get(x[1], 1))
        updated_files.sort(key=lambda x: priority_order.get(x[1], 1))
        
        # 批量处理新增文件
        if new_files:
            logger.info(f"发现 {len(new_files)} 个新文件或需要重新处理的文件")
            # 提交到处理队列
            for file_path, priority in new_files:
                self.processing_queue.put({
                    'type': 'new',
                    'path': file_path,
                    'priority': priority
                })
        else:
            logger.info("没有发现需要重新处理的文件，所有文件状态正常")
        
        # 批量处理更新的文件
        if updated_files:
            logger.info(f"发现 {len(updated_files)} 个更新的文件")
            # 提交到处理队列
            for file_path, priority in updated_files:
                self.processing_queue.put({
                    'type': 'update',
                    'path': file_path,
                    'priority': priority
                })
        
        # 处理删除的文件
        if deleted_files:
            logger.info(f"发现 {len(deleted_files)} 个删除的文件")
            for file_path in deleted_files:
                self.processing_queue.put({
                    'type': 'delete',
                    'path': file_path,
                    'priority': 'medium'
                })
        
        # 记录文档检查完成
        logger.info(f"文档检查完成，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"当前向量存储元数据状态: 总文件数={len(self.vector_store_metadata)}, 成功={len([m for m in self.vector_store_metadata.values() if m.get('status') == 'success'])}, 失败={len([m for m in self.vector_store_metadata.values() if m.get('status') == 'failed'])}")
        logger.info(f"处理统计: 总处理={self.processing_stats['total_processed']}, 成功={self.processing_stats['successful']}, 失败={self.processing_stats['failed']}, 重试次数={self.processing_stats['retries']}")
        
        # 保存元数据变更
        self._save_vector_store_metadata()
    
    def _should_ignore_file(self, file_name):
        """判断是否应该忽略某个文件
        
        Args:
            file_name: 文件名
        
        Returns:
            bool: 是否忽略
        """
        # 忽略临时文件和隐藏文件
        if file_name.startswith('~$') or file_name.startswith('.'):
            return True
        # 忽略常见的临时文件扩展名
        temp_extensions = ['.tmp', '.temp', '.swp', '.bak']
        if any(file_name.lower().endswith(ext) for ext in temp_extensions):
            return True
        return False
    
    def _determine_file_priority(self, file_path):
        """确定文件处理优先级
        
        Args:
            file_path: 文件路径
        
        Returns:
            str: 优先级 ('high', 'medium', 'low')
        """
        # 根据文件名、大小或扩展名确定优先级
        file_name = os.path.basename(file_path).lower()
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # 紧急文档优先
        if any(keyword in file_name for keyword in ['urgent', '紧急', 'important', '重要']):
            return 'high'
        
        # 小文件优先处理
        try:
            file_size = os.path.getsize(file_path)
            if file_size < 1024 * 10:  # 10KB以下的文件
                return 'high'
        except:
            pass
        
        # 某些格式优先处理
        if file_ext in ['.txt', '.md']:
            return 'high'
        elif file_ext in ['.pdf', '.docx']:
            return 'medium'
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            # 图片文件可能需要OCR，处理较慢
            return 'low'
        
        return 'medium'
    
    def _process_new_file_with_retry(self, file_path):
        """带重试机制的新文件处理
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 处理是否成功
        """
        success = False
        retry_count = 0
        last_error = None
        
        while retry_count <= self.retry_attempts and not success:
            try:
                if retry_count > 0:
                    logger.info(f"重试处理新文件(第{retry_count}次): {file_path}")
                    time.sleep(self.retry_delay)
                
                success = self._process_new_file(file_path)
                if not success:
                    retry_count += 1
                    self.processing_stats['retries'] += 1
                    last_error = "处理失败"
                
            except Exception as e:
                retry_count += 1
                self.processing_stats['retries'] += 1
                last_error = str(e)
                logger.error(f"处理文件时发生异常(第{retry_count}次): {file_path}, 错误: {str(e)}")
                logger.debug(traceback.format_exc())
                time.sleep(self.retry_delay)
        
        if not success:
            logger.error(f"文件处理失败，已达到最大重试次数: {file_path}, 最后错误: {last_error}")
            self._notify_event('on_document_processing_failed', file_path, last_error)
        
        return success
    
    def _process_new_file(self, file_path):
        """处理新增的文件或需要重新处理的文件"""
        try:
            logger.info(f"开始处理新文件: {file_path}")
            
            # 计算文件哈希值
            file_hash = self._calculate_file_hash(file_path)
            if file_hash:
                # 检查文件哈希值是否已经存在于记录中（幂等处理）
                if file_hash in self.document_hashes.values():
                    # 查找已存在的相同哈希值的文件路径
                    existing_file_path = next((path for path, hash_val in self.document_hashes.items() if hash_val == file_hash), None)
                    if existing_file_path and existing_file_path != file_path:
                        logger.info(f"检测到内容相同的文件，跳过处理: {file_path} (与 {existing_file_path} 内容相同)")
                        # 直接将文件标记为成功状态，但不添加到向量数据库
                        metadata_item = {
                            'path': file_path,
                            'status': 'success',
                            'timestamp': datetime.now().isoformat(),
                            'version': 1,
                            'size': os.path.getsize(file_path),
                            'extension': os.path.splitext(os.path.basename(file_path))[1].lower(),
                            'priority': self._determine_file_priority(file_path),
                            'hash': file_hash,
                            'duplicate_of': existing_file_path
                        }
                        self.vector_store_metadata[file_path] = metadata_item
                        self.document_hashes[file_path] = file_hash
                        self._save_vector_store_metadata()
                        return True
            
            # 检查文件是否已经在向量存储中，但允许重新处理unknown状态的文件
            if file_path in self.vector_store_metadata:
                metadata = self.vector_store_metadata.get(file_path, {})
                if metadata.get('status') == 'success':
                    # 如果文件状态已成功，但哈希值不同，可能是文件被修改但路径没变
                    if file_hash and metadata.get('hash') != file_hash:
                        logger.info(f"文件路径相同但内容已变更，重新处理: {file_path}")
                    else:
                        logger.info(f"文件已存在于向量存储中且处理成功，跳过: {file_path}")
                        return True
                else:
                    logger.info(f"文件已存在于向量存储中，但状态为{metadata.get('status', 'unknown')}，重新处理: {file_path}")
            else:
                logger.info(f"添加新文件到向量存储: {file_path}")
            
            # 创建或更新版本历史
            if file_path not in self.version_history:
                self.version_history[file_path] = []
            
            # 获取文件信息
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower()
            file_size = os.path.getsize(file_path)
            
            # 记录版本信息
            version_info = {
                'version': len(self.version_history[file_path]) + 1,
                'timestamp': datetime.now().isoformat(),
                'action': 'add',
                'size': file_size,
                'path': file_path,
                'hash': file_hash
            }
            self.version_history[file_path].append(version_info)
            
            # 处理文件
            logger.info(f"调用rag_pipeline.add_single_document处理文件: {file_path}")
            success = rag_pipeline.add_single_document(file_path)
            logger.info(f"处理文件结果: {'成功' if success else '失败'}，文件: {file_path}")
            
            # 更新处理统计
            self.processing_stats['total_processed'] += 1
            if success:
                self.processing_stats['successful'] += 1
            else:
                self.processing_stats['failed'] += 1
            
            # 无论成功与否，都记录文件到元数据，防止重复处理
            metadata_item = {
                'path': file_path,
                'status': 'success' if success else 'failed',
                'timestamp': datetime.now().isoformat(),
                'version': version_info['version'],
                'size': file_size,
                'extension': file_ext,
                'priority': self._determine_file_priority(file_path),
                'hash': file_hash
            }
            self.vector_store_metadata[file_path] = metadata_item
            if success and file_hash:
                self.document_hashes[file_path] = file_hash
            self._save_vector_store_metadata()
            self._save_version_history()
            
            if success:
                logger.info(f"成功添加文件: {file_path}")
                # 备份到backup文件夹并删除原文件
                logger.info(f"开始备份文件: {file_path}")
                self._backup_and_remove_file(file_path)
                # 触发事件通知
                self._notify_event('on_document_added', file_path, metadata_item)
            else:
                logger.warning(f"添加文件失败，但已记录到元数据以防止重复处理: {file_path}")
                # 触发事件通知
                self._notify_event('on_document_processing_failed', file_path, "处理失败")
            
            return success
        except Exception as e:
            # 更新处理统计
            self.processing_stats['total_processed'] += 1
            self.processing_stats['failed'] += 1
            
            # 即使发生异常，也要记录文件到元数据
            try:
                # 计算文件哈希值
                file_hash = self._calculate_file_hash(file_path)
                metadata_item = {
                    'path': file_path,
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat(),
                    'version': 0,
                    'error': str(e),
                    'hash': file_hash
                }
                self.vector_store_metadata[file_path] = metadata_item
                self._save_vector_store_metadata()
                logger.warning(f"处理新文件时出错，但已记录到元数据以防止重复处理: {file_path}, 错误: {str(e)}")
                # 触发事件通知
                self._notify_event('on_document_processing_failed', file_path, str(e))
            except Exception as inner_e:
                logger.error(f"记录文件元数据失败: {file_path}, 错误: {str(inner_e)}")
            
            return False
                
    def _backup_and_remove_file(self, file_path):
        """备份文件到backup文件夹并删除原文件，支持版本控制和中文文件名"""
        try:
            # 创建backup文件夹（如果不存在）
            backup_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backup')
            os.makedirs(backup_dir, exist_ok=True)
            
            # 获取文件名和相对路径
            file_name = os.path.basename(file_path)
            # 确保正确处理中文文件名
            file_name = file_name  # Python 3 中默认使用UTF-8，这里保持原样即可
            
            # 计算相对于DOCUMENTS_PATH的路径
            try:
                relative_path = os.path.relpath(file_path, global_config.DOCUMENTS_PATH)
                is_in_documents_path = True
            except ValueError:
                # 如果文件不在DOCUMENTS_PATH中，使用绝对路径作为标识
                relative_path = file_name
                is_in_documents_path = False
                logger.warning(f"文件不在配置的DOCUMENTS_PATH中: {file_path}")
            
            # 创建备份文件路径，保留相对目录结构
            if is_in_documents_path and relative_path != file_name:  # 如果文件不在根目录
                backup_subdir = os.path.join(backup_dir, os.path.dirname(relative_path))
                os.makedirs(backup_subdir, exist_ok=True)
                backup_path = os.path.join(backup_subdir, file_name)
            else:
                backup_path = os.path.join(backup_dir, file_name)
            
            # 如果备份文件已存在，添加时间戳避免覆盖
            if os.path.exists(backup_path):
                base_name, ext = os.path.splitext(file_name)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = os.path.join(os.path.dirname(backup_path), f"{base_name}_{timestamp}{ext}")
            
            # 备份文件
            logger.info(f"正在备份文件: {file_path} -> {backup_path}")
            shutil.copy2(file_path, backup_path)
            logger.info(f"文件已成功备份到: {backup_path}")
            
            # 删除原文件
            if is_in_documents_path:  # 只删除DOCUMENTS_PATH中的文件
                os.remove(file_path)
                logger.info(f"原文件已成功删除: {file_path}")
                # 触发文件删除事件通知
                self._notify_event('on_document_backup_completed', file_path, {'backup_path': backup_path})
            else:
                logger.info(f"文件不在DOCUMENTS_PATH中，跳过删除: {file_path}")
        except Exception as e:
            logger.error(f"备份和删除文件失败: {file_path}, 错误: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def _process_updated_file_with_retry(self, file_path):
        """带重试机制的文件更新处理
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 处理是否成功
        """
        success = False
        retry_count = 0
        last_error = None
        
        while retry_count <= self.retry_attempts and not success:
            try:
                if retry_count > 0:
                    logger.info(f"重试处理更新文件(第{retry_count}次): {file_path}")
                    time.sleep(self.retry_delay)
                
                success = self._process_updated_file(file_path)
                if not success:
                    retry_count += 1
                    self.processing_stats['retries'] += 1
                    last_error = "更新失败"
                
            except Exception as e:
                retry_count += 1
                self.processing_stats['retries'] += 1
                last_error = str(e)
                logger.error(f"更新文件时发生异常(第{retry_count}次): {file_path}, 错误: {str(e)}")
                logger.debug(traceback.format_exc())
                time.sleep(self.retry_delay)
        
        if not success:
            logger.error(f"文件更新失败，已达到最大重试次数: {file_path}, 最后错误: {last_error}")
            self._notify_event('on_document_processing_failed', file_path, last_error)
        
        return success
    
    def _process_updated_file(self, file_path):
        """处理更新的文件，支持版本控制"""
        try:
            logger.info(f"更新文件在向量存储中的内容: {file_path}")
            
            # 计算文件哈希值
            file_hash = self._calculate_file_hash(file_path)
            if file_hash:
                # 检查文件哈希值是否已经存在于记录中（幂等处理）
                existing_metadata = self.vector_store_metadata.get(file_path, {})
                if existing_metadata.get('status') == 'success' and existing_metadata.get('hash') == file_hash:
                    logger.info(f"文件内容未变化，跳过更新处理: {file_path}")
                    return True
            
            # 获取当前版本信息
            current_version = 1
            if file_path in self.vector_store_metadata:
                current_version = self.vector_store_metadata[file_path].get('version', 1)
            
            # 记录版本历史
            if file_path not in self.version_history:
                self.version_history[file_path] = []
            
            # 获取文件信息
            file_size = os.path.getsize(file_path)
            version_info = {
                'version': current_version + 1,
                'timestamp': datetime.now().isoformat(),
                'action': 'update',
                'size': file_size,
                'path': file_path,
                'hash': file_hash
            }
            self.version_history[file_path].append(version_info)
            
            # 由于当前向量存储没有提供更新单个文档的功能，我们先删除旧文档再添加新文档
            # 注意：这是一个简化的实现，实际应用中可能需要更复杂的处理
            success = rag_pipeline.add_single_document(file_path)
            
            # 更新处理统计
            self.processing_stats['total_processed'] += 1
            if success:
                self.processing_stats['successful'] += 1
            else:
                self.processing_stats['failed'] += 1
            
            # 更新元数据中的状态信息
            if file_path in self.vector_store_metadata:
                self.vector_store_metadata[file_path]['status'] = 'success' if success else 'failed'
                self.vector_store_metadata[file_path]['timestamp'] = datetime.now().isoformat()
                self.vector_store_metadata[file_path]['version'] = version_info['version']
                self.vector_store_metadata[file_path]['size'] = file_size
                if file_hash:
                    self.vector_store_metadata[file_path]['hash'] = file_hash
                    self.document_hashes[file_path] = file_hash
                self._save_vector_store_metadata()
                self._save_version_history()
            
            if success:
                logger.info(f"成功更新文件: {file_path}，新版本: {version_info['version']}")
                # 备份到backup文件夹并删除原文件
                self._backup_and_remove_file(file_path)
                # 触发事件通知
                self._notify_event('on_document_updated', file_path, self.vector_store_metadata.get(file_path))
            else:
                logger.warning(f"更新文件失败: {file_path}")
                # 触发事件通知
                self._notify_event('on_document_processing_failed', file_path, "更新失败")
            
            return success
        except Exception as e:
            # 更新处理统计
            self.processing_stats['total_processed'] += 1
            self.processing_stats['failed'] += 1
            
            logger.error(f"处理更新文件时出错: {file_path}, 错误: {str(e)}")
            logger.debug(traceback.format_exc())
            # 触发事件通知
            self._notify_event('on_document_processing_failed', file_path, str(e))
            return False
    
    def _process_deleted_file(self, file_path):
        """处理删除的文件"""
        try:
            logger.info(f"处理删除的文件: {file_path}")
            
            # 记录版本历史
            if file_path in self.version_history:
                version_info = {
                    'version': len(self.version_history[file_path]) + 1,
                    'timestamp': datetime.now().isoformat(),
                    'action': 'delete',
                    'path': file_path
                }
                self.version_history[file_path].append(version_info)
                self._save_version_history()
            
            # 从元数据中删除文件
            if file_path in self.vector_store_metadata:
                metadata = self.vector_store_metadata.pop(file_path)
                self._save_vector_store_metadata()
                
                # 从last_checked中移除
                if file_path in self.last_checked:
                    del self.last_checked[file_path]
                
                logger.info(f"已从向量存储元数据中删除文件记录: {file_path}")
                # 触发事件通知
                self._notify_event('on_document_deleted', file_path, metadata)
                return True
            
            return False
        except Exception as e:
            logger.error(f"处理删除文件时出错: {file_path}, 错误: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def batch_process_files(self, file_paths, priority='medium'):
        """批量处理文件
        
        Args:
            file_paths: 文件路径列表
            priority: 优先级
            
        Returns:
            dict: 处理结果统计
        """
        logger.info(f"开始批量处理 {len(file_paths)} 个文件，优先级: {priority}")
        
        # 提交任务到队列
        for file_path in file_paths:
            self.processing_queue.put({
                'type': 'new',
                'path': file_path,
                'priority': priority
            })
        
        # 等待所有任务完成
        self.processing_queue.join()
        
        logger.info(f"批量处理完成")
        return {
            'total': len(file_paths),
            'processed': self.processing_stats['total_processed'],
            'successful': self.processing_stats['successful'],
            'failed': self.processing_stats['failed']
        }
    
    def get_file_version_history(self, file_path):
        """获取文件的版本历史
        
        Args:
            file_path: 文件路径
            
        Returns:
            list: 版本历史记录
        """
        return self.version_history.get(file_path, [])
    
    def get_processing_stats(self):
        """获取处理统计信息
        
        Returns:
            dict: 处理统计
        """
        return self.processing_stats.copy()
    
    def get_monitoring_status(self):
        """获取监控状态"""
        return {
            'running': self.running,
            'update_interval': self.update_interval,
            'last_checked_count': len(self.last_checked),
            'vector_store_document_count': rag_pipeline.get_vector_count(),
            'vector_store_metadata_count': len(self.vector_store_metadata),
            'processing_queue_size': self.processing_queue.qsize(),
            'active_workers': len(self.worker_threads),
            'processing_stats': self.processing_stats.copy(),
            'version_history_count': len(self.version_history)
        }
    
    def pause_monitoring(self):
        """暂停监控但保持处理队列"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("文档监控已暂停，处理队列仍在处理中")
    
    def resume_monitoring(self):
        """恢复监控"""
        if not self.running:
            self.start_monitoring()
        logger.info("文档监控已恢复")
    
    def clear_all_metadata(self):
        """清除所有元数据"""
        self.vector_store_metadata.clear()
        self.last_checked.clear()
        self.version_history.clear()
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'retries': 0
        }
        self._save_vector_store_metadata()
        self._save_version_history()
        logger.info("已清除所有元数据和历史记录")


class DefaultDocumentChangeHandler(DocumentChangeHandler):
    """默认的文档变更处理器实现"""
    
    def on_document_added(self, file_path, metadata=None):
        logger.info(f"事件: 文档已添加 - {file_path}, 版本: {metadata.get('version', 'N/A')}")
    
    def on_document_updated(self, file_path, metadata=None):
        logger.info(f"事件: 文档已更新 - {file_path}, 新版本: {metadata.get('version', 'N/A')}")
    
    def on_document_deleted(self, file_path, metadata=None):
        logger.info(f"事件: 文档已删除 - {file_path}")

    def on_document_processing_failed(self, file_path, error=None):
        logger.warning(f"事件: 文档处理失败 - {file_path}, 错误: {error}")

    def on_batch_complete(self, processed_files, failed_files):
        logger.info(f"事件: 批次处理完成 - 成功: {len(processed_files)}, 失败: {len(failed_files)}")
        
    def on_document_backup_completed(self, file_path, metadata=None):
        if metadata and 'backup_path' in metadata:
            logger.info(f"事件: 文档备份完成 - {file_path} -> {metadata['backup_path']}")
        else:
            logger.info(f"事件: 文档备份完成 - {file_path}")


# 创建文档监控器实例，配置更合理的默认参数
document_monitor = DocumentMonitor(
    max_workers=4,
    retry_attempts=3,
    retry_delay=5
)

# 注册默认事件处理器
default_handler = DefaultDocumentChangeHandler()
document_monitor.register_event_handler(default_handler)