import os
import time
import logging
import threading
import json
import os
import shutil
from datetime import datetime
from src.rag_pipeline import rag_pipeline
from src.config import global_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentMonitor:
    """文档监控器，定时检查文档变化并自动更新向量存储"""
    
    def __init__(self):
        """初始化文档监控器"""
        self.running = False
        self.monitor_thread = None
        self.last_checked = {}  # 记录每个文件的最后修改时间
        self.update_interval = global_config.DOCUMENT_UPDATE_INTERVAL
        # 初始化时强制使用空字典，避免旧的set类型数据结构问题
        self.vector_store_metadata = {}
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
                        'timestamp': datetime.now().isoformat()
                    }
            elif isinstance(loaded_metadata, dict):
                self.vector_store_metadata = loaded_metadata
        except Exception as e:
            logger.error(f"加载元数据时出错，使用空字典: {str(e)}")
    
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
                                    'timestamp': datetime.now().isoformat()
                                }
                        elif len(metadata) > 0 and isinstance(metadata[0], dict):
                            # 新格式（包含状态等信息的对象列表）
                            logger.info(f"加载新格式的元数据，包含 {len(metadata)} 个文件")
                            for item in metadata:
                                if 'path' in item:
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
                                'timestamp': datetime.now().isoformat()
                            }
                    
                    logger.info(f"成功加载向量存储元数据，共 {len(result)} 个文件")
                    return result
        except Exception as e:
            logger.error(f"加载向量存储元数据失败: {str(e)}")
        
        # 默认返回空字典
        return {}
    
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
    
    def start_monitoring(self):
        """启动文档监控"""
        if self.running:
            logger.info("文档监控已经在运行中")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True  # 设置为守护线程，主线程结束时自动退出
        self.monitor_thread.start()
        logger.info(f"文档监控已启动，检查间隔: {self.update_interval}秒")
    
    def stop_monitoring(self):
        """停止文档监控"""
        if not self.running:
            logger.info("文档监控未运行")
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)  # 等待监控线程结束，最多等待5秒
        logger.info("文档监控已停止")
    
    def _monitor_loop(self):
        """监控循环，定时检查文档变化"""
        while self.running:
            try:
                self._check_documents()
            except Exception as e:
                logger.error(f"文档检查过程中出错: {str(e)}")
            
            # 等待指定的时间间隔
            for _ in range(self.update_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def _check_documents(self):
        """检查文档变化并更新向量存储"""
        documents_path = global_config.DOCUMENTS_PATH
        
        if not os.path.exists(documents_path):
            logger.warning(f"文档目录不存在: {documents_path}")
            return
        
        # 记录当前检查周期
        logger.info(f"开始文档检查周期 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        
        # 收集目录下的所有文件
        files_to_check = []
        for root, _, files in os.walk(documents_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file_path)[1].lower()
                
                # 只检查支持的文件类型
                if file_extension in global_config.SUPPORTED_FILE_TYPES:
                    files_to_check.append(file_path)
        
        logger.info(f"本次检查共扫描 {len(files_to_check)} 个文件")
        
        # 检查文件变化
        updated_files = []
        new_files = []
        
        for file_path in files_to_check:
            try:
                # 获取文件的最后修改时间
                last_modified = os.path.getmtime(file_path)
                
                # 检查文件是否是新文件或已更新
                if file_path not in self.last_checked:
                    # 检查文件是否已经在向量存储中
                    if file_path not in self.vector_store_metadata:
                        new_files.append(file_path)
                    else:
                        # 检查文件在元数据中的状态
                        metadata = self.vector_store_metadata.get(file_path, {})
                        # 如果状态不是success，或者状态是unknown，都需要重新处理
                        if metadata.get('status') != 'success':
                            logger.info(f"文件存在于向量存储中但状态非成功({metadata.get('status', 'unknown')})，重新处理: {file_path}")
                            new_files.append(file_path)
                        else:
                            logger.info(f"文件已存在于向量存储中且处理成功，跳过: {file_path}")
                elif last_modified > self.last_checked[file_path]:
                    updated_files.append(file_path)
                else:
                    # 文件未变化，检查其在向量存储中的状态
                    metadata = self.vector_store_metadata.get(file_path, {})
                    if metadata.get('status') == 'success':
                        logger.debug(f"文件未变化且处理成功，跳过: {file_path}")
                
                # 更新记录的最后修改时间
                self.last_checked[file_path] = last_modified
            except Exception as e:
                logger.error(f"检查文件时出错: {file_path}, 错误: {str(e)}")
        
        # 处理新增文件
        if new_files:
            logger.info(f"发现 {len(new_files)} 个新文件或需要重新处理的文件")
            for file_path in new_files:
                self._process_new_file(file_path)
        else:
            logger.info("没有发现需要重新处理的文件，所有文件状态正常")
        
        # 处理更新的文件
        if updated_files:
            logger.info(f"发现 {len(updated_files)} 个更新的文件")
            for file_path in updated_files:
                self._process_updated_file(file_path)
        
        # 记录文档检查完成
        logger.info(f"文档检查完成，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"当前向量存储元数据状态: 总文件数={len(self.vector_store_metadata)}, 成功={len([m for m in self.vector_store_metadata.values() if m.get('status') == 'success'])}, 失败={len([m for m in self.vector_store_metadata.values() if m.get('status') == 'failed'])}")
        
        # 保存元数据变更
        self._save_vector_store_metadata()
    
    def _process_new_file(self, file_path):
        """处理新增的文件或需要重新处理的文件"""
        try:
            logger.info(f"开始处理新文件: {file_path}")
            # 检查文件是否已经在向量存储中，但允许重新处理unknown状态的文件
            if file_path in self.vector_store_metadata:
                metadata = self.vector_store_metadata.get(file_path, {})
                if metadata.get('status') == 'success':
                    logger.info(f"文件已存在于向量存储中且处理成功，跳过: {file_path}")
                    return
                else:
                    logger.info(f"文件已存在于向量存储中，但状态为{metadata.get('status', 'unknown')}，重新处理: {file_path}")
            else:
                logger.info(f"添加新文件到向量存储: {file_path}")
            
            # 处理文件
            logger.info(f"调用rag_pipeline.add_single_document处理文件: {file_path}")
            success = rag_pipeline.add_single_document(file_path)
            logger.info(f"处理文件结果: {'成功' if success else '失败'}，文件: {file_path}")
            
            # 无论成功与否，都记录文件到元数据，防止重复处理
            # 并设置正确的状态信息
            metadata_item = {
                'path': file_path,
                'status': 'success' if success else 'failed',
                'timestamp': datetime.now().isoformat()
            }
            self.vector_store_metadata[file_path] = metadata_item
            self._save_vector_store_metadata()
            
            if success:
                logger.info(f"成功添加文件: {file_path}")
                # 备份到backup文件夹并删除原文件
                logger.info(f"开始备份文件: {file_path}")
                self._backup_and_remove_file(file_path)
            else:
                logger.warning(f"添加文件失败，但已记录到元数据以防止重复处理: {file_path}")
        except Exception as e:
            # 即使发生异常，也要记录文件到元数据
            try:
                metadata_item = {
                    'path': file_path,
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat()
                }
                self.vector_store_metadata[file_path] = metadata_item
                self._save_vector_store_metadata()
                logger.warning(f"处理新文件时出错，但已记录到元数据以防止重复处理: {file_path}, 错误: {str(e)}")
            except Exception as inner_e:
                logger.error(f"记录文件元数据失败: {file_path}, 错误: {str(inner_e)}")
                
    def _backup_and_remove_file(self, file_path):
        """备份文件到backup文件夹并删除原文件"""
        try:
            # 创建backup文件夹（如果不存在）
            backup_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backup')
            os.makedirs(backup_dir, exist_ok=True)
            
            # 获取文件名和相对路径
            file_name = os.path.basename(file_path)
            relative_path = os.path.relpath(file_path, global_config.DOCUMENTS_PATH)
            
            # 创建备份文件路径，保留相对目录结构
            if relative_path != file_name:  # 如果文件不在根目录
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
            shutil.copy2(file_path, backup_path)
            logger.info(f"文件已备份到: {backup_path}")
            
            # 删除原文件
            os.remove(file_path)
            logger.info(f"原文件已删除: {file_path}")
        except Exception as e:
            logger.error(f"备份和删除文件失败: {file_path}, 错误: {str(e)}")
    
    def _process_updated_file(self, file_path):
        """处理更新的文件"""
        try:
            logger.info(f"更新文件在向量存储中的内容: {file_path}")
            # 由于当前向量存储没有提供更新单个文档的功能，我们先删除旧文档再添加新文档
            # 注意：这是一个简化的实现，实际应用中可能需要更复杂的处理
            success = rag_pipeline.add_single_document(file_path)
            
            # 更新元数据中的状态信息
            if file_path in self.vector_store_metadata:
                self.vector_store_metadata[file_path]['status'] = 'success' if success else 'failed'
                self.vector_store_metadata[file_path]['timestamp'] = datetime.now().isoformat()
                self._save_vector_store_metadata()
            
            if success:
                logger.info(f"成功更新文件: {file_path}")
                # 备份到backup文件夹并删除原文件
                self._backup_and_remove_file(file_path)
            else:
                logger.warning(f"更新文件失败: {file_path}")
        except Exception as e:
            logger.error(f"处理更新文件时出错: {file_path}, 错误: {str(e)}")
    
    def get_monitoring_status(self):
        """获取监控状态"""
        return {
            'running': self.running,
            'update_interval': self.update_interval,
            'last_checked_count': len(self.last_checked),
            'vector_store_document_count': rag_pipeline.get_vector_count(),
            'vector_store_metadata_count': len(self.vector_store_metadata)
        }

# 创建文档监控器实例
document_monitor = DocumentMonitor()