import json
import uuid
import time
from datetime import datetime
import traceback
import re
from typing import Any, List, Generator


def format_time(timestamp: float) -> str:
    """格式化时间戳为可读的时间字符串"""
    try:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return '无效时间'


def generate_unique_id(prefix: str = '') -> str:
    """生成唯一标识符"""
    unique_id = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id


def safe_json_loads(json_str: str, default=None) -> Any:
    """安全地解析JSON字符串"""
    if default is None:
        default = {}
    try:
        if not json_str or not isinstance(json_str, str):
            return default
        return json.loads(json_str)
    except Exception:
        return default


def calculate_processing_time(start_time: float) -> float:
    """计算处理时间（秒）"""
    return round(time.time() - start_time, 3)


def sanitize_string(text: str) -> str:
    """清理字符串，移除特殊字符"""
    if not text:
        return ''
    # 保留中文、英文、数字和常见标点符号
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：,.!?;:"\'\s\-]', '', text)


def get_error_traceback() -> str:
    """获取当前错误的堆栈信息"""
    return traceback.format_exc()


def truncate_string(text: str, max_length: int = 1000) -> str:
    """截断字符串到指定长度"""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length] + '...'


def batch_process(items: list, batch_size: int = 50) -> list:
    """批量处理列表"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]