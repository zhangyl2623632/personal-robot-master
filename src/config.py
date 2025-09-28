import os
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    # DeepSeek API 配置
    DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
    # Qwen API 配置
    QWEN_API_KEY = os.environ.get('QWEN_API_KEY', '')
    # OpenAI API 配置
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    # Moonshot API 配置
    MOONSHOT_API_KEY = os.environ.get('MOONSHOT_API_KEY', '')
    
    # 当前使用的模型配置
    MODEL_PROVIDER = "deepseek"  # deepseek, openai, moonshot, qwen
    MODEL_NAME = "deepseek-chat"
    MODEL_URL = "https://api.deepseek.com/v1/chat/completions"
    TEMPERATURE = 0.1
    MAX_TOKENS = 2048
    TIMEOUT = 60
    
    # 所有支持的模型配置
    SUPPORTED_MODELS = {
        "deepseek": {
            "provider": "deepseek",
            "name": "deepseek-chat",
            "url": "https://api.deepseek.com/v1/chat/completions",
            "api_key_name": "DEEPSEEK_API_KEY"
        },
        "qwen": {
            "provider": "qwen",
            "name": "qwen-plus",
            "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            "api_key_name": "QWEN_API_KEY"  # 使用QWEN_API_KEY环境变量
        },
        "openai": {
            "provider": "openai",
            "name": "gpt-3.5-turbo",
            "url": "https://api.openai.com/v1/chat/completions",
            "api_key_name": "OPENAI_API_KEY"
        },
        "moonshot": {
            "provider": "moonshot",
            "name": "moonshot-v1-8k",
            "url": "https://api.moonshot.cn/v1/chat/completions",
            "api_key_name": "MOONSHOT_API_KEY"
        }
    }
    
    # 嵌入模型配置 - 使用中文优化的模型
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
    EMBEDDING_DIM = 768
    
    # 向量存储配置 - 更适合中文的分块大小
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    CHUNK_SIZE = 500  # 减小块大小，更适合中文
    CHUNK_OVERLAP = 100  # 增加重叠，保持上下文连续性
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.15"))  # 临时降低阈值，适应后备嵌入模型
    
    # 文档处理配置
    DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "./data")
    SUPPORTED_FILE_TYPES = [
        ".txt", ".md", ".pdf", ".docx", ".xlsx", ".csv",
        ".jpg", ".jpeg", ".png", ".gif"
    ]
    
    # 文档监控配置
    DOCUMENT_UPDATE_INTERVAL = int(os.getenv("DOCUMENT_UPDATE_INTERVAL", "300"))  # 默认为5分钟（300秒）
    
    # 优化的系统提示词
    SYSTEM_PROMPT = "你是一个智能问答助手，请严格根据以下提供的上下文回答问题。\n"
    SYSTEM_PROMPT += "如果上下文不包含答案，请直接回答：\"根据现有资料，无法回答该问题。\"\n"
    SYSTEM_PROMPT += "不要添加任何上下文之外的信息，不要凭空猜测。\n"
    SYSTEM_PROMPT += "请用简洁明了的语言回答问题，保持专业和客观。"
    
    # Agent 配置
    USE_AGENT_BY_DEFAULT = os.getenv("USE_AGENT_BY_DEFAULT", "true").lower() == "true"  # 默认使用Agent模式
    AGENT_MAX_HISTORY_LENGTH = int(os.getenv("AGENT_MAX_HISTORY_LENGTH", "10"))  # 对话历史最大长度
    AGENT_TOOL_CALL_TIMEOUT = int(os.getenv("AGENT_TOOL_CALL_TIMEOUT", "30"))  # 工具调用超时时间
    AGENT_MEMORY_PATH = os.getenv("AGENT_MEMORY_PATH", "./agent_memory")  # 代理记忆存储路径
    
    # 流式响应配置
    STREAMING_ENABLED = os.getenv("STREAMING_ENABLED", "true").lower() == "true"  # 默认启用流式响应
    STREAMING_CHUNK_SIZE = int(os.getenv("STREAMING_CHUNK_SIZE", "50"))  # 流式响应的块大小
    
    # 多索引管理配置
    DEFAULT_INDEX_NAME = os.getenv("DEFAULT_INDEX_NAME", "default")  # 默认索引名称
    INDEX_BACKUP_ENABLED = os.getenv("INDEX_BACKUP_ENABLED", "true").lower() == "true"  # 是否启用索引备份
    INDEX_BACKUP_PATH = os.getenv("INDEX_BACKUP_PATH", "./index_backups")  # 索引备份路径
    
    # 搜索缓存配置
    SEARCH_CACHE_ENABLED = os.getenv("SEARCH_CACHE_ENABLED", "true").lower() == "true"  # 是否启用搜索缓存
    SEARCH_CACHE_SIZE = int(os.getenv("SEARCH_CACHE_SIZE", "100"))  # 搜索缓存大小
    SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "3600"))  # 搜索缓存过期时间（秒）
    
    # 混合检索配置
    HYBRID_RETRIEVAL_ENABLED = os.getenv("HYBRID_RETRIEVAL_ENABLED", "true").lower() == "true"  # 是否启用混合检索
    VECTOR_SCORE_WEIGHT = float(os.getenv("VECTOR_SCORE_WEIGHT", "0.7"))  # 向量搜索权重
    KEYWORD_SCORE_WEIGHT = float(os.getenv("KEYWORD_SCORE_WEIGHT", "0.3"))  # 关键词搜索权重

# 创建配置实例
global_config = Config()

# 添加测试所需的方法
def _reset(self):
    """重置配置到初始状态"""
    # 创建一个新的Config实例
    new_config = Config()
    
    # 清空当前实例的所有属性
    for attr in list(self.__dict__.keys()):
        if not attr.startswith('__'):
            delattr(self, attr)
    
    # 复制新实例的所有属性
    for attr in dir(new_config):
        if not attr.startswith('__') and not callable(getattr(new_config, attr)):
            setattr(self, attr, getattr(new_config, attr))

# 将方法绑定到Config类
Config._reset = _reset