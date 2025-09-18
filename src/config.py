import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    # DeepSeek API 配置
    DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
    # Qwen API 配置
    # 注意：通义千问模型现在使用DASHSCOPE_API_KEY
    QWEN_API_KEY = os.environ.get('QWEN_API_KEY', '')
    DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', '')
    # OpenAI API 配置
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    # Moonshot API 配置
    MOONSHOT_API_KEY = os.environ.get('MOONSHOT_API_KEY', '')
    
    # 当前使用的模型配置
    MODEL_PROVIDER = "qwen_dashscope"  # deepseek, openai, moonshot, qwen_dashscope
    MODEL_NAME = "qwen-plus"
    MODEL_URL = ""  # DashScope库封装了API URL
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
        "qwen_dashscope": {
            "provider": "qwen_dashscope",
            "name": "qwen-plus",
            "url": "",  # DashScope库封装了API URL
            "api_key_name": "DASHSCOPE_API_KEY"  # 通义千问模型现在使用DASHSCOPE_API_KEY
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

# 创建配置实例
global_config = Config()