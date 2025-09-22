import os
import tempfile
import logging
import time
import threading
import functools
import platform
import flask
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from src.rag_pipeline import rag_pipeline
from src.config import global_config
from src.document_monitor import document_monitor
from src.version_manager import version_manager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# 配置应用密钥（用于安全相关功能）
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')

# 配置请求超时时间（秒）
app.config['REQUEST_TIMEOUT'] = 30

# 初始化请求限制器
limiter = Limiter(
    app,
    default_limits=["100 per day", "20 per hour"],
    storage_uri="memory://"
)

# 定义全局错误处理装饰器
def handle_exceptions(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        endpoint = request.endpoint
        try:
            # 设置请求超时
            with time_limit(app.config['REQUEST_TIMEOUT']):
                result = func(*args, **kwargs)
                # 记录请求处理时间
                process_time = time.time() - start_time
                if process_time > 5:  # 如果处理时间超过5秒，记录警告
                    logger.warning(f"Slow request: {endpoint} took {process_time:.2f} seconds")
                return result
        except RequestTimeout:
            error_msg = f"Request timed out after {app.config['REQUEST_TIMEOUT']} seconds"
            logger.error(f"{endpoint} - {error_msg}")
            return jsonify({'error': error_msg}), 408
        except FileNotFoundError:
            error_msg = "Required file or resource not found"
            logger.error(f"{endpoint} - {error_msg}")
            return jsonify({'error': error_msg}), 404
        except PermissionError:
            error_msg = "Permission denied when accessing resource"
            logger.error(f"{endpoint} - {error_msg}")
            return jsonify({'error': error_msg}), 403
        except ValueError as e:
            error_msg = f"Invalid data provided: {str(e)}"
            logger.error(f"{endpoint} - {error_msg}")
            return jsonify({'error': error_msg}), 400
        except Exception as e:
            error_msg = f"Internal server error: {str(e)}"
            logger.error(f"{endpoint} - {error_msg}", exc_info=True)
            return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500
    return wrapper

# 定义请求超时上下文管理器
class RequestTimeout(Exception):
    pass

class time_limit:
    def __init__(self, seconds):
        self.seconds = seconds
        self.timer = None
    
    def __enter__(self):
        self.timer = threading.Timer(self.seconds, self.handle_timeout)
        self.timer.daemon = True
        self.timer.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.cancel()
        return exc_type is None
    
    def handle_timeout(self):
        raise RequestTimeout()

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
@handle_exceptions
@limiter.limit("10 per minute")
def api_ask():
    """单次提问API，不使用对话历史"""
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({'error': '查询内容不能为空'}), 400

    # 使用RAG流水线回答问题，不使用对话历史
    answer = rag_pipeline.answer_query(query, use_history=False)
    
    if answer:
        return jsonify({'answer': answer})
    else:
        return jsonify({'error': '无法生成回答'}), 500

@app.route('/api/chat', methods=['POST'])
@handle_exceptions
@limiter.limit("20 per minute")
def api_chat():
    """聊天API，使用对话历史"""
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({'error': '查询内容不能为空'}), 400

    # 使用RAG流水线回答问题，使用对话历史
    answer = rag_pipeline.answer_query(query, use_history=True)
    
    if answer:
        return jsonify({'answer': answer})
    else:
        return jsonify({'error': '无法生成回答'}), 500

@app.route('/api/clear_history', methods=['POST'])
@handle_exceptions
def api_clear_history():
    """清空对话历史API"""
    rag_pipeline.clear_conversation_history()
    return jsonify({'success': True})

@app.route('/api/status', methods=['GET'])
@handle_exceptions
def api_status():
    """获取系统状态API"""
    # 检查核心组件健康状态
    health_status = {
        'document_loader': True,
        'vector_store': True,
        'llm_client': True,
        'rag_pipeline': True
    }
    
    # 尝试检查各个组件的可用性
    try:
        from src.document_loader import document_loader
        # 简单验证文档加载器
        if not hasattr(document_loader, 'load_document'):
            health_status['document_loader'] = False
    except Exception as e:
        logger.error(f"Document loader health check failed: {str(e)}")
        health_status['document_loader'] = False
    
    try:
        from src.vector_store import vector_store
        # 简单验证向量存储
        if not hasattr(vector_store, 'similarity_search'):
            health_status['vector_store'] = False
    except Exception as e:
        logger.error(f"Vector store health check failed: {str(e)}")
        health_status['vector_store'] = False
    
    try:
        from src.llm_client import llm_client
        # 简单验证LLM客户端
        if not hasattr(llm_client, 'generate_response'):
            health_status['llm_client'] = False
    except Exception as e:
        logger.error(f"LLM client health check failed: {str(e)}")
        health_status['llm_client'] = False
    
    # 系统整体状态
    overall_status = all(health_status.values())
    
    # 获取当前使用的模型配置
    current_model_provider = global_config.MODEL_PROVIDER
    current_model_name = global_config.MODEL_NAME
    current_model_url = global_config.MODEL_URL
    
    # 获取当前使用的API密钥是否配置
    current_api_key_configured = False
    if current_model_provider == 'deepseek':
        current_api_key_configured = bool(global_config.DEEPSEEK_API_KEY)
    elif current_model_provider == 'qwen':
        current_api_key_configured = bool(global_config.QWEN_API_KEY)
    elif current_model_provider == 'openai':
        current_api_key_configured = bool(global_config.OPENAI_API_KEY)
    elif current_model_provider == 'moonshot':
        current_api_key_configured = bool(global_config.MOONSHOT_API_KEY)
    
    status = {
        'api_key_valid': rag_pipeline.validate_api_key(),
        'vector_count': rag_pipeline.get_vector_count(),
        'version': version_manager.get_version(),
        'config': {
            'model_name': global_config.MODEL_NAME,
            'model_provider': global_config.MODEL_PROVIDER,
            'vector_store_path': global_config.VECTOR_STORE_PATH,
            'documents_path': global_config.DOCUMENTS_PATH,
            'document_update_interval': global_config.DOCUMENT_UPDATE_INTERVAL
        },
        'document_monitor': document_monitor.get_monitoring_status(),
        'supported_models': global_config.SUPPORTED_MODELS,
        'current_model': {
            'provider': current_model_provider,
            'name': current_model_name,
            'url': current_model_url,
            'api_key_configured': current_api_key_configured
        },
        'available_apis': {
            'deepseek': bool(global_config.DEEPSEEK_API_KEY),
            'qwen': bool(global_config.QWEN_API_KEY),
            'openai': bool(global_config.OPENAI_API_KEY),
            'moonshot': bool(global_config.MOONSHOT_API_KEY)
        },
        'health': {
            'status': 'healthy' if overall_status else 'unhealthy',
            'components': health_status
        },
        'system_info': {
            'python_version': platform.python_version() if 'platform' in globals() else 'unknown',
            'flask_version': flask.__version__ if 'flask' in globals() else 'unknown',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    return jsonify(status)

@app.route('/api/switch_model', methods=['POST'])
@handle_exceptions
def api_switch_model():
    """切换模型API"""
    data = request.json
    model_provider = data.get('model_provider')
    
    if not model_provider:
        return jsonify({'error': '请提供模型提供商'}), 400
    
    if model_provider not in global_config.SUPPORTED_MODELS:
        return jsonify({'error': f'不支持的模型提供商: {model_provider}'}), 400
    
    # 获取模型配置
    model_config = global_config.SUPPORTED_MODELS[model_provider]
    
    # 检查API密钥是否配置
    api_key_name = model_config['api_key_name']
    if not getattr(global_config, api_key_name):
        return jsonify({'error': f'{model_provider} 的API密钥未配置'}), 400
    
    # 更新全局配置
    global_config.MODEL_PROVIDER = model_provider
    global_config.MODEL_NAME = model_config['name']
    global_config.MODEL_URL = model_config['url']
    
    # 刷新LLM客户端
    from src.llm_client import llm_client
    llm_client.refresh_client()
    
    logger.info(f"已切换模型到: {model_provider} - {model_config['name']}")
    
    return jsonify({
        'success': True,
        'message': f'已成功切换到{model_provider}模型',
        'model_provider': model_provider,
        'model_name': model_config['name']
    })

@app.route('/api/version', methods=['GET'])
@handle_exceptions
def api_version():
    """获取当前系统版本API"""
    return jsonify({
        'version': version_manager.get_version()
    })

@app.route('/api/update_version', methods=['POST'])
@handle_exceptions
def api_update_version():
    """更新系统版本API（仅供开发使用）"""
    # 在实际生产环境中，这里应该添加适当的身份验证和权限检查
    data = request.json
    part = data.get('part', 'patch').lower()
    
    if part not in ['major', 'minor', 'patch']:
        return jsonify({'error': f'无效的版本部分: {part}'}), 400
    
    success = version_manager.increment_version(part)
    
    if success:
        return jsonify({
            'success': True,
            'new_version': version_manager.get_version()
        })
    else:
        return jsonify({'error': '版本更新失败'}), 500

@app.route('/api/upload', methods=['POST'])
@handle_exceptions
@limiter.limit("5 per minute")
def api_upload():
    """文件上传API，用于上传新文档并添加到向量存储"""
    # 检查是否有文件在请求中
    if 'file' not in request.files:
        return jsonify({'error': '未找到文件'}), 400
    
    file = request.files['file']
    
    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({'error': '文件名不能为空'}), 400
    
    # 检查文件类型是否支持
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in global_config.SUPPORTED_FILE_TYPES:
        return jsonify({'error': f'不支持的文件类型: {file_extension}'}), 400
    
    # 安全处理文件名
    filename = secure_filename(file.filename)
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        file.save(temp_file.name)
        temp_file_path = temp_file.name
    
    try:
        # 添加文档到向量存储
        success = rag_pipeline.add_single_document(temp_file_path)
        
        if success:
            # 将文件复制到documents目录（如果需要）
            target_path = os.path.join(global_config.DOCUMENTS_PATH, filename)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            # 如果文件已存在，添加时间戳避免覆盖
            if os.path.exists(target_path):
                base_name, ext = os.path.splitext(filename)
                timestamp = os.path.getmtime(temp_file_path)
                target_path = os.path.join(global_config.DOCUMENTS_PATH, f"{base_name}_{int(timestamp)}{ext}")
            
            # 复制文件
            with open(temp_file_path, 'rb') as src, open(target_path, 'wb') as dst:
                dst.write(src.read())
            
            return jsonify({
                'success': True,
                'message': f'文件上传成功: {filename}',
                'vector_count': rag_pipeline.get_vector_count()
            })
        else:
            return jsonify({'error': '文件处理失败'}), 500
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.route('/api/health', methods=['GET'])
def api_health():
    """简单的健康检查端点，用于监控系统基本可用性"""
    try:
        # 检查关键组件可用性
        from src.vector_store import vector_store
        from src.llm_client import llm_client
        
        # 简单验证
        components_ok = all([
            hasattr(vector_store, 'similarity_search'),
            hasattr(llm_client, 'generate_response')
        ])
        
        if components_ok:
            return jsonify({
                'status': 'UP',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'version': version_manager.get_version() if 'version_manager' in globals() else 'unknown'
            }), 200
        else:
            return jsonify({
                'status': 'DEGRADED',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }), 503
    except Exception:
        return jsonify({
            'status': 'DOWN',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }), 503

@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件"""
    return send_from_directory(app.static_folder, filename)

def process_documents_async():
    """异步处理文档"""
    try:
        # 检查并处理data文件夹中的文档
        print("正在检查data文件夹中的文档...")
        vector_count = rag_pipeline.get_vector_count()
        if vector_count == 0:
            print("向量存储为空，正在处理data文件夹中的文档...")
            success = rag_pipeline.process_documents()
            if success:
                print(f"成功处理文档，向量数量: {rag_pipeline.get_vector_count()}")
            else:
                print("文档处理失败，请检查日志获取更多信息")
        else:
            print(f"向量存储中已有 {vector_count} 个文档")
        
        # 启动文档监控
        print("启动文档监控功能...")
        document_monitor.start_monitoring()
    except Exception as e:
        print(f"异步处理文档或启动监控时出错: {str(e)}")

if __name__ == '__main__':
    # 创建templates和static目录（如果不存在）
    os.makedirs(os.path.join(os.path.dirname(__file__), '../templates'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), '../static'), exist_ok=True)
    
    # 导入threading模块，用于异步处理文档
    import threading
    
    # 在后台线程中处理文档
    document_thread = threading.Thread(target=process_documents_async)
    document_thread.daemon = True  # 设置为守护线程，主程序结束时自动结束
    document_thread.start()
    
    # 立即启动Flask应用
    print("启动Web服务器...")
    app.run(host='0.0.0.0', port=5000, debug=True)