import os
import tempfile
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from src.rag_pipeline import rag_pipeline
from src.config import global_config
from src.document_monitor import document_monitor
from src.version_manager import version_manager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='../templates', static_folder='../static')

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def api_ask():
    """单次提问API，不使用对话历史"""
    try:
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
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """聊天API，使用对话历史"""
    try:
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
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_history', methods=['POST'])
def api_clear_history():
    """清空对话历史API"""
    try:
        rag_pipeline.clear_conversation_history()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    """获取系统状态API"""
    try:
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
            'available_apis': {
                'deepseek': bool(global_config.DEEPSEEK_API_KEY),
                'qwen_dashscope': bool(global_config.DASHSCOPE_API_KEY),
                'openai': bool(global_config.OPENAI_API_KEY),
                'moonshot': bool(global_config.MOONSHOT_API_KEY)
            }
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/switch_model', methods=['POST'])
def api_switch_model():
    """切换模型API"""
    try:
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
    except Exception as e:
        logger.error(f"切换模型失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/version', methods=['GET'])
def api_version():
    """获取当前系统版本API"""
    try:
        return jsonify({
            'version': version_manager.get_version()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_version', methods=['POST'])
def api_update_version():
    """更新系统版本API（仅供开发使用）"""
    try:
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
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """文件上传API，用于上传新文档并添加到向量存储"""
    try:
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
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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