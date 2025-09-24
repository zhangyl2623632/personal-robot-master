// 等待DOM加载完成
window.addEventListener('DOMContentLoaded', function() {
    // 刷新系统状态
    refreshStatus();
    
    // 初始化文件上传功能
    initFileUpload();
    
    // 初始化模式选择器
    initModeSelector();
});

// 初始化模式选择器
function initModeSelector() {
    // 默认为Agent模式
    localStorage.setItem('useAgent', 'true');
    const currentModeElement = document.getElementById('current-mode');
    const modeSelectElement = document.getElementById('mode-select');
    
    if (currentModeElement) {
        currentModeElement.textContent = 'Agent模式';
    }
    
    if (modeSelectElement) {
        modeSelectElement.value = 'agent';
    }
}

// 切换选项卡
function switchTab(tabName) {
    // 隐藏所有面板
    const panels = document.querySelectorAll('.panel');
    panels.forEach(panel => panel.classList.remove('active'));
    
    // 移除所有选项卡的活动状态
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => button.classList.remove('active'));
    
    // 显示选中的面板和选项卡
    document.getElementById(`${tabName}-panel`).classList.add('active');
    document.getElementById(`${tabName}-tab`).classList.add('active');
}

// 发送单次提问
function sendAskQuery() {
    const input = document.getElementById('ask-input');
    const query = input.value.trim();
    
    if (!query) {
        alert('请输入您的问题');
        return;
    }
    
    // 清空输入框
    input.value = '';
    
    // 添加用户消息到界面
    const messagesContainer = document.getElementById('ask-messages');
    addMessageToUI(messagesContainer, 'user', query);
    
    // 添加加载中消息
    const loadingMessageElement = addMessageToUI(messagesContainer, 'loading', '正在思考...');
    
    // 获取当前模式
    const useAgent = localStorage.getItem('useAgent') === 'true';
    
    // 发送请求到服务器
    fetch('/api/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            query: query, 
            use_agent: useAgent 
        })
    })
    .then(response => response.json())
    .then(data => {
        // 移除加载中消息
        messagesContainer.removeChild(loadingMessageElement);
        
        if (data.error) {
            addMessageToUI(messagesContainer, 'bot', `错误: ${data.error}`);
        } else {
            addMessageToUI(messagesContainer, 'bot', data.answer);
        }
        
        // 滚动到底部
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    })
    .catch(error => {
        // 移除加载中消息
        messagesContainer.removeChild(loadingMessageElement);
        
        addMessageToUI(messagesContainer, 'bot', `请求失败: ${error.message}`);
        
        // 滚动到底部
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    });
}

// 发送对话查询
function sendChatQuery() {
    const input = document.getElementById('chat-input');
    const query = input.value.trim();
    
    if (!query) {
        alert('请输入您的问题');
        return;
    }
    
    // 清空输入框
    input.value = '';
    
    // 添加用户消息到界面
    const messagesContainer = document.getElementById('chat-messages');
    addMessageToUI(messagesContainer, 'user', query);
    
    // 添加加载中消息
    const loadingMessageElement = addMessageToUI(messagesContainer, 'loading', '正在思考...');
    
    // 获取当前模式
    const useAgent = localStorage.getItem('useAgent') === 'true';
    
    // 发送请求到服务器
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            query: query, 
            use_agent: useAgent 
        })
    })
    .then(response => response.json())
    .then(data => {
        // 移除加载中消息
        messagesContainer.removeChild(loadingMessageElement);
        
        if (data.error) {
            addMessageToUI(messagesContainer, 'bot', `错误: ${data.error}`);
        } else {
            addMessageToUI(messagesContainer, 'bot', data.answer);
        }
        
        // 滚动到底部
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    })
    .catch(error => {
        // 移除加载中消息
        messagesContainer.removeChild(loadingMessageElement);
        
        addMessageToUI(messagesContainer, 'bot', `请求失败: ${error.message}`);
        
        // 滚动到底部
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    });
}

// 清空对话历史
function clearChatHistory() {
    if (confirm('确定要清空对话历史吗？')) {
        // 发送请求到服务器
        fetch('/api/clear_history', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 清空界面上的消息
                document.getElementById('chat-messages').innerHTML = '';
                alert('对话历史已清空');
            } else {
                alert(`清空失败: ${data.error}`);
            }
        })
        .catch(error => {
            alert(`请求失败: ${error.message}`);
        });
    }
}

// 更新当前时间
function updateCurrentTime() {
    const now = new Date();
    const timeElement = document.getElementById('current-time');
    if (timeElement) {
        timeElement.textContent = now.toLocaleString('zh-CN');
    }
}

// 刷新系统状态
function refreshStatus() {
    // 更新当前时间
    updateCurrentTime();
    
    // 发送请求到服务器
    fetch('/api/status', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(`获取状态失败: ${data.error}`);
        } else {
            // 更新默认API密钥状态
            const defaultApiStatusElement = document.getElementById('default-api-key-status');
            if (defaultApiStatusElement) {
                const statusDot = defaultApiStatusElement.querySelector('.status-dot');
                const statusText = defaultApiStatusElement.querySelector('span:last-child');
                
                if (data.api_key_valid) {
                    statusDot.className = 'status-dot valid';
                    statusText.textContent = '有效';
                    statusText.style.color = '#28a745';
                } else {
                    statusDot.className = 'status-dot invalid';
                    statusText.textContent = '无效或未配置';
                    statusText.style.color = '#dc3545';
                }
            }
            
            // 更新问答模式显示
            const useAgent = localStorage.getItem('useAgent') === 'true';
            const currentModeElement = document.getElementById('current-mode');
            const modeSelectElement = document.getElementById('mode-select');
            
            if (currentModeElement) {
                currentModeElement.textContent = useAgent ? 'Agent模式' : 'RAG模式';
            }
            
            if (modeSelectElement) {
                modeSelectElement.value = useAgent ? 'agent' : 'rag';
            }
            
            // 更新向量数量
            const vectorCountElement = document.getElementById('vector-count');
            if (vectorCountElement) {
                vectorCountElement.textContent = data.vector_count;
                // 添加数字动画效果
                vectorCountElement.classList.add('number-animation');
                setTimeout(() => {
                    vectorCountElement.classList.remove('number-animation');
                }, 500);
            }
            
            // 显示系统版本
            if (data.version) {
                document.getElementById('system-version').textContent = 'v' + data.version;
            } else {
                document.getElementById('system-version').textContent = '未知';
            }
            
            // 显示元数据跟踪文档数
            if (data.document_monitor && data.document_monitor.vector_store_metadata_count !== undefined) {
                document.getElementById('vector-store-metadata-count').textContent = data.document_monitor.vector_store_metadata_count;
            } else {
                document.getElementById('vector-store-metadata-count').textContent = '加载失败';
            }
            
            // 从config字段获取配置信息
            if (data.config) {
                document.getElementById('documents-path').textContent = data.config.documents_path;
                document.getElementById('vector-store-path').textContent = data.config.vector_store_path;
                document.getElementById('model-name').textContent = data.config.model_name;
                document.getElementById('model-provider').textContent = data.config.model_provider;
                document.getElementById('document-update-interval').textContent = `${data.config.document_update_interval}秒`;
            }
            
            // 显示文档监控状态
            if (data.document_monitor) {
                const monitorStatusElement = document.getElementById('document-monitor-status');
                if (monitorStatusElement) {
                    const statusDot = monitorStatusElement.querySelector('.status-dot');
                    const statusText = monitorStatusElement.querySelector('span:last-child');
                    
                    if (data.document_monitor.running) {
                        statusDot.className = 'status-dot running';
                        statusText.textContent = '运行中';
                        statusText.style.color = '#28a745';
                    } else {
                        statusDot.className = 'status-dot stopped';
                        statusText.textContent = '已停止';
                        statusText.style.color = '#dc3545';
                    }
                }
                document.getElementById('monitored-file-count').textContent = data.document_monitor.last_checked_count;
            }
            
            // 填充模型选择下拉菜单
            if (data.supported_models && data.available_apis) {
                const modelSelect = document.getElementById('model-select');
                modelSelect.innerHTML = '<option value="">请选择模型</option>';
                
                // 当前使用的模型提供商
                const currentProvider = data.config?.model_provider || '';
                
                // 遍历所有支持的模型
                for (const [provider, modelInfo] of Object.entries(data.supported_models)) {
                    const isAvailable = data.available_apis[provider];
                    const option = document.createElement('option');
                    option.value = provider;
                    option.textContent = `${provider} - ${modelInfo.name}`;
                    
                    // 如果API密钥未配置，禁用选项
                    if (!isAvailable) {
                        option.disabled = true;
                        option.textContent += ' (API密钥未配置)';
                    }
                    
                    // 设置当前使用的模型为选中状态
                    if (provider === currentProvider) {
                        option.selected = true;
                    }
                    
                    modelSelect.appendChild(option);
                }
            }
            
            // 显示各模型API密钥状态
            if (data.available_apis) {
                let apiStatusHTML = '<div class="api-keys-status">';
                
                // 定义模型提供商的显示名称和颜色
                const providerDisplayInfo = {
                    'deepseek': { name: 'DeepSeek', color: '#667eea' },
                    'qwen': { name: '通义千问', color: '#28a745' },
                    'openai': { name: 'OpenAI', color: '#17a2b8' },
                    'moonshot': { name: '月之暗面', color: '#6f42c1' }
                };
                
                for (const [provider, isAvailable] of Object.entries(data.available_apis)) {
                    const displayInfo = providerDisplayInfo[provider] || { name: provider, color: '#6c757d' };
                    
                    apiStatusHTML += `
                        <div class="api-key-item">
                            <span class="api-key-name" style="color: ${displayInfo.color};">${displayInfo.name}</span>
                            <span class="api-key-status-badge api-key-status-${isAvailable ? 'configured' : 'not-configured'}">
                                ${isAvailable ? '已配置' : '未配置'}
                            </span>
                        </div>
                    `;
                }
                
                apiStatusHTML += '</div>';
                
                // 更新API密钥状态显示
                const apiKeyStatusElement = document.getElementById('api-key-status');
                if (apiKeyStatusElement) {
                    apiKeyStatusElement.innerHTML = apiStatusHTML;
                }
            }
        }
    })
    .catch(error => {
        console.error('刷新状态失败:', error);
        
        // 显示错误状态
        document.getElementById('system-version').textContent = '获取失败';
        document.getElementById('vector-count').textContent = '获取失败';
        document.getElementById('current-time').textContent = new Date().toLocaleString('zh-CN');
        
        // 设置API密钥状态为错误
        const defaultApiStatusElement = document.getElementById('default-api-key-status');
        if (defaultApiStatusElement) {
            const statusDot = defaultApiStatusElement.querySelector('.status-dot');
            const statusText = defaultApiStatusElement.querySelector('span:last-child');
            statusDot.className = 'status-dot invalid';
            statusText.textContent = '无法连接服务器';
            statusText.style.color = '#dc3545';
        }
    });
}

// 显示模型切换状态信息
function showModelSwitchStatus(message, isError = false) {
    const statusElement = document.getElementById('model-switch-status');
    statusElement.textContent = message;
    
    // 移除所有状态类
    statusElement.classList.remove('error', 'success', 'show');
    
    // 添加相应的状态类
    if (isError) {
        statusElement.classList.add('error');
    } else {
        statusElement.classList.add('success');
    }
    
    // 显示状态信息
    statusElement.classList.add('show');
    
    // 3秒后自动隐藏
    setTimeout(() => {
        statusElement.classList.remove('show');
    }, 3000);
}

// 切换模型
function switchModel() {
    const modelSelect = document.getElementById('model-select');
    const switchButton = document.getElementById('switch-model-btn');
    const selectedProvider = modelSelect.value;
    
    if (!selectedProvider) {
        showModelSwitchStatus('请选择一个模型', true);
        return;
    }
    
    // 显示加载状态
    switchButton.disabled = true;
    switchButton.textContent = '切换中...';
    modelSelect.disabled = true;
    
    console.log(`开始切换模型到: ${selectedProvider}`);
    
    // 发送请求到服务器切换模型
    fetch('/api/switch_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model_provider: selectedProvider })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP错误! 状态码: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            console.error('切换模型失败:', data.error);
            showModelSwitchStatus(`切换失败: ${data.error}`, true);
        } else {
            console.log('切换模型成功:', data);
            showModelSwitchStatus('模型切换成功，已刷新系统状态', false);
            
            // 刷新系统状态，显示新的模型信息
            refreshStatus();
        }
    })
    .catch(error => {
        console.error('模型切换请求失败:', error);
        showModelSwitchStatus(`请求失败: ${error.message}`, true);
    })
    .finally(() => {
        // 恢复按钮状态
        switchButton.disabled = false;
        switchButton.textContent = '切换模型';
        modelSelect.disabled = false;
    });
}

// 切换问答模式
function switchMode() {
    const modeSelect = document.getElementById('mode-select');
    const switchButton = document.getElementById('switch-mode-btn');
    const modeSwitchStatus = document.getElementById('mode-switch-status');
    
    // 禁用按钮，防止重复点击
    switchButton.disabled = true;
    switchButton.textContent = '切换中...';
    
    // 获取选择的模式
    const selectedMode = modeSelect.value;
    const useAgent = selectedMode === 'agent';
    
    // 保存到本地存储
    localStorage.setItem('useAgent', useAgent.toString());
    
    // 更新显示
    const currentModeElement = document.getElementById('current-mode');
    if (currentModeElement) {
        currentModeElement.textContent = useAgent ? 'Agent模式' : 'RAG模式';
    }
    
    // 显示切换状态
    showModeSwitchStatus(`已切换到${useAgent ? 'Agent' : 'RAG'}模式`, false);
    
    // 恢复按钮状态
    switchButton.disabled = false;
    switchButton.textContent = '切换模式';
}

// 显示模式切换状态
function showModeSwitchStatus(message, isError = false) {
    const modeSwitchStatus = document.getElementById('mode-switch-status');
    if (!modeSwitchStatus) return;
    
    // 清空之前的状态
    modeSwitchStatus.textContent = '';
    
    // 创建状态元素
    const statusElement = document.createElement('div');
    statusElement.classList.add('mode-switch-status-message');
    
    if (isError) {
        statusElement.classList.add('error');
        statusElement.textContent = message;
    } else {
        statusElement.classList.add('success');
        statusElement.textContent = message;
    }
    
    modeSwitchStatus.appendChild(statusElement);
    
    // 3秒后自动清除状态
    setTimeout(() => {
        if (modeSwitchStatus) {
            modeSwitchStatus.innerHTML = '';
        }
    }, 3000);
}

// 添加消息到界面
function addMessageToUI(container, type, content) {
    const messageElement = document.createElement('div');
    
    if (type === 'user') {
        messageElement.classList.add('message', 'user-message');
        messageElement.textContent = content;
    } else if (type === 'bot') {
        messageElement.classList.add('message', 'bot-message');
        // 使用marked库将Markdown转换为HTML
        messageElement.innerHTML = marked.parse(content);
    } else if (type === 'loading') {
        messageElement.classList.add('message', 'loading-message');
        messageElement.textContent = content;
    }
    
    container.appendChild(messageElement);
    
    // 滚动到底部
    container.scrollTop = container.scrollHeight;
    
    return messageElement;
}

// 为输入框添加回车键发送功能
document.addEventListener('DOMContentLoaded', function() {
    // 单次提问输入框
    const askInput = document.getElementById('ask-input');
    askInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendAskQuery();
        }
    });
    
    // 对话模式输入框
    const chatInput = document.getElementById('chat-input');
    chatInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendChatQuery();
        }
    });
});

// 初始化文件上传功能
function initFileUpload() {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    
    // 存储已选择的文件
    let selectedFiles = [];
    
    // 拖拽事件处理
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // 高亮拖放区域
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('drag-over');
    }
    
    function unhighlight() {
        dropArea.classList.remove('drag-over');
    }
    
    // 处理文件拖放
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        handleFiles(files);
    }
    
    // 处理文件选择
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });
    
    // 处理选择的文件
    function handleFiles(files) {
        if (files.length === 0) return;
        
        // 将File对象转换为数组并添加到selectedFiles
        Array.from(files).forEach(file => {
            // 检查文件类型是否支持
            const fileExtension = getFileExtension(file.name).toLowerCase();
            const supportedTypes = ['.txt', '.md', '.pdf', '.docx', '.doc', '.xlsx', '.xls', '.csv', '.jpg', '.jpeg', '.png', '.gif'];
            
            if (!supportedTypes.includes(fileExtension)) {
                appendToUploadLog(`不支持的文件类型: ${file.name}`);
                return;
            }
            
            // 避免重复文件
            if (!selectedFiles.some(f => f.name === file.name && f.size === file.size && f.lastModified === file.lastModified)) {
                selectedFiles.push(file);
            }
        });
        
        // 更新文件列表显示
        updateFileList();
    }
    
    // 获取文件扩展名
    function getFileExtension(filename) {
        return filename.slice(filename.lastIndexOf('.'));
    }
    
    // 更新文件列表显示
    function updateFileList() {
        if (selectedFiles.length === 0) {
            fileList.innerHTML = '<p>已选择的文件将显示在此处</p>';
            return;
        }
        
        fileList.innerHTML = '';
        
        selectedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.classList.add('file-item');
            
            // 文件图标
            const fileIcon = document.createElement('div');
            fileIcon.classList.add('file-icon');
            fileIcon.textContent = getFileIcon(file.name);
            
            // 文件信息
            const fileInfo = document.createElement('div');
            fileInfo.classList.add('file-info');
            fileInfo.appendChild(fileIcon);
            
            const fileDetails = document.createElement('div');
            
            const fileName = document.createElement('div');
            fileName.classList.add('file-name');
            fileName.textContent = file.name;
            
            const fileSize = document.createElement('div');
            fileSize.classList.add('file-size');
            fileSize.textContent = formatFileSize(file.size);
            
            fileDetails.appendChild(fileName);
            fileDetails.appendChild(fileSize);
            fileInfo.appendChild(fileDetails);
            
            // 移除按钮
            const removeBtn = document.createElement('button');
            removeBtn.classList.add('remove-file-btn');
            removeBtn.textContent = '移除';
            removeBtn.onclick = function() {
                selectedFiles.splice(index, 1);
                updateFileList();
            };
            
            fileItem.appendChild(fileInfo);
            fileItem.appendChild(removeBtn);
            fileList.appendChild(fileItem);
        });
    }
    
    // 获取文件图标
    function getFileIcon(filename) {
        const extension = getFileExtension(filename).toLowerCase();
        
        switch(extension) {
            case '.txt':
            case '.md':
                return '📄';
            case '.pdf':
                return '📑';
            case '.docx':
            case '.doc':
                return '📝';
            case '.xlsx':
            case '.xls':
            case '.csv':
                return '📊';
            case '.jpg':
            case '.jpeg':
            case '.png':
            case '.gif':
                return '🖼️';
            default:
                return '📁';
        }
    }
    
    // 格式化文件大小
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // 清空文件选择
    window.clearFileSelection = function() {
        selectedFiles = [];
        fileInput.value = '';
        updateFileList();
        appendToUploadLog('已清空文件选择');
    };
    
    // 上传文件
    window.uploadFiles = async function() {
        if (selectedFiles.length === 0) {
            appendToUploadLog('请先选择要上传的文件');
            return;
        }
        
        // 禁用上传按钮
        const uploadBtn = document.getElementById('upload-btn');
        const originalText = uploadBtn.textContent;
        uploadBtn.disabled = true;
        uploadBtn.textContent = '上传中...';
        
        appendToUploadLog(`开始上传 ${selectedFiles.length} 个文件`);
        
        let successCount = 0;
        let failCount = 0;
        
        // 逐个上传文件
        for (const file of selectedFiles) {
            try {
                appendToUploadLog(`正在上传: ${file.name}`);
                const result = await uploadSingleFile(file);
                
                if (result.success) {
                    successCount++;
                    appendToUploadLog(`✅ 上传成功: ${file.name}`);
                } else {
                    failCount++;
                    appendToUploadLog(`❌ 上传失败: ${file.name} - ${result.error || '未知错误'}`);
                }
            } catch (error) {
                failCount++;
                appendToUploadLog(`❌ 上传异常: ${file.name} - ${error.message}`);
            }
        }
        
        appendToUploadLog(`上传完成: 成功 ${successCount} 个, 失败 ${failCount} 个`);
        
        // 如果有成功上传的文件，刷新系统状态
        if (successCount > 0) {
            setTimeout(() => {
                refreshStatus();
                appendToUploadLog('已刷新系统状态');
            }, 1000);
        }
        
        // 恢复上传按钮
        uploadBtn.disabled = false;
        uploadBtn.textContent = originalText;
    };
    
    // 上传单个文件
    async function uploadSingleFile(file) {
        return new Promise((resolve, reject) => {
            const formData = new FormData();
            formData.append('file', file);
            
            fetch('/api/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP错误! 状态码: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                resolve(data);
            })
            .catch(error => {
                reject(error);
            });
        });
    }
    
    // 添加日志到上传状态区域
    function appendToUploadLog(message) {
        const uploadLog = document.getElementById('upload-log');
        const logEntry = document.createElement('div');
        
        // 添加时间戳
        const timestamp = new Date().toLocaleTimeString();
        logEntry.textContent = `[${timestamp}] ${message}`;
        
        uploadLog.appendChild(logEntry);
        
        // 滚动到底部
        uploadLog.scrollTop = uploadLog.scrollHeight;
    }
}