# 项目功能优化路线图

## 版本历史

- **v1.0.1**：当前版本，支持在线模型问答和离线嵌入/重排序模型

## 待优化功能清单

### 1. 离线大语言模型支持

#### 功能描述
添加对本地大语言模型的支持，使用户能够在无网络环境下进行文档问答。

#### 实现方案

##### 1. 创建离线LLM客户端类

在`src/llm_client.py`中添加新的客户端类：

```python
# src/llm_client.py
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    print("请安装transformers和torch以支持离线模型: pip install transformers torch")

class OfflineLLMClient(BaseLLMClient):
    def __init__(self, config=None):
        super().__init__(config)
        # 设置本地模型路径（从配置或环境变量获取）
        self.model_path = getattr(self.config, 'LOCAL_MODEL_PATH', 
                                 os.getenv('LOCAL_MODEL_PATH', 
                                          'models/offline/your-model'))
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            print(f"正在加载本地模型: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # 可根据需要选择不同的设备（'cuda' 或 'cpu'）
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            print("本地模型加载成功")
        except Exception as e:
            print(f"本地模型加载失败: {str(e)}")
    
    def _get_headers(self) -> Dict[str, str]:
        # 离线模型不需要headers
        return {}
    
    def _get_api_url(self) -> str:
        # 离线模型不需要API URL
        return "local"
    
    def _build_payload(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        # 构建模型输入文本
        text = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            text += f"{role}: {content}\n"
        text += "assistant:"
        return {"text": text, "stream": stream}
    
    def _call_api(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        try:
            payload = self._build_payload(messages)
            inputs = self.tokenizer(payload["text"], return_tensors="pt").to(self.model.device)
            
            # 生成回答
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True
            )
            
            # 解码生成的文本
            content = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 提取assistant的回答部分
            assistant_response = content.split("assistant:")[-1].strip()
            
            return {"choices": [{"message": {"content": assistant_response}}]}
        except Exception as e:
            print(f"离线模型推理失败: {str(e)}")
            return None
    
    def validate_api_key(self) -> bool:
        # 离线模型不需要验证API密钥
        return self.model is not None
```

##### 2. 更新LLMClientFactory类

在`src/llm_client.py`中更新工厂类：

```python
class LLMClientFactory:
    @staticmethod
    def create_client(config=None) -> BaseLLMClient:
        config = config or global_config
        provider = config.MODEL_PROVIDER.lower()

        client_map = {
            "deepseek": DeepSeekClient,
            "openai": OpenAIClient,
            "moonshot": MoonshotClient,
            "qwen_dashscope": QwenDashScopeClient,
            "offline": OfflineLLMClient,  # 添加离线模型支持
        }

        if provider in client_map:
            return client_map[provider](config)
        else:
            print(f"未知模型提供商: {provider}，默认使用 DeepSeekClient")
            return DeepSeekClient(config)
```

##### 3. 修改web_interface.py以添加离线模型选项

在`src/web_interface.py`中添加离线模型的支持：

```python
# 在web_interface.py中更新get_status函数
@app.route('/api/status')
def get_status():
    try:
        vector_count = rag_pipeline.get_vector_count()
        llm_status = llm_client.get_status()
        
        # 检查是否可以使用离线大语言模型
        has_local_model = os.path.exists(getattr(global_config, 'LOCAL_MODEL_PATH', 
                                               os.getenv('LOCAL_MODEL_PATH', '')))
        
        available_apis = {
            'deepseek': bool(global_config.DEEPSEEK_API_KEY),
            'openai': bool(global_config.OPENAI_API_KEY),
            'moonshot': bool(global_config.MOONSHOT_API_KEY),
            'qwen_dashscope': bool(global_config.DASHSCOPE_API_KEY or global_config.QWEN_API_KEY),
            'offline': has_local_model,  # 添加离线模型支持
        }
        
        return jsonify({
            'vector_count': vector_count,
            'llm_status': llm_status,
            'available_apis': available_apis,
            'config': {
                'model_provider': global_config.MODEL_PROVIDER,
                'model_name': global_config.MODEL_NAME,
                'temperature': global_config.TEMPERATURE,
                'max_tokens': global_config.MAX_TOKENS,
            },
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
```

##### 4. 配置要求

在`.env`文件中添加以下配置项：

```env
# 离线大语言模型配置
LOCAL_MODEL_PATH=models/offline/your-model-name
```

##### 5. 依赖安装

需要安装额外的Python库：

```bash
pip install transformers torch
```

### 2. 模型管理界面

#### 功能描述
添加Web界面支持管理和切换不同的模型，包括在线和离线模型。

#### 实现方案
- 扩展前端界面，添加模型选择下拉菜单
- 实现模型切换API接口
- 增加模型配置验证功能

### 3. 文档批量处理优化

#### 功能描述
优化大批量文档的处理速度和内存占用。

#### 实现方案
- 实现文档分批处理机制
- 添加处理进度显示
- 优化大文件处理逻辑

## 优先级排序

1. **离线大语言模型支持**（高优先级）
2. **模型管理界面**（中优先级）
3. **文档批量处理优化**（低优先级）

## 实现时间线

- **离线大语言模型支持**：预计在下个版本(v1.1.0)中实现
- **模型管理界面**：预计在下下个版本(v1.2.0)中实现
- **文档批量处理优化**：预计在后续版本中实现

## 注意事项

- 实现离线大语言模型需要考虑硬件性能要求，特别是内存和GPU显存
- 不同的本地模型可能需要不同的适配代码
- 离线模型的性能和响应速度可能低于在线模型