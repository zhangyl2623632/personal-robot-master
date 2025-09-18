# 离线模型使用指南

## 概述
本目录用于存放离线模型，以便在没有网络连接的环境中运行个人智能问答机器人。

## 需要下载的模型

### 1. BAAI/bge-large-zh-v1.5（嵌入模型）
- **下载链接**: https://huggingface.co/BAAI/bge-large-zh-v1.5
- **用途**: 用于将文本转换为向量表示，是向量检索的核心组件
- **放置路径**: `d:\workSpace-trae\personal-robot-master\models\offline\bge-large-zh-v1.5`

### 2. BAAI/bge-reranker-large（重排序模型）
- **下载链接**: https://huggingface.co/BAAI/bge-reranker-large
- **用途**: 用于对检索结果进行重排序，提高相关性
- **放置路径**: `d:\workSpace-trae\personal-robot-master\models\offline\bge-reranker-large`

## 下载方法

### 方法1: 使用huggingface-cli（推荐）
在有网络的环境中，使用以下命令下载模型：

```bash
# 安装huggingface-cli（如果尚未安装）
pip install huggingface_hub

# 下载嵌入模型
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir d:\workSpace-trae\personal-robot-master\models\offline\bge-large-zh-v1.5 --local-dir-use-symlinks False

# 下载重排序模型
huggingface-cli download BAAI/bge-reranker-large --local-dir d:\workSpace-trae\personal-robot-master\models\offline\bge-reranker-large --local-dir-use-symlinks False
```

### 方法2: 手动从网页下载
1. 访问模型的HuggingFace页面
2. 下载所有必要的文件（包括config.json、pytorch_model.bin等）
3. 将文件放入对应的目录中

## 模型放置位置

您有三种选择来放置下载的模型文件：

### 选项1: 标准HuggingFace缓存位置（推荐）
- BAAI/bge-large-zh-v1.5: `C:\Users\13352\.cache\huggingface\hub\models--BAAI\bge-large-zh-v1.5`
- BAAI/bge-reranker-large: `C:\Users\13352\.cache\huggingface\hub\models--BAAI\bge-reranker-large`

### 选项2: 标准Sentence Transformers缓存位置
- BAAI/bge-large-zh-v1.5: `C:\Users\13352\.cache\torch\sentence_transformers\BAAI-bge-large-zh-v1.5`
- BAAI/bge-reranker-large: `C:\Users\13352\.cache\torch\sentence_transformers\BAAI-bge-reranker-large`

### 选项3: 项目内离线模型目录
- BAAI/bge-large-zh-v1.5: `d:\workSpace-trae\personal-robot-master\models\offline\bge-large-zh-v1.5`
- BAAI/bge-reranker-large: `d:\workSpace-trae\personal-robot-master\models\offline\bge-reranker-large`

> **注意**: 如果使用选项3（项目内目录），需要修改代码以支持从本地路径加载模型。

## 修改配置以使用离线模型

### 1. 修改.env文件
如果您想使用项目内的离线模型目录，可以在`.env`文件中添加以下配置：

```env
# 使用本地路径的模型
EMBEDDING_MODEL_PATH=d:\workSpace-trae\personal-robot-master\models\offline\bge-large-zh-v1.5
RERANKER_MODEL_PATH=d:\workSpace-trae\personal-robot-master\models\offline\bge-reranker-large
```

### 2. 修改vector_store.py文件
为了支持从本地路径加载模型，需要修改`vector_store.py`文件中的模型初始化逻辑。

## 验证模型是否正常加载
启动应用后，可以查看日志输出，确认模型是否成功从本地路径加载：

```
成功初始化本地嵌入模型: BAAI/bge-large-zh-v1.5
成功初始化重排序模型: BAAI/bge-reranker-large
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件是否完整
   - 确认模型路径是否正确
   - 查看日志中的具体错误信息

2. **性能问题**
   - 确保下载了完整的模型文件
   - 考虑使用GPU加速（如果可用）

3. **内存不足**
   - 对于大型模型，可能需要调整系统内存配置

## 其他说明

- 模型文件可能较大，请确保有足够的存储空间
- 定期检查模型更新，以获取更好的性能
- 如果需要，可以使用较小版本的模型来节省资源