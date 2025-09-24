# 个人智能问答机器人

版本：v1.0.1

基于大语言模型 + RAG技术 + 本地文档的智能问答系统，能够基于你的本地文档提供精准、上下文相关的智能问答服务。本项目专为个人用户设计，支持多种文档格式，提供直观的Web界面，让AI知识服务触手可及。

## 功能特点

### 文档处理能力
- **多格式文档支持**：全面支持TXT、Markdown、PDF、Word（DOCX）、Excel、PowerPoint等常用文档格式
- **文档自动解析**：智能提取文档内容，保留结构化信息和重要上下文
- **文档批量处理**：支持批量导入和处理目录下的所有文档，提高效率
- **文档监控更新**：自动监控指定目录，实时检测并处理新增或修改的文档
- **文档分类系统**：自动识别和分类不同类型的文档，优化检索策略

### 检索与问答能力
- **高效向量检索**：基于Chroma向量数据库实现快速、准确的相似度搜索
- **自适应RAG技术**：根据文档类型和查询意图动态选择最优RAG策略，提高回答质量
- **重排序优化**：使用专业重排序模型（BAAI/bge-reranker-large）进一步提升检索结果质量
- **上下文管理**：智能处理多轮对话，保持上下文一致性
- **多语言支持**：特别优化了中英文文档间的查询和回答能力

### 用户体验
- **直观Web界面**：提供美观、易用的Web交互界面，无需编程知识
- **实时响应**：支持流式输出，提供流畅的对话体验
- **多模型支持**：内置支持DeepSeek、OpenAI、Qwen、Moonshot等多种主流大语言模型
- **本地模型支持**：支持使用BAAI/bge-large-zh-v1.5嵌入模型和BAAI/bge-reranker-large重排序模型的离线使用

### 扩展与管理
- **灵活配置系统**：通过环境变量和配置文件实现高度可定制的系统行为
- **命令行工具**：提供丰富的命令行工具，便于自动化和批量操作
- **向量存储管理**：支持向量库的创建、查询、更新和清理
- **健康检查**：内置系统状态监控和自检功能

## 技术架构

### 核心组件

1. **DocumentLoader**（文档加载器）
   - 负责加载和解析各种格式的本地文档
   - 支持文档分块、元数据提取
   - 自动处理文档编码和格式转换

2. **DocumentClassifier**（文档分类器）
   - 自动识别文档类型和内容特征
   - 为不同类型文档应用不同的处理策略
   - 支持自定义文档类型配置

3. **QueryIntentClassifier**（查询意图分类器）
   - 分析用户查询意图和需求类型
   - 为不同类型的查询选择最佳处理路径
   - 支持复杂查询意图的识别

4. **VectorStoreManager**（向量存储管理器）
   - 管理Chroma向量数据库的创建和维护
   - 负责文档向量化和向量检索
   - 支持向量库的优化和压缩

5. **LLMClient**（大语言模型客户端）
   - 封装各种大语言模型的API调用
   - 支持流式和非流式响应处理
   - 负责提示词工程和模型参数优化
   - 支持动态模型切换

6. **AdaptiveRAGPipeline**（自适应RAG流水线）
   - 根据文档类型和查询意图动态选择最优RAG策略
   - 整合文档检索和大语言模型生成
   - 实现检索结果重排序和上下文构建
   - 优化生成回答的相关性和准确性

7. **WebInterface**（Web界面）
   - 基于Flask框架实现的Web服务
   - 提供用户友好的交互界面
   - 处理前端请求和后端响应
   - 支持模型切换和系统状态监控

8. **DocumentMonitor**（文档监控器）
   - 实时监控指定目录的文件变化
   - 自动处理新增或修改的文档
   - 支持定时扫描和事件触发模式

### 技术栈

- **后端**：Python 3.8+, Flask
- **向量数据库**：Chroma
- **文档处理**：python-docx, PyPDF2, python-pptx, pandas
- **自然语言处理**：Transformers, SentenceTransformers, LangChain
- **大语言模型**：DeepSeek, OpenAI, Qwen, Moonshot
- **前端**：HTML, CSS, JavaScript

## 环境要求

- **操作系统**：Windows 10/11, macOS 11+, Linux (Ubuntu 20.04+)
- **Python版本**：3.8 或更高版本
- **内存**：建议16GB以上（处理大型文档时）
- **存储空间**：至少1GB可用空间（用于存储文档和向量数据）
- **网络连接**：需要网络连接以访问大语言模型API（除非全部使用离线模型）

## 安装指南

### 1. 准备Python环境

确保您的系统已安装Python 3.8或更高版本。您可以通过以下命令检查Python版本：

```bash
python --version
# 或
python3 --version
```

如果未安装Python，请访问[Python官网](https://www.python.org/downloads/)下载并安装最新版本。

### 2. 克隆项目或下载源码

如果项目在Git仓库中：

```bash
# 克隆项目仓库
git clone https://github.com/your-username/personal-robot.git
cd personal-robot
```

如果是直接下载的源码包，请解压并进入项目目录。

### 3. 创建虚拟环境（推荐）

为避免依赖冲突，建议创建一个专用的Python虚拟环境：

```bash
# 在Windows上
python -m venv .venv
.venv\Scripts\activate

# 在macOS/Linux上
python3 -m venv .venv
source .venv/bin/activate
```

### 4. 安装依赖包

使用pip安装项目所需的所有依赖：

```bash
pip install -r requirements.txt
```

### 5. 配置环境变量

复制`.env.example`文件并重命名为`.env`：

```bash
# 在Windows上
copy .env.example .env

# 在macOS/Linux上
cp .env.example .env
```

编辑`.env`文件，填入您的配置信息：

```bash
# 大语言模型API配置
DEEPSEEK_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here
MOONSHOT_API_KEY=your_api_key_here
QWEN_API_KEY=your_api_key_here

# 模型选择配置
MODEL_PROVIDER=deepseek  # 可选: deepseek, openai, moonshot, qwen
MODEL_NAME=deepseek-chat

# 向量存储配置
VECTOR_STORE_PATH=./vector_store
DOCUMENTS_PATH=./data

# 嵌入和重排序模型配置
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
RERANKER_MODEL=BAAI/bge-reranker-large

# 生成参数配置
TEMPERATURE=0.1
MAX_TOKENS=2048
TOP_P=0.9

# 检索参数配置
TOP_K=4
TOP_R=2
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

## 使用指南

### 通过Web界面使用

1. **启动Web服务**
   
   在Windows系统上，您可以直接双击运行`start_web.bat`文件。
   
   或者，通过命令行启动：
   
   ```bash
   python -m src.web_interface
   # 或
   flask run --app src/web_interface.py --debug
   ```
   
2. **访问Web界面**
   
   在浏览器中输入以下地址：
   
   ```
   http://localhost:5000
   ```
   
3. **使用功能**
   
   Web界面提供以下主要功能：
   
   - **提问**：在输入框中输入您的问题，系统会基于已加载的文档进行回答
   - **文档上传**：通过上传按钮添加新的文档到系统
   - **对话历史**：查看和管理之前的对话
   - **系统状态**：查看向量库大小、模型连接状态等系统信息
   - **模型切换**：在支持的大语言模型之间切换

### 通过命令行管理

项目提供了丰富的命令行工具，用于文档管理和系统操作：

#### 1. 添加文档

```bash
# 添加单个文件
python src/main.py add --path data/example.docx

# 添加整个目录下的所有文件
python src/main.py add --path data/

# 添加目录并递归处理子目录
python src/main.py add --path data/ --recursive
```

#### 2. 查看系统状态

```bash
python src/main.py status
```

此命令将显示：
- 向量存储中的文档数量
- 已加载的模型信息
- 系统配置状态

#### 3. 清理数据

```bash
# 清空向量存储
python src/main.py clear --what vectors

# 清空指定目录下的文档
python src/main.py clear --what documents --path data/

# 清空所有数据
python src/main.py clear --what all
```

#### 4. 重建向量存储

如果您修改了配置或需要重新处理所有文档，可以使用以下命令重建向量存储：

```bash
python rebuild_vector_store.py
```

此脚本会自动处理`data`目录下的所有文档，并重新构建向量存储。

## 项目结构详解

```
personal-robot/
├── src/                 # 源代码目录
│   ├── adaptive_rag_pipeline.py  # 自适应RAG流水线实现
│   ├── config.py        # 全局配置管理
│   ├── document_classifier.py    # 文档分类器
│   ├── document_loader.py        # 文档加载与处理模块
│   ├── document_monitor.py       # 文档监控与自动更新
│   ├── llm_client.py    # 大语言模型客户端实现
│   ├── main.py          # 命令行入口
│   ├── query_intent_classifier.py # 查询意图分类器
│   ├── rag_pipeline.py  # RAG流程核心实现
│   ├── vector_store.py  # 向量存储管理与检索
│   ├── version_manager.py # 版本管理器
│   └── web_interface.py # Web服务入口
├── config/              # 配置文件目录
│   ├── document_types.yaml      # 文档类型配置
│   ├── multilingual_rag_prompts.yaml # 多语言RAG提示词配置
│   └── rag_strategies.yaml      # RAG策略配置
├── data/                # 默认文档存放目录
├── vector_store/        # 向量存储数据目录
├── models/              # 模型存放目录
│   └── offline/         # 离线模型文件
├── templates/           # Web模板文件
│   └── index.html       # 主页面模板
├── static/              # Web静态资源
│   ├── style.css        # 样式表
│   └── script.js        # JavaScript脚本
├── tests/               # 测试代码目录
│   ├── unit/            # 单元测试
│   ├── integration/     # 集成测试
│   ├── performance/     # 性能测试
│   ├── real_world/      # 真实场景测试
│   └── system/          # 系统测试
├── test_data/           # 测试数据目录
├── .env                 # 环境变量配置
├── .env.example         # 配置示例
├── requirements.txt     # 项目依赖列表
├── rebuild_vector_store.py # 重建向量存储脚本
├── start_web.bat        # Windows启动脚本
└── README.md            # 项目文档
```

## 详细配置说明

项目的所有配置项都可以在`.env`文件中设置。以下是主要配置项的详细说明：

### 大语言模型配置

| 配置项 | 说明 | 默认值 | 是否必需 |
|-------|------|-------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek大语言模型API密钥 | - | 至少需要一个API密钥 |
| `OPENAI_API_KEY` | OpenAI大语言模型API密钥 | - | 可选 |
| `MOONSHOT_API_KEY` | 月之暗面大语言模型API密钥 | - | 可选 |
| `QWEN_API_KEY` | 通义千问大语言模型API密钥 | - | 可选 |
| `MODEL_PROVIDER` | 选择使用的模型提供商 | deepseek | 必需 |
| `MODEL_NAME` | 大语言模型名称 | deepseek-chat | 必需 |
| `TEMPERATURE` | 生成温度，控制输出随机性 | 0.1 | 可选 |
| `MAX_TOKENS` | 最大生成token数 | 2048 | 可选 |
| `TOP_P` | 核采样参数 | 0.9 | 可选 |

### 文档与向量存储配置

| 配置项 | 说明 | 默认值 | 是否必需 |
|-------|------|-------|--------|
| `VECTOR_STORE_PATH` | 向量存储路径 | ./vector_store | 必需 |
| `DOCUMENTS_PATH` | 文档存放路径 | ./data | 必需 |
| `EMBEDDING_MODEL` | 嵌入模型名称 | BAAI/bge-large-zh-v1.5 | 必需 |
| `RERANKER_MODEL` | 重排序模型名称 | BAAI/bge-reranker-large | 必需 |
| `TOP_K` | 检索时返回的最相关文档数 | 4 | 可选 |
| `TOP_R` | 重排序后保留的文档数 | 2 | 可选 |
| `CHUNK_SIZE` | 文档分块大小 | 1000 | 可选 |
| `CHUNK_OVERLAP` | 文档块之间的重叠大小 | 100 | 可选 |

## 自适应RAG策略

项目实现了15种不同的RAG策略，根据文档类型和查询意图自动选择最合适的策略。主要策略包括：

- **多语言RAG策略**：优化中英文文档间的查询和回答能力
- **学术论文策略**：针对学术内容的特殊处理，支持引用格式输出
- **技术文档策略**：增强技术文档的理解和解释能力
- **需求文档策略**：聚焦于文档中的具体要求和功能点
- **小说内容策略**：保持叙事风格一致性的回答生成

每种策略可配置以下参数：
- `embedding_model`：使用的嵌入模型
- `prompt_template`：使用的提示模板
- `reranker_weight`：重排序权重
- `score_threshold`：相关性分数阈值
- `top_k`：检索文档数量
- `max_context_length`：最大上下文长度

## 离线模型使用指南

项目支持使用离线的嵌入模型和重排序模型。这些模型存放在`models/offline/`目录下。

### 使用步骤

1. **下载模型**
   
   下载BAAI/bge-large-zh-v1.5嵌入模型和BAAI/bge-reranker-large重排序模型，并放置在`models/offline/`目录下。
   
2. **配置模型路径**
   
   在`.env`文件中确保以下配置正确：
   
   ```bash
   EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
   RERANKER_MODEL=BAAI/bge-reranker-large
   ```
   
   系统会自动检测`models/offline/`目录下是否存在这些模型，如果存在则优先使用本地模型。
   
3. **验证模型加载**
   
   启动系统后，可以通过`status`命令或Web界面查看模型是否成功加载。

## 高级功能

### 文档监控

系统支持实时监控指定目录下的文档变化。当有新文档添加或现有文档修改时，系统会自动处理并更新向量存储。

启用文档监控：

```bash
# 在命令行中启动文档监控
python src/main.py monitor --path data/
```

### 自定义提示词

您可以在`config/rag_strategies.yaml`文件中自定义各种提示模板，以优化模型的回答质量。系统已预置12种不同场景的提示模板。

### 批量操作脚本

项目提供了多个批量操作脚本：
- `rebuild_vector_store.py`：一键重建向量存储
- `check_vector_store.py`：检查向量存储状态
- `clear_vector_store.py`：清空向量存储
- `verify_vector_store.py`：验证向量存储内容

## 测试系统

项目包含完善的测试框架，涵盖以下测试类型：

- **单元测试**：测试各个模块的基本功能
- **集成测试**：测试模块间的交互
- **性能测试**：测试系统在不同负载下的表现
- **真实场景测试**：在模拟真实用户场景下测试系统

运行测试：

```bash
# 运行所有测试
pytest

# 运行特定类型的测试
pytest tests/unit/
pytest tests/integration/
```

## 注意事项

1. **API密钥保护**
   
   请妥善保管您的API密钥，不要将包含密钥的`.env`文件上传到公共仓库。
   
2. **性能优化**
   
   - 对于大型文档集，建议增加系统内存或调整`CHUNK_SIZE`参数
   - 如果查询速度较慢，可以适当减少`TOP_K`值
   - 对于低配置设备，可以考虑使用更小的嵌入模型
   
3. **文档准备**
   
   - 尽量使用清晰、结构化的文档格式
   - 对于扫描版PDF，请确保OCR质量良好
   - 避免使用过于复杂的文档结构，这可能影响解析效果
   
4. **多语言支持**
   
   系统特别优化了中英文文档的处理，对于其他语言可能需要调整模型配置。

## 故障排除

### 常见问题及解决方案

**问题：API调用失败**
- 检查`.env`文件中的API密钥是否正确
- 确认网络连接正常，无防火墙限制
- 检查API调用配额是否已用完

**问题：文档无法正常解析**
- 确认文档格式受支持
- 检查文档是否损坏
- 对于加密文档，需要先解密

**问题：问答结果不准确**
- 尝试调整`TOP_K`和`score_threshold`值
- 优化文档质量，确保内容清晰、结构化
- 检查是否选择了合适的RAG策略

**问题：内存不足错误**
- 减少同时处理的文档数量
- 增大系统内存
- 调整`CHUNK_SIZE`参数，减小文档块大小

**问题：Web界面无法访问**
- 检查服务是否正常启动
- 确认端口5000未被其他程序占用
- 尝试使用不同的浏览器访问

## 未来规划

项目计划在未来版本中实现以下功能：

1. **离线大语言模型支持**：允许在无网络环境下使用本地大语言模型
2. **更多文档格式支持**：扩展支持更多专业文档格式
3. **多模态支持**：增加对图片、表格等非文本内容的处理能力
4. **个性化配置**：提供更多自定义选项，满足不同用户需求
5. **模型管理界面**：直观的模型选择和配置界面
6. **数据可视化**：提供问答质量和系统性能的可视化分析

具体实现方案请参考`FEATURE_ROADMAP.md`文件。

## 贡献指南

我们欢迎并感谢任何形式的贡献！如果您有兴趣为项目贡献代码、文档或想法，请按照以下步骤进行：

1. Fork项目仓库
2. 创建您的特性分支
3. 提交您的更改
4. 推送到您的Fork
5. 创建一个Pull Request

## 许可证

本项目采用[MIT许可证](https://opensource.org/licenses/MIT)。

## 更新日志

### v1.0.1 (2024-09-16)
- 优化向量存储管理和检索效率
- 改进文档处理流程，提高解析准确性
- 实现自适应RAG流水线，支持15种RAG策略
- 增强系统稳定性和错误处理能力
- 更新依赖包版本
- 修复模型切换功能，确保正确更新LLM客户端实例

### v1.0.0 (2024-09-15)
- 首次发布个人智能问答机器人
- 支持多格式文档加载和处理
- 实现向量存储和相似度搜索功能
- 提供Web交互界面
- 支持单次提问和对话模式

## 致谢

感谢以下开源项目和技术支持：

- [Chroma](https://www.trychroma.com/) - 高效的向量数据库
- [Transformers](https://huggingface.co/transformers/) - 自然语言处理模型库
- [SentenceTransformers](https://www.sbert.net/) - 句子嵌入模型库
- [Flask](https://flask.palletsprojects.com/) - Web框架
- [BAAI](https://www.baai.ac.cn/) - 提供的开源嵌入和重排序模型
- 各大语言模型提供商的API支持

---

祝您使用愉快！如有任何问题或建议，请随时联系我们。