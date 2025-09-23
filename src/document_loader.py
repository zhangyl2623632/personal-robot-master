import logging
import os
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    ImageCaptionLoader
)
from langchain_community.document_loaders.pdf import PyPDFLoader as PDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
import pytesseract
from src.config import global_config

# 导入文档分类器
from src.document_classifier import document_classifier

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentLoader:
    """文档加载器，用于加载和处理不同类型的本地文档"""
    
    def __init__(self, config=None):
        """初始化文档加载器"""
        self.config = config or global_config
        # 优化：使用更适合中文的分块策略
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]  # 添加中文标点符号
        )
        
        # 为PDF文档准备特殊的分块策略
        self.pdf_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(self.config.CHUNK_SIZE * 0.8),  # PDF文档块大小稍小
            chunk_overlap=int(self.config.CHUNK_OVERLAP * 1.2),  # 重叠稍大
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
    
    def load_document(self, file_path):
        """加载单个文档"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            # 检查文件是否为空
            if os.path.getsize(file_path) == 0:
                logger.warning(f"空文件，跳过加载: {file_path}")
                return []
                
            # 检查文件是否为有效的文本文件
            if file_extension in ['.txt', '.md', '.csv']:
                try:
                    # 尝试以UTF-8编码打开文件，检查是否为有效文本
                    with open(file_path, 'r', encoding='utf-8') as f:
                        f.read(4096)  # 只读取一部分进行验证
                except UnicodeDecodeError:
                    logger.warning(f"文件编码错误，不是有效文本文件: {file_path}")
                    return []
                    
            # 对于PDF文件，进行特殊处理以检测损坏的文件
            elif file_extension == '.pdf':
                try:
                    # 简单验证PDF文件头
                    with open(file_path, 'rb') as f:
                        header = f.read(4)
                        if header != b'%PDF':
                            logger.warning(f"无效的PDF文件: {file_path}")
                            return []
                except Exception as e:
                    logger.warning(f"PDF文件验证失败: {file_path}, 错误: {str(e)}")
                    return []
                    
            # 加载文档的原有逻辑
            if file_extension == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_extension == ".md":
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_extension == ".pdf":
                loader = PDFLoader(file_path)
            elif file_extension in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(file_path)
            elif file_extension == ".csv":
                loader = CSVLoader(file_path, encoding="utf-8")
            elif file_extension in [".jpg", ".jpeg", ".png", ".gif"]:
                # 对于图片，使用OCR提取文本
                try:
                    text = pytesseract.image_to_string(Image.open(file_path), lang='chi_sim+eng')
                    from langchain_core.documents import Document
                    return [Document(page_content=text, metadata={"source": file_path})]
                except Exception as e:
                    logger.error(f"图片OCR处理失败: {file_path}, 错误: {str(e)}")
                    return []
            else:
                logger.warning(f"不支持的文件类型: {file_path}")
                return []
            
            # 加载文档
            documents = loader.load()
            # 如果是临时文件，尝试获取原始文件名（从元数据或路径推导）
            if "temp" in file_path.lower():
                # 尝试从元数据获取原始文件名
                original_filename = documents[0].metadata.get('source', file_path) if documents else file_path
                logger.info(f"成功加载文档: {file_path} (原始文件名: {original_filename})")
            else:
                logger.info(f"成功加载文档: {file_path}")
            return documents
        except Exception as e:
            logger.error(f"加载文档失败: {file_path}, 错误: {str(e)}")
            return []
    
    def load_directory(self, directory_path=None):
        """加载目录中的所有文档"""
        directory_path = directory_path or self.config.DOCUMENTS_PATH
        
        if not os.path.exists(directory_path):
            logger.warning(f"目录不存在: {directory_path}")
            return []
        
        all_documents = []
        try:
            # 获取目录下的所有文件
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 逐个加载文件
                    docs = self.load_document(file_path)
                    all_documents.extend(docs)
            
            logger.info(f"成功加载目录中的文档，共 {len(all_documents)} 个")
            return all_documents
        except Exception as e:
            logger.error(f"加载目录文档失败: {directory_path}, 错误: {str(e)}")
            return []
    
    def _get_loader_cls(self, file_path):
        """根据文件扩展名获取对应的加载器实例"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension in [".txt", ".md"]:
                return TextLoader(file_path, encoding="utf-8")
            elif file_extension == ".pdf":
                return PDFLoader(file_path)
            elif file_extension in [".docx", ".doc"]:
                return UnstructuredWordDocumentLoader(file_path)
            elif file_extension in [".xlsx", ".xls"]:
                return UnstructuredExcelLoader(file_path)
            elif file_extension == ".csv":
                return CSVLoader(file_path, encoding="utf-8")
            elif file_extension in [".jpg", ".jpeg", ".png", ".gif"]:
                # 对于图片，使用OCR提取文本
                text = pytesseract.image_to_string(Image.open(file_path), lang='chi_sim+eng')
                from langchain_core.documents import Document
                return [Document(page_content=text, metadata={"source": file_path})]
            else:
                logger.warning(f"不支持的文件类型: {file_path}")
                return None
        except Exception as e:
            logger.error(f"创建加载器失败: {file_path}, 错误: {str(e)}")
            return None
    
    def split_documents(self, documents):
        """将文档分割成小块"""
        if not documents:
            return []
        
        try:
            # 为PDF文档使用特殊的分割策略
            split_docs = []
            normal_docs = []
            pdf_docs = []
            
            # 分离PDF文档和其他文档
            for doc in documents:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata and doc.metadata['source'].lower().endswith('.pdf'):
                    pdf_docs.append(doc)
                else:
                    normal_docs.append(doc)
            
            # 先分割非PDF文档
            if normal_docs:
                normal_split_docs = self.text_splitter.split_documents(normal_docs)
                split_docs.extend(normal_split_docs)
                logger.info(f"成功分割非PDF文档，从 {len(normal_docs)} 个原始文档分割为 {len(normal_split_docs)} 个小块")
            
            # 对PDF文档使用更稳健的分割策略
            if pdf_docs:
                logger.info(f"正在使用特殊策略分割 {len(pdf_docs)} 个PDF文档")
                
                pdf_split_docs = []
                for pdf_doc in pdf_docs:
                    try:
                        # 使用我们预定义的PDF专用分割器
                        doc_splits = self.pdf_text_splitter.split_documents([pdf_doc])
                        pdf_split_docs.extend(doc_splits)
                        logger.info(f"成功分割PDF文档，获得 {len(doc_splits)} 个小块")
                    except Exception as e:
                        logger.error(f"分割PDF文档失败: {str(e)}")
                        try:
                            # 尝试使用常规文本分割器
                            doc_splits = self.text_splitter.split_documents([pdf_doc])
                            pdf_split_docs.extend(doc_splits)
                            logger.warning(f"使用常规分割器成功分割PDF文档，获得 {len(doc_splits)} 个小块")
                        except Exception as inner_e:
                            logger.error(f"PDF文档备用分割方法也失败: {str(inner_e)}")
                            # 最后尝试：直接将整个文档作为一个块
                            if len(pdf_doc.page_content) > 0:
                                pdf_split_docs.append(pdf_doc)
                                logger.warning(f"对文档使用原始内容作为单个块")
            
            split_docs.extend(pdf_split_docs)
            
            # 为每个文档块添加分类信息
            classified_docs = []
            for doc in split_docs:
                try:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        file_path = doc.metadata['source']
                        # 使用文档分类器对文档进行分类
                        classification_result = document_classifier.classify_document(file_path, doc.page_content)
                        
                        # 合并原始元数据和分类信息
                        enhanced_metadata = {
                            **doc.metadata,
                            "document_type": classification_result.get("document_type", "unknown"),
                            "document_type_name": classification_result.get("document_type_name", "未知文档类型"),
                            "classification_confidence": classification_result.get("confidence", 0.0),
                            "classification_method": classification_result.get("classification_method", "unknown"),
                            "source_file": file_path,
                            "source_file_name": os.path.basename(file_path)
                        }
                        
                        # 添加分类详情（如果有）
                        if "details" in classification_result:
                            enhanced_metadata["classification_details"] = classification_result["details"]
                        
                        # 添加从分类器获取的文档元数据
                        if "metadata" in classification_result:
                            for key, value in classification_result["metadata"].items():
                                # 避免覆盖已有键
                                if key not in enhanced_metadata:
                                    enhanced_metadata[key] = value
                        
                        # 创建带有增强元数据的新文档
                        from langchain_core.documents import Document
                        classified_doc = Document(
                            page_content=doc.page_content,
                            metadata=enhanced_metadata
                        )
                        classified_docs.append(classified_doc)
                    else:
                        classified_docs.append(doc)
                except Exception as e:
                    logger.error(f"文档分类失败: {str(e)}")
                    classified_docs.append(doc)  # 如果分类失败，使用原始文档
            
            # 关键修复：确保即使分割后得到0个文档块，也至少返回原始文档
            if not classified_docs:
                logger.warning(f"文档分割后得到0个文档块，返回原始文档")
                return documents
            
            logger.info(f"成功分割并分类所有文档，从 {len(documents)} 个原始文档分割为 {len(classified_docs)} 个小块")
            return classified_docs
        except Exception as e:
            logger.error(f"分割文档失败: {str(e)}")
            # 作为最后的后备方案，返回原始文档
            return documents

# 创建文档加载器实例
document_loader = DocumentLoader()