import os
import sys
import logging
import argparse
from src.rag_pipeline import rag_pipeline
from src.config import global_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonalRobotCLI:
    """个人智能问答机器人命令行界面"""
    
    def __init__(self):
        """初始化命令行界面"""
        self.rag_pipeline = rag_pipeline
        self.parser = self._create_parser()
    
    def _create_parser(self):
        """创建命令行参数解析器"""
        parser = argparse.ArgumentParser(description="个人智能问答机器人")
        subparsers = parser.add_subparsers(dest="command", help="支持的命令")
        
        # 添加文档命令
        add_parser = subparsers.add_parser("add", help="添加文档到向量存储")
        add_parser.add_argument("--path", type=str, help="文档路径，可以是文件或目录")
        
        # 问答命令
        ask_parser = subparsers.add_parser("ask", help="向机器人提问")
        ask_parser.add_argument("--query", type=str, help="查询内容")
        
        # 查看状态命令
        status_parser = subparsers.add_parser("status", help="查看系统状态")
        
        # 清空命令
        clear_parser = subparsers.add_parser("clear", help="清空数据")
        clear_parser.add_argument("--what", choices=["history", "vectors", "all"], default="history",
                              help="要清空的内容: history(对话历史), vectors(向量存储), all(全部)")
        
        # 启动交互式问答
        chat_parser = subparsers.add_parser("chat", help="启动交互式问答")
        
        return parser
    
    def run(self, args=None):
        """运行命令行界面"""
        if args is None:
            args = sys.argv[1:]
            
        # 如果没有参数，显示帮助信息
        if not args:
            self.parser.print_help()
            return
            
        parsed_args = self.parser.parse_args(args)
        
        # 根据命令执行相应的操作
        if parsed_args.command == "add":
            self._handle_add(parsed_args)
        elif parsed_args.command == "ask":
            self._handle_ask(parsed_args)
        elif parsed_args.command == "status":
            self._handle_status()
        elif parsed_args.command == "clear":
            self._handle_clear(parsed_args)
        elif parsed_args.command == "chat":
            self._handle_chat()
        else:
            self.parser.print_help()
    
    def _handle_add(self, args):
        """处理添加文档命令"""
        if not args.path:
            print("错误: 请提供文档路径")
            return
            
        path = os.path.abspath(args.path)
        if not os.path.exists(path):
            print(f"错误: 路径不存在: {path}")
            return
            
        if os.path.isfile(path):
            # 添加单个文件
            success = self.rag_pipeline.add_single_document(path)
            if success:
                print(f"成功添加文件到向量存储: {path}")
            else:
                print(f"添加文件失败: {path}")
        elif os.path.isdir(path):
            # 添加目录下的所有文件
            success = self.rag_pipeline.process_documents(path)
            if success:
                print(f"成功处理目录下的文档: {path}")
            else:
                print(f"处理目录下的文档失败: {path}")
    
    def _handle_ask(self, args):
        """处理单次提问命令"""
        if not args.query:
            print("错误: 请提供查询内容")
            return
            
        print(f"问题: {args.query}")
        print("正在思考...")
        
        # 生成回答
        answer = self.rag_pipeline.answer_query(args.query)
        
        if answer:
            print(f"回答: {answer}")
        else:
            print("生成回答失败，请检查日志获取更多信息")
    
    def _handle_status(self):
        """处理查看状态命令"""
        print("===== 系统状态 =====")
        
        # 检查API密钥
        api_key_valid = self.rag_pipeline.validate_api_key()
        print(f"API密钥状态: {'有效' if api_key_valid else '无效或未配置'}")
        
        # 向量存储状态
        vector_count = self.rag_pipeline.get_vector_count()
        print(f"向量存储中文档数量: {vector_count}")
        
        # 配置信息
        print(f"文档目录: {global_config.DOCUMENTS_PATH}")
        print(f"向量存储路径: {global_config.VECTOR_STORE_PATH}")
        print(f"使用的模型: {global_config.MODEL_NAME}")
        print("=================")
    
    def _handle_clear(self, args):
        """处理清空命令"""
        if args.what == "history" or args.what == "all":
            self.rag_pipeline.clear_conversation_history()
            print("对话历史已清空")
            
        if args.what == "vectors" or args.what == "all":
            success = self.rag_pipeline.clear_vector_store()
            if success:
                print("向量存储已清空")
            else:
                print("清空向量存储失败")
    
    def _handle_chat(self):
        """处理交互式问答命令"""
        print("===== 个人智能问答机器人 =====")
        print("提示: 输入 'exit' 或 'quit' 退出聊天，输入 'clear' 清空对话历史")
        
        # 检查向量存储状态
        vector_count = self.rag_pipeline.get_vector_count()
        if vector_count == 0:
            print("警告: 向量存储为空，回答将不基于本地文档")
        else:
            print(f"当前向量存储中有 {vector_count} 个文档")
        
        # 检查API密钥
        if not global_config.DEEPSEEK_API_KEY:
            print("警告: 未配置DeepSeek API密钥，请在.env文件中设置DEEPSEEK_API_KEY")
        
        print("================================")
        
        while True:
            try:
                query = input("\n你: ")
                
                if query.lower() in ["exit", "quit", "退出"]:
                    print("再见！")
                    break
                elif query.lower() in ["clear", "清空"]:
                    self.rag_pipeline.clear_conversation_history()
                    print("对话历史已清空")
                    continue
                
                print("机器人: 正在思考...")
                
                # 生成回答
                answer = self.rag_pipeline.answer_query(query)
                
                if answer:
                    print(f"机器人: {answer}")
                else:
                    print("机器人: 抱歉，我无法回答这个问题。")
                    
            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"发生错误: {str(e)}")
                logger.error(f"交互式问答错误: {str(e)}")

# 创建命令行界面实例
cli = PersonalRobotCLI()

if __name__ == "__main__":
    cli.run()