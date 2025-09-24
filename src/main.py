import os
import sys
import logging
import argparse
import os
import asyncio
from src.rag_pipeline import rag_pipeline
from src.agent import agent
from src.config import global_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonalRobotCLI:
    """个人智能问答机器人命令行界面"""
    
    def __init__(self):
        """初始化命令行界面"""
        self.rag_pipeline = rag_pipeline
        self.agent = agent  # 使用新的agent模块
        self.use_agent = True  # 默认使用agent
        self.use_streaming = True  # 默认使用流式响应
        self.parser = self._create_parser()
    
    def _create_parser(self):
        """创建命令行参数解析器"""
        parser = argparse.ArgumentParser(description="个人智能问答机器人")
        
        # 全局参数
        parser.add_argument("--no-agent", action="store_true", help="使用传统RAG流水线而不是新的Agent")
        parser.add_argument("--no-streaming", action="store_true", help="禁用流式响应")
        
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
        chat_parser.add_argument("--no-knowledge", action="store_true", help="不使用知识库回答")
        
        # 工具命令组
        tools_parser = subparsers.add_parser("tools", help="知识库管理工具")
        tools_subparsers = tools_parser.add_subparsers(dest="tool_command")
        
        # 创建索引
        create_index_parser = tools_subparsers.add_parser("create-index", help="创建新索引")
        create_index_parser.add_argument("--name", type=str, required=True, help="索引名称")
        
        # 切换索引
        switch_index_parser = tools_subparsers.add_parser("switch-index", help="切换索引")
        switch_index_parser.add_argument("--name", type=str, required=True, help="索引名称")
        
        # 优化索引
        optimize_index_parser = tools_subparsers.add_parser("optimize-index", help="优化索引")
        optimize_index_parser.add_argument("--name", type=str, help="索引名称（可选，默认当前索引）")
        
        # 获取统计信息
        stats_parser = tools_subparsers.add_parser("stats", help="获取统计信息")
        
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
        
        # 处理全局参数
        self.use_agent = not getattr(parsed_args, "no_agent", False)
        self.use_streaming = not getattr(parsed_args, "no_streaming", False)
        
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
            self._handle_chat(parsed_args)
        elif parsed_args.command == "tools":
            self._handle_tools(parsed_args)
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
        if self.use_agent:
            # 使用新的agent
            if self.use_streaming:
                # 流式响应
                async def get_streaming_answer():
                    print("回答: ", end="", flush=True)
                    async for chunk in self.agent.generate_streaming_response(args.query):
                        print(chunk, end="", flush=True)
                    print()  # 换行
                
                asyncio.run(get_streaming_answer())
            else:
                # 非流式响应
                answer = self.agent.generate_response(args.query)
                print(f"回答: {answer}")
        else:
            # 使用传统rag_pipeline
            answer = self.rag_pipeline.answer_query(args.query)
            
            if answer:
                print(f"回答: {answer}")
            else:
                print("生成回答失败，请检查日志获取更多信息")
    
    def _handle_status(self):
        """处理查看状态命令"""
        print("===== 系统状态 =====")
        
        # 显示当前模式
        print(f"运行模式: {'Agent模式' if self.use_agent else '传统RAG模式'}")
        print(f"响应模式: {'流式响应' if self.use_streaming else '非流式响应'}")
        
        # 检查API密钥
        api_key_valid = self.rag_pipeline.validate_api_key()
        print(f"API密钥状态: {'有效' if api_key_valid else '无效或未配置'}")
        
        # 向量存储状态
        vector_count = self.rag_pipeline.get_vector_count()
        print(f"向量存储中文档数量: {vector_count}")
        
        # 如果使用agent，显示对话统计
        if self.use_agent:
            stats = self.agent.get_conversation_stats()
            print(f"\n===== 对话统计 =====")
            print(f"总响应数: {stats.get('total_responses', 0)}")
            print(f"总搜索查询: {stats.get('total_search_queries', 0)}")
            print(f"总工具使用: {stats.get('total_tools_used', 0)}")
        
        # 配置信息
        print(f"\n===== 配置信息 =====")
        print(f"文档目录: {global_config.DOCUMENTS_PATH}")
        print(f"向量存储路径: {global_config.VECTOR_STORE_PATH}")
        print(f"使用的模型: {global_config.MODEL_NAME}")
        print("=================")
    
    def _handle_clear(self, args):
        """处理清空命令"""
        if args.what == "history" or args.what == "all":
            if self.use_agent:
                self.agent.clear_chat_history()
                self.agent.reset_conversation()
                print("Agent对话历史已清空")
            else:
                self.rag_pipeline.clear_conversation_history()
                print("对话历史已清空")
            
        if args.what == "vectors" or args.what == "all":
            success = self.rag_pipeline.clear_vector_store()
            if success:
                print("向量存储已清空")
            else:
                print("清空向量存储失败")
    
    def _handle_chat(self, args):
        """处理交互式问答命令"""
        print("===== 个人智能问答机器人 =====")
        print(f"当前模式: {'Agent模式' if self.use_agent else '传统RAG模式'}")
        print(f"响应模式: {'流式响应' if self.use_streaming else '非流式响应'}")
        print(f"知识库使用: {'开启' if not getattr(args, 'no_knowledge', False) else '关闭'}")
        print("提示: 输入 'exit' 或 'quit' 退出聊天，输入 'clear' 清空对话历史")
        print("提示: 输入 '!status' 查看对话统计，输入 '!help' 查看更多命令")
        
        # 检查向量存储状态
        vector_count = self.rag_pipeline.get_vector_count()
        if vector_count == 0:
            print("警告: 向量存储为空，回答将不基于本地文档")
        else:
            print(f"当前向量存储中有 {vector_count} 个文档")
        
        # 检查API密钥
        if not global_config.DEEPSEEK_API_KEY:
            print("警告: 未配置DeepSeek API密钥，请在.env文件中设置DEEPSEEK_API_KEY")
        
        print("===============================")
        
        # 交互式聊天的异步函数
        async def chat_loop():
            while True:
                try:
                    query = input("\n你: ")
                    
                    # 处理特殊命令
                    if query.lower() in ["exit", "quit", "退出"]:
                        print("再见！")
                        break
                    elif query.lower() in ["clear", "清空"]:
                        if self.use_agent:
                            self.agent.clear_chat_history()
                            print("对话历史已清空")
                        else:
                            self.rag_pipeline.clear_conversation_history()
                            print("对话历史已清空")
                        continue
                    elif query == "!status":
                        stats = self.agent.get_conversation_stats() if self.use_agent else {}
                        print("\n===== 对话统计 =====")
                        print(f"总响应数: {stats.get('total_responses', 0)}")
                        print(f"总搜索查询: {stats.get('total_search_queries', 0)}")
                        print(f"总工具使用: {stats.get('total_tools_used', 0)}")
                        print("=================")
                        continue
                    elif query == "!help":
                        self._show_chat_help()
                        continue
                    elif query.startswith("!switch-"):
                        if query == "!switch-agent":
                            self.use_agent = True
                            print("已切换到Agent模式")
                        elif query == "!switch-rag":
                            self.use_agent = False
                            print("已切换到传统RAG模式")
                        elif query == "!switch-stream":
                            self.use_streaming = True
                            print("已切换到流式响应模式")
                        elif query == "!switch-nostream":
                            self.use_streaming = False
                            print("已切换到非流式响应模式")
                        continue
                    
                    use_knowledge = not getattr(args, 'no_knowledge', False)
                    
                    if self.use_agent:
                        print("机器人: 正在思考...")
                        if self.use_streaming:
                            # 流式响应
                            print("机器人: ", end="", flush=True)
                            async for chunk in self.agent.generate_streaming_response(query, use_knowledge=use_knowledge):
                                print(chunk, end="", flush=True)
                            print()  # 换行
                        else:
                            # 非流式响应
                            answer = self.agent.generate_response(query, use_knowledge=use_knowledge)
                            print(f"机器人: {answer}")
                    else:
                        # 使用传统rag_pipeline
                        print("机器人: 正在思考...")
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
        
        # 运行异步聊天循环
        asyncio.run(chat_loop())
    
    def _handle_tools(self, args):
        """处理工具命令"""
        if args.tool_command == "create-index":
            result = self.agent._create_index(args.name)
            print(result)
        elif args.tool_command == "switch-index":
            result = self.agent._switch_index(args.name)
            print(result)
        elif args.tool_command == "optimize-index":
            result = self.agent._optimize_index(getattr(args, "name", None))
            print(result)
        elif args.tool_command == "stats":
            result = self.agent._get_knowledge_stats()
            print(result)
        else:
            print("未知的工具命令，请使用 --help 查看支持的命令")
    
    def _show_chat_help(self):
        """显示聊天帮助信息"""
        print("\n===== 聊天命令帮助 =====")
        print("exit/quit/退出 - 退出聊天")
        print("clear/清空 - 清空对话历史")
        print("!status - 查看对话统计信息")
        print("!help - 显示此帮助信息")
        print("!switch-agent - 切换到Agent模式")
        print("!switch-rag - 切换到传统RAG模式")
        print("!switch-stream - 切换到流式响应模式")
        print("!switch-nostream - 切换到非流式响应模式")
        print("\n此外，您可以直接使用以下指令与知识库交互:")
        print("搜索[关键词] - 在知识库中搜索内容")
        print("统计信息 - 查看知识库统计")
        print("=======================")

# 创建命令行界面实例
cli = PersonalRobotCLI()

if __name__ == "__main__":
    cli.run()