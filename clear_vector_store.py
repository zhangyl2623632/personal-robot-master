import os
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.vector_store import vector_store_manager

if __name__ == "__main__":
    print("开始清理向量数据库...")
    
    # 获取当前向量数量
    current_count = vector_store_manager.get_vector_count()
    print(f"清理前向量数量: {current_count}")
    
    # 执行清理操作
    success = vector_store_manager.clear_vector_store()
    
    if success:
        # 获取清理后的向量数量
        new_count = vector_store_manager.get_vector_count()
        print(f"向量数据库清理成功！")
        print(f"清理后向量数量: {new_count}")
    else:
        print("向量数据库清理失败，请查看日志获取更多信息")