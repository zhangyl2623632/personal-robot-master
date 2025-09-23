import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置向量存储路径
VECTOR_STORE_PATH = os.environ.get("VECTOR_STORE_PATH", "./vector_store")

if __name__ == "__main__":
    print("验证向量数据库清理结果...")
    print(f"向量存储路径: {VECTOR_STORE_PATH}")
    
    # 检查目录是否存在
    if os.path.exists(VECTOR_STORE_PATH):
        print("向量存储目录存在")
        
        # 检查目录内容
        contents = os.listdir(VECTOR_STORE_PATH)
        print(f"目录内容数量: {len(contents)}")
        
        if len(contents) == 0:
            print("✓ 向量存储目录为空，清理成功！")
        else:
            print("向量存储目录包含以下文件/目录:")
            for item in contents:
                item_path = os.path.join(VECTOR_STORE_PATH, item)
                if os.path.isfile(item_path):
                    print(f"  - 文件: {item} ({os.path.getsize(item_path)} 字节)")
                else:
                    print(f"  - 目录: {item}")
            print("✓ 向量存储已被重置，可能包含新的元数据文件")
    else:
        print("✗ 向量存储目录不存在，请检查路径是否正确")
    
    # 检查备份目录
    backup_dirs = [d for d in os.listdir('.') if d.startswith('vector_store_backup_')]
    if backup_dirs:
        print(f"找到 {len(backup_dirs)} 个备份目录")
        # 显示最近的备份目录
        backup_dirs.sort(reverse=True)
        recent_backup = backup_dirs[0]
        backup_path = os.path.join('.', recent_backup)
        print(f"最近的备份: {recent_backup}")
        
        # 检查备份目录大小
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(backup_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        print(f"备份大小: {total_size / (1024 * 1024):.2f} MB")
    else:
        print("未找到备份目录")