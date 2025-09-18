import os
import re
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VersionManager:
    """版本管理器，实现版本号的读取和更新功能"""
    
    def __init__(self):
        """初始化版本管理器"""
        self.readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'README.md')
        self.current_version = None
        self.load_version()
    
    def load_version(self):
        """从README.md文件中加载当前版本号"""
        if not os.path.exists(self.readme_path):
            logger.error(f"README.md文件不存在: {self.readme_path}")
            return
        
        try:
            with open(self.readme_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # 查找版本号格式：版本：vX.Y.Z
                version_match = re.search(r'版本：v(\d+\.\d+\.\d+)', content)
                if version_match:
                    self.current_version = version_match.group(1)
                    logger.info(f"成功加载当前版本号: v{self.current_version}")
                else:
                    logger.warning("未在README.md中找到版本号")
        except Exception as e:
            logger.error(f"读取README.md时出错: {str(e)}")
    
    def increment_version(self, part='patch'):
        """
        增加版本号
        part: 'major' | 'minor' | 'patch'，指定要增加的版本部分
        """
        if not self.current_version:
            logger.error("未加载版本号，无法增加版本")
            return False
        
        try:
            # 解析版本号
            major, minor, patch = map(int, self.current_version.split('.'))
            
            # 根据指定的部分增加版本号
            if part == 'major':
                major += 1
                minor = 0
                patch = 0
            elif part == 'minor':
                minor += 1
                patch = 0
            elif part == 'patch':
                patch += 1
            else:
                logger.error(f"无效的版本部分: {part}")
                return False
            
            # 构建新版本号
            new_version = f"{major}.{minor}.{patch}"
            
            # 更新README.md中的版本号
            if self._update_readme_version(new_version):
                self.current_version = new_version
                logger.info(f"成功更新版本号: v{new_version}")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"增加版本号时出错: {str(e)}")
            return False
    
    def _update_readme_version(self, new_version):
        """更新README.md文件中的版本号"""
        try:
            with open(self.readme_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 替换版本号
            new_content = re.sub(r'版本：v\d+\.\d+\.\d+', f'版本：v{new_version}', content)
            
            with open(self.readme_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            
            return True
        except Exception as e:
            logger.error(f"更新README.md版本号时出错: {str(e)}")
            return False
    
    def get_version(self):
        """获取当前版本号"""
        return self.current_version

# 创建版本管理器实例
version_manager = VersionManager()