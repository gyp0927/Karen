"""
桌面应用入口 - 无控制台窗口
"""
import sys
import os
import socket
import subprocess

# 获取程序所在目录
if getattr(sys, 'frozen', False):
    app_dir = os.path.dirname(sys.executable)
else:
    app_dir = os.path.dirname(os.path.abspath(__file__))

# 切换到程序目录，确保能找到 .env 和静态文件
os.chdir(app_dir)


def check_and_install_deps():
    """检查依赖是否已安装，缺失时提示用户手动安装。"""
    required = {
        "flask": "flask>=3.0.0",
        "flask_socketio": "flask-socketio>=5.3.0",
        "flask_cors": "flask-cors>=4.0.0",
        "langchain_openai": "langchain-openai>=0.2.0",
        "langgraph": "langgraph>=0.2.0",
        "dotenv": "python-dotenv>=1.0.0",
    }
    missing = []
    for module, pkg in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("[ERROR] 缺少以下依赖，请手动安装:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("[INFO] 运行: pip install -r requirements.txt")
        input("按 Enter 退出...")
        sys.exit(1)

    # 检查 numpy (optional, for RAG)
    try:
        import numpy  # noqa
    except ImportError:
        print("[WARN] 缺少 numpy，RAG 知识库功能将被禁用。")
        print("[INFO] 运行: pip install numpy 以启用该功能。")

    # 检查 mcp (optional, for MCP servers)
    try:
        import mcp  # noqa
    except ImportError:
        print("[WARN] 缺少 mcp，MCP 工具集成功能将被禁用。")
        print("[INFO] 运行: pip install 'mcp>=1.20.0' 以启用该功能。")


check_and_install_deps()


def is_port_in_use(port=5000):
    """检查端口是否已被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def check_single_instance():
    """检查是否已有实例在运行"""
    lock_file = os.path.join(app_dir, '.app.lock')
    try:
        # 尝试创建锁文件（如果不存在则创建，存在则报错）
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def cleanup_lock():
    """清理锁文件"""
    lock_file = os.path.join(app_dir, '.app.lock')
    try:
        if os.path.exists(lock_file):
            os.remove(lock_file)
    except Exception:
        pass


# 检查是否已有实例在运行
if not check_single_instance():
    if is_port_in_use():
        print("检测到已有实例在运行，正在打开浏览器...")
        import webbrowser
        webbrowser.open("http://127.0.0.1:5000")
        sys.exit(0)
    else:
        # 锁文件残留，清理后重新尝试
        cleanup_lock()
        if not check_single_instance():
            if is_port_in_use():
                print("检测到已有实例在运行，正在打开浏览器...")
                import webbrowser
                webbrowser.open("http://127.0.0.1:5000")
                sys.exit(0)

# 注册退出时清理锁文件
import atexit
atexit.register(cleanup_lock)

# 隐藏控制台窗口（Windows）——只在非交互式环境下（如打包后的exe）
if sys.platform == 'win32' and not sys.stdout.isatty():
    import ctypes
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

# 导入并启动Flask应用
from web.app import app, init_agents, socketio

if __name__ == "__main__":
    try:
        print("Initializing agents...")
        init_agents()
        print("Agents initialized!")
        print("Starting 凯伦 Desktop App...")
        print("Open http://127.0.0.1:5000/config in your browser")
        print("LAN access: http://0.0.0.0:5000")
        import webbrowser
        webbrowser.open("http://127.0.0.1:5000/")
        socketio.run(app, host="0.0.0.0", port=5000, debug=False)
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
        cleanup_lock()
        sys.exit(1)
