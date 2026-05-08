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
    """检查并安装缺失的依赖"""
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
        print("[INFO] Installing missing dependencies...")
        print(f"[INFO] Packages: {missing}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing)
            print("[INFO] Dependencies installed successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to install dependencies: {e}")
            print("[ERROR] Please run: pip install -r requirements.txt")
            input("Press Enter to exit...")
            sys.exit(1)

    # 检查 numpy (optional, for RAG)
    try:
        import numpy  # noqa
    except ImportError:
        print("[INFO] Installing numpy (required for RAG knowledge base)...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "numpy"])
            print("[INFO] numpy installed.")
        except Exception:
            print("[WARN] numpy installation failed. RAG features will be disabled.")

    # 检查 mcp (optional, for MCP servers)
    try:
        import mcp  # noqa
    except ImportError:
        print("[INFO] Installing mcp (required for MCP tool integration)...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "mcp>=1.20.0"])
            print("[INFO] mcp installed.")
        except Exception:
            print("[WARN] mcp installation failed. MCP features will be disabled.")


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
    except:
        pass


# 检查是否已有实例在运行
if not check_single_instance():
    if is_port_in_use():
        print("检测到已有实例在运行，正在打开浏览器...")
        import webbrowser
        webbrowser.open("http://127.0.0.1:5000")
        sys.exit(0)
    else:
        # 锁文件残留，清理后重新创建
        cleanup_lock()
        check_single_instance()

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
