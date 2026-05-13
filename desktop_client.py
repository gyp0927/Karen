"""
独立窗口桌面客户端 - 使用系统 WebView2 渲染网页
不需要额外安装浏览器
"""
import os
import sys
import socket
import threading
import time
import atexit
import subprocess

# 获取程序所在目录
if getattr(sys, 'frozen', False):
    app_dir = os.path.dirname(sys.executable)
else:
    app_dir = os.path.dirname(os.path.abspath(__file__))
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
        "webview": "pywebview>=5.0",
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


def get_resource_path(filename):
    """获取资源文件路径（兼容开发环境和 PyInstaller 打包后）"""
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后，资源在 _MEIPASS 临时目录
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(app_dir, filename)


def is_port_in_use(port=5000):
    """检查端口是否已被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def check_single_instance():
    """检查是否已有实例在运行"""
    lock_file = os.path.join(app_dir, '.app.lock')
    try:
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


def find_free_port(start=5000):
    """查找可用端口"""
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start


def start_flask_server(port):
    """在后台线程启动 Flask 服务器"""
    # 设置 CORS 白名单，包含实际使用的端口（避免 Socket.IO origin 被拒绝）
    origin = f"http://127.0.0.1:{port}"
    existing = os.getenv("CORS_ORIGINS", "")
    if origin not in existing:
        os.environ["CORS_ORIGINS"] = f"{existing},{origin}" if existing else origin

    sys.path.insert(0, app_dir)
    from web.app import app, init_agents

    try:
        init_agents()
        print(f"[Server] Flask server starting on port {port}...")
    except Exception as e:
        print(f"[Server] init_agents failed: {e}")

    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    # 使用 threading 模式（Socket.IO 走长轮询，token 流式照常工作）
    from flask_socketio import SocketIO
    from web.app import socketio
    socketio.run(app, host="127.0.0.1", port=port, debug=False, use_reloader=False)


def wait_for_server(port, timeout=30):
    """等待服务器就绪"""
    start = time.time()
    while time.time() - start < timeout:
        if is_port_in_use(port):
            return True
        time.sleep(0.2)
    return False


def main():
    # 单实例检查
    if not check_single_instance():
        if is_port_in_use():
            print("检测到已有实例在运行，退出...")
            sys.exit(0)
        else:
            cleanup_lock()
            if not check_single_instance():
                print("检测到已有实例在运行，退出...")
                sys.exit(0)

    atexit.register(cleanup_lock)

    # 隐藏控制台窗口（Windows）
    if sys.platform == 'win32' and not sys.stdout.isatty():
        import ctypes
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

    # 查找可用端口
    port = 5000
    if is_port_in_use(port):
        port = find_free_port(port + 1)

    # 启动 Flask 服务器线程
    server_thread = threading.Thread(target=start_flask_server, args=(port,), daemon=True)
    server_thread.start()

    # 等待服务器就绪
    if not wait_for_server(port):
        print("[Error] 服务器启动超时")
        sys.exit(1)

    print(f"[Client] Server ready at http://127.0.0.1:{port}")

    # WebView2 缓存策略:按 script.js 哈希命中。
    # 旧实现每次启动都 rmtree 整个缓存,前端没改也会清空 cookie/localStorage,
    # 用户每次启动都得重新登录。改成仅当前端 JS 变化时才清。
    import hashlib
    webview_data_dir = os.path.join(os.getenv("LOCALAPPDATA", ""), "Karen", "WebView2")
    script_path = os.path.join(app_dir, "web", "static", "script.js")
    hash_file = os.path.join(webview_data_dir, ".script_hash")
    try:
        with open(script_path, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
    except OSError:
        current_hash = ""
    cached_hash = ""
    if os.path.exists(hash_file):
        try:
            with open(hash_file, "r", encoding="utf-8") as f:
                cached_hash = f.read().strip()
        except OSError:
            pass
    if current_hash and current_hash != cached_hash and os.path.exists(webview_data_dir):
        import shutil
        try:
            shutil.rmtree(webview_data_dir)
            print(f"[Client] script.js changed, cleared WebView2 cache")
        except Exception as e:
            print(f"[Client] Warning: could not clear cache: {e}")
    os.makedirs(webview_data_dir, exist_ok=True)
    if current_hash:
        try:
            with open(hash_file, "w", encoding="utf-8") as f:
                f.write(current_hash)
        except OSError:
            pass
    os.environ["WEBVIEW2_USER_DATA_FOLDER"] = webview_data_dir

    # 启动 WebView 窗口
    try:
        import webview
    except ImportError:
        print("[Error] 缺少 pywebview，请运行: pip install pywebview")
        input("按 Enter 退出...")
        sys.exit(1)

    # 创建窗口
    window = webview.create_window(
        title='凯伦',
        url=f'http://127.0.0.1:{port}',
        width=1400,
        height=900,
        min_size=(900, 600),
        text_select=True,
    )

    # 窗口关闭时退出
    def on_closing():
        print("[Client] Window closing, shutting down...")
        cleanup_lock()
        os._exit(0)

    window.events.closing += on_closing

    # 启动 GUI（阻塞）
    webview.start(
        debug=False,
        http_server=False,
        gui='edgechromium',  # Windows 使用 Edge WebView2
    )


if __name__ == '__main__':
    main()
