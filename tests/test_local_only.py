"""LOCAL_ONLY 中间件测试"""

import pytest
from flask import Flask


@pytest.fixture
def app_with_local_only():
    """创建带 LOCAL_ONLY 中间件的 Flask 应用。"""
    app = Flask(__name__)
    app.config["TESTING"] = True

    # 模拟 web/app.py 中的 LOCAL_ONLY 逻辑
    LOCAL_ONLY_PREFIXES = ["/config", "/api/config", "/api/auth/users"]

    def _is_local_address(remote: str) -> bool:
        if not remote:
            return False
        if remote == "localhost":
            return True
        import ipaddress

        try:
            addr = ipaddress.ip_address(remote)
        except ValueError:
            return False
        if addr.is_loopback:
            return True
        mapped = getattr(addr, "ipv4_mapped", None)
        if mapped is not None and mapped.is_loopback:
            return True
        return False

    @app.before_request
    def restrict_local_only():
        from flask import request

        path = request.path
        for prefix in LOCAL_ONLY_PREFIXES:
            if path.startswith(prefix):
                remote = request.remote_addr
                if not _is_local_address(remote):
                    return "Access denied: configuration is local only", 403

    @app.route("/api/config")
    def config():
        return {"success": True}

    @app.route("/api/health")
    def health():
        return {"status": "ok"}

    return app


@pytest.fixture
def client(app_with_local_only):
    return app_with_local_only.test_client()


def test_local_access_config(client):
    """从 127.0.0.1 访问 /api/config 应返回 200"""
    response = client.get("/api/config", environ_overrides={"REMOTE_ADDR": "127.0.0.1"})
    assert response.status_code == 200


def test_non_local_access_config_denied(client):
    """从 192.168.1.1 访问 /api/config 应返回 403"""
    response = client.get("/api/config", environ_overrides={"REMOTE_ADDR": "192.168.1.1"})
    assert response.status_code == 403
    assert b"local only" in response.data


def test_non_local_access_health_allowed(client):
    """从 192.168.1.1 访问 /api/health 应返回 200（不在 LOCAL_ONLY 列表中）"""
    response = client.get("/api/health", environ_overrides={"REMOTE_ADDR": "192.168.1.1"})
    assert response.status_code == 200


def test_ipv6_loopback_allowed(client):
    """IPv6 loopback ::1 应被允许"""
    response = client.get("/api/config", environ_overrides={"REMOTE_ADDR": "::1"})
    assert response.status_code == 200
