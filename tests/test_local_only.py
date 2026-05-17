"""LOCAL_ONLY 中间件测试 — 直接测试 web/app.py 中的真实实现。"""

import pytest


@pytest.fixture
def client():
    from web.app import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_local_access_config(client):
    """从 127.0.0.1 访问 /api/config 应返回 200"""
    response = client.get("/api/config", environ_overrides={"REMOTE_ADDR": "127.0.0.1"})
    assert response.status_code == 200


def test_non_local_access_config_denied(client):
    """从 192.168.1.1 访问 /api/config 应返回 403"""
    response = client.get("/api/config", environ_overrides={"REMOTE_ADDR": "192.168.1.1"})
    assert response.status_code == 403
    assert b"local only" in response.data


def test_non_local_access_unprotected_route(client):
    """从 192.168.1.1 访问 /api/stats 应返回 200（不在 LOCAL_ONLY 列表中）"""
    response = client.get("/api/stats", environ_overrides={"REMOTE_ADDR": "192.168.1.1"})
    # /api/stats 可能需要认证，但至少不应被 LOCAL_ONLY 拦截为 403
    assert response.status_code != 403


def test_ipv6_loopback_allowed(client):
    """IPv6 loopback ::1 应被允许"""
    response = client.get("/api/config", environ_overrides={"REMOTE_ADDR": "::1"})
    assert response.status_code == 200


def test_ipv4_mapped_ipv6_loopback(client):
    """IPv4-mapped IPv6 loopback ::ffff:127.0.0.1 应被允许"""
    response = client.get("/api/config", environ_overrides={"REMOTE_ADDR": "::ffff:127.0.0.1"})
    assert response.status_code == 200
