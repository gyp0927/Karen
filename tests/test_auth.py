"""认证系统测试"""

import pytest


def test_auth_db_initialization():
    """认证数据库在模块导入时已自动初始化。"""
    from core.auth import _DB_PATH
    import os

    assert os.path.exists(_DB_PATH)


def test_password_hashing():
    from core.auth import _hash_password, _check_password, _BCRYPT_AVAILABLE

    if not _BCRYPT_AVAILABLE:
        pytest.skip("bcrypt 未安装")

    password = "test_password_123"
    hashed = _hash_password(password)
    assert _check_password(password, hashed)
    assert not _check_password("wrong_password", hashed)


def test_derive_key():
    from core.auth import _derive_key
    import hashlib

    salt = b"test_salt_123456"
    key1 = _derive_key("api_key_1", salt)
    key2 = _derive_key("api_key_1", salt)
    key3 = _derive_key("api_key_2", salt)

    assert key1 == key2
    assert key1 != key3
    assert len(bytes.fromhex(key1)) == 32  # SHA256 digest size


def test_rate_limit():
    from core.auth import _check_rate_limit, _record_auth_failure, _auth_failures

    ip = "127.0.0.1"
    # 清理该 IP 的历史记录
    _auth_failures.pop(ip, None)

    assert _check_rate_limit(ip) is True

    # 模拟 4 次失败（<_AUTH_RATE_MAX=5 仍允许）
    for _ in range(4):
        _record_auth_failure(ip)

    assert _check_rate_limit(ip) is True

    # 第 5 次失败后达到上限，应该被限流
    _record_auth_failure(ip)
    assert _check_rate_limit(ip) is False

    # 清理
    _auth_failures.pop(ip, None)
