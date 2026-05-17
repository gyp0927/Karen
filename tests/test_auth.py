"""认证系统测试"""

import pytest


def test_auth_db_initialization():
    """认证数据库在模块导入时已自动初始化。"""
    import os

    from core.auth import _DB_PATH

    assert os.path.exists(_DB_PATH)


def test_password_hashing():
    from core.auth import _BCRYPT_AVAILABLE, _check_password, _hash_password

    if not _BCRYPT_AVAILABLE:
        pytest.skip("bcrypt 未安装")

    password = "test_password_123"
    hashed = _hash_password(password)
    assert _check_password(password, hashed)
    assert not _check_password("wrong_password", hashed)


def test_derive_key():

    from core.auth import _derive_key

    salt = b"test_salt_123456"
    key1 = _derive_key("api_key_1", salt)
    key2 = _derive_key("api_key_1", salt)
    key3 = _derive_key("api_key_2", salt)

    assert key1 == key2
    assert key1 != key3
    assert len(bytes.fromhex(key1)) == 32  # SHA256 digest size


def test_rate_limit():
    from core.auth import _auth_failures, _check_rate_limit, _record_auth_failure

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


def test_password_hashing_empty_password():
    """空密码应正确哈希和校验"""
    from core.auth import _BCRYPT_AVAILABLE, _check_password, _hash_password

    if not _BCRYPT_AVAILABLE:
        pytest.skip("bcrypt 未安装")

    hashed = _hash_password("")
    assert _check_password("", hashed)
    assert not _check_password("nonempty", hashed)


def test_derive_key_empty_string():
    """空字符串 key 应返回有效结果"""
    from core.auth import _derive_key

    salt = b"test_salt_123456"
    key = _derive_key("", salt)
    assert len(key) == 64  # SHA256 hex = 64 chars
    assert key != ""  # 空输入不应输出空


def test_rate_limit_expired_cleanup():
    """过期记录应被清理"""
    from core.auth import _auth_failures, _check_rate_limit, _record_auth_failure

    ip = "127.0.0.99"
    _auth_failures.pop(ip, None)

    # 记录一次失败
    _record_auth_failure(ip)
    assert _check_rate_limit(ip) is True

    # 手动将记录设为过期
    import time
    old_time = time.time() - 3601  # 超过 BLOCK 窗口
    _auth_failures[ip] = [old_time for _ in range(10)]

    # 过期记录不应影响当前判断
    assert _check_rate_limit(ip) is True

    _auth_failures.pop(ip, None)


def test_rate_limit_multiple_ips():
    """不同 IP 应独立计数"""
    from core.auth import _auth_failures, _check_rate_limit, _record_auth_failure

    ip1 = "127.0.0.10"
    ip2 = "127.0.0.11"
    _auth_failures.pop(ip1, None)
    _auth_failures.pop(ip2, None)

    # ip1 失败 5 次
    for _ in range(5):
        _record_auth_failure(ip1)
    assert _check_rate_limit(ip1) is False
    assert _check_rate_limit(ip2) is True  # ip2 不受影响

    _auth_failures.pop(ip1, None)
    _auth_failures.pop(ip2, None)
