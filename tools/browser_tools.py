"""浏览器控制工具 —— 用 Playwright 实现网页自动化

支持导航、点击、填表、截图、获取页面文本等操作。
Playwright 需要已安装浏览器（chromium）。
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_HOME_DIR = Path.home()
_SCREENSHOT_DIR = _HOME_DIR / ".karen" / "screenshots"

# 全局浏览器实例缓存（延迟初始化）
_browser_ctx = None
_playwright = None


def _ensure_screenshot_dir() -> Path:
    """确保截图目录存在。"""
    _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return _SCREENSHOT_DIR


def _is_safe_url(url: str) -> bool:
    """检查 URL 是否安全（只允许 http/https）。"""
    if not url:
        return False
    url_lower = url.lower().strip()
    return url_lower.startswith("http://") or url_lower.startswith("https://")


async def _get_browser_context():
    """获取或创建全局 BrowserContext。"""
    global _browser_ctx, _playwright
    if _browser_ctx is not None:
        return _browser_ctx

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError("Playwright 未安装，请运行: pip install playwright && playwright install chromium")

    _playwright = await async_playwright().start()
    browser = await _playwright.chromium.launch(headless=True)
    _browser_ctx = await browser.new_context(
        viewport={"width": 1280, "height": 720},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    logger.info("[Browser] Playwright chromium 启动成功")
    return _browser_ctx


async def browser_control(
    action: str,
    url: str = "",
    selector: str = "",
    text: str = "",
    wait_ms: int = 1000,
) -> str:
    """控制浏览器执行操作。

    Args:
        action: 操作类型 - navigate(导航), click(点击), fill(填表), screenshot(截图), get_text(获取文本)
        url: 导航目标 URL（action=navigate 时必填）
        selector: CSS 选择器（click/fill/get_text 时必填）
        text: 要填入的文本（action=fill 时必填）
        wait_ms: 操作后等待毫秒数，默认 1000

    Returns:
        操作结果或页面文本/截图路径
    """
    action = action.lower().strip()

    if action not in {"navigate", "click", "fill", "screenshot", "get_text"}:
        return f"[错误: 不支持的操作 '{action}'，支持: navigate, click, fill, screenshot, get_text]"

    if action == "navigate" and not url:
        return "[错误: navigate 操作需要提供 url]"

    if action in {"click", "fill", "get_text"} and not selector:
        return f"[错误: {action} 操作需要提供 selector]"

    if action == "navigate" and not _is_safe_url(url):
        return "[错误: URL 只允许 http:// 或 https:// 协议]"

    try:
        ctx = await _get_browser_context()
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        if action == "navigate":
            await page.goto(url, timeout=15000, wait_until="domcontentloaded")
            if wait_ms > 0:
                await page.wait_for_timeout(wait_ms)
            title = await page.title()
            return f"导航成功: {title}\nURL: {page.url}"

        if action == "click":
            await page.click(selector, timeout=10000)
            if wait_ms > 0:
                await page.wait_for_timeout(wait_ms)
            return f"点击成功: {selector}"

        if action == "fill":
            await page.fill(selector, text, timeout=10000)
            if wait_ms > 0:
                await page.wait_for_timeout(wait_ms)
            return f"填表成功: {selector} = {text[:50]}{'...' if len(text) > 50 else ''}"

        if action == "screenshot":
            _ensure_screenshot_dir()
            # 使用时间戳命名
            import time
            filename = f"screenshot_{int(time.time() * 1000)}.png"
            filepath = _SCREENSHOT_DIR / filename
            await page.screenshot(path=str(filepath), full_page=True)
            return f"截图已保存: {filepath}"

        if action == "get_text":
            # 获取可见文本
            body_text = await page.inner_text("body")
            # 清理多余空白
            lines = [ln.strip() for ln in body_text.split("\n") if ln.strip()]
            result = "\n".join(lines[:100])  # 最多 100 行
            if len(lines) > 100:
                result += "\n\n... (文本已截断)"
            return result

    except Exception as e:
        logger.warning(f"[Browser] {action} 失败: {e}")
        return f"[浏览器操作失败: {e}]"


async def close_browser() -> None:
    """关闭浏览器实例（用于清理）。"""
    global _browser_ctx, _playwright
    if _browser_ctx:
        await _browser_ctx.close()
        _browser_ctx = None
    if _playwright:
        await _playwright.stop()
        _playwright = None
    logger.info("[Browser] 浏览器已关闭")
