"""联网搜索工具 - 为 Researcher Agent 提供搜索能力。

搜索源优先级：
1. duckduckgo-search 库（最稳定，已安装）
2. 360 搜索（国内可访问，无需代理）
3. Bing 搜索（fallback）

代理配置：通过环境变量 HTTP_PROXY / HTTPS_PROXY 设置。
"""

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)

# 超时配置
_SEARCH_TIMEOUT = 15
_FETCH_TIMEOUT = 8
_MAX_FETCH_WORKERS = 3
_MAX_RETRIES = 2

# ========== HTTP 客户端（使用 requests）==========

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


def _get_session() -> "requests.Session | None":
    """创建带统一配置的 requests Session。"""
    if not _HAS_REQUESTS:
        return None
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    })
    # 自动检测系统代理
    proxies = {}
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
    if proxies:
        session.proxies.update(proxies)
        logger.info(f"Using proxy: {proxies}")
    return session


_session = _get_session()


def _fetch(url: str, timeout: int = _FETCH_TIMEOUT) -> str:
    """获取网页内容。优先用 requests，回退到 urllib。"""
    if _session:
        try:
            resp = _session.get(url, timeout=timeout, verify=False)
            resp.raise_for_status()
            return resp.text
        except Exception:
            pass  # 回退到 urllib

    import ssl
    import urllib.request
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
        return resp.read().decode("utf-8", errors="ignore")


# ========== 缓存 ==========

_search_cache: dict[str, tuple[float, list[dict]]] = {}
_SEARCH_CACHE_TTL = 300


def _get_cached_search(query: str) -> list[dict] | None:
    if query in _search_cache:
        ts, results = _search_cache[query]
        if time.time() - ts < _SEARCH_CACHE_TTL:
            logger.debug(f"Search cache hit: '{query}'")
            return results
    return None


def _set_cached_search(query: str, results: list[dict]):
    _search_cache[query] = (time.time(), results)


# ========== 解析通用函数 ==========

def _dedupe_results(results: list[dict], max_results: int) -> list[dict]:
    seen = set()
    out = []
    for r in results:
        key = r.get("href", "")
        if key and key not in seen:
            seen.add(key)
            out.append(r)
        if len(out) >= max_results:
            break
    return out


# ========== 1. duckduckgo-search 库 ==========

def _try_ddg_library(query: str, max_results: int) -> list[dict] | None:
    """使用 duckduckgo-search 库搜索。"""
    try:
        from duckduckgo_search import DDGS
        proxies = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        with DDGS(timeout=_SEARCH_TIMEOUT, proxy=proxies) as ddgs:
            raw = ddgs.text(query, max_results=max_results * 2)
            results = []
            for item in raw:
                title = item.get("title", "").strip()
                href = item.get("href", "").strip()
                if title and href and href.startswith("http"):
                    results.append({"title": title, "href": href})
                if len(results) >= max_results:
                    break
            return results if results else None
    except Exception as e:
        logger.debug(f"ddg library failed: {e}")
        return None


# ========== 2. 360 搜索（国内可访问）==========

def _parse_360_html(html: str, max_results: int) -> list[dict]:
    """从 360 搜索 HTML 中解析结果。"""
    results = []
    # 360 结果：<h3><a href="...">标题</a></h3>
    patterns = [
        r'<h3[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?</h3>',
        r'<a[^>]+data-url="([^"]+)"[^>]*>(.*?)</a>',
        r'<a[^>]+href="([^"]+)"[^>]*class="[^"]*result[^"]*"[^>]*>(.*?)</a>',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, html, re.IGNORECASE | re.DOTALL):
            href = match.group(1)
            title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            # 过滤广告和无效链接
            if not title or len(title) < 5:
                continue
            if "广告" in title or "推广" in title:
                continue
            # 跳过 360 的跳转链接（重定向到 CSDN 等，抓取困难）
            if "so.com/link" in href:
                continue
            if href.startswith("//"):
                href = "https:" + href
            if title and href and href.startswith("http"):
                results.append({"title": title, "href": href})
            if len(results) >= max_results * 2:
                break
        if len(results) >= max_results * 2:
            break
    return _dedupe_results(results, max_results)


def _so_search(query: str, max_results: int) -> list[dict] | None:
    """使用 360 搜索（国内无需代理）。"""
    import urllib.parse
    encoded = urllib.parse.quote(query)
    url = f"https://www.so.com/s?q={encoded}"
    html = _fetch(url, timeout=_SEARCH_TIMEOUT)
    results = _parse_360_html(html, max_results)
    return results if results else None


# ========== 3. Bing 搜索 ==========

def _parse_bing_html(html: str, max_results: int) -> list[dict]:
    results = []
    patterns = [
        r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?</li>',
        r'<h2[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?</h2>',
        r'<a[^>]+href="([^"]+)"[^>]*target="_blank"[^>]*>(.*?)</a>',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, html, re.IGNORECASE | re.DOTALL):
            href = match.group(1)
            if href.startswith("/"):
                href = "https://www.bing.com" + href
            title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            if title in ("缓存", "相似", "Cached", "Similar", ""):
                continue
            if "microsoft" in href.lower() and "bing" in href.lower():
                continue
            if title and href and href.startswith("http"):
                results.append({"title": title, "href": href})
            if len(results) >= max_results * 2:
                break
        if len(results) >= max_results * 2:
            break
    return _dedupe_results(results, max_results)


def _bing_search(query: str, max_results: int) -> list[dict] | None:
    import urllib.parse
    encoded = urllib.parse.quote(query)
    url = f"https://www.bing.com/search?q={encoded}&setmkt=en-US&setlang=en"
    html = _fetch(url, timeout=_SEARCH_TIMEOUT)
    results = _parse_bing_html(html, max_results)
    return results if results else None


# ========== 统一搜索入口 ==========

def duckduckgo_search(query: str, max_results: int = 2) -> list[dict]:
    """搜索入口，带多源 fallback 和重试。

    搜索优先级：
    1. duckduckgo-search 库（最稳定）
    2. 360 搜索（国内无需代理）
    3. Bing 搜索（fallback）
    """
    cached = _get_cached_search(query)
    if cached is not None:
        return cached

    last_error = None
    sources = [
        ("ddg_library", _try_ddg_library),
        ("360_search", _so_search),
        ("bing", _bing_search),
    ]

    for source_name, source_fn in sources:
        for attempt in range(_MAX_RETRIES):
            try:
                results = source_fn(query, max_results)
                if results:
                    logger.info(
                        f"Search '{query}' via {source_name}: {len(results)} results"
                    )
                    _set_cached_search(query, results)
                    return results
                logger.debug(
                    f"Source {source_name} returned empty for '{query}'"
                )
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Search source={source_name} "
                    f"attempt={attempt + 1}/{_MAX_RETRIES} "
                    f"failed for '{query}': {e}"
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(0.5 * (attempt + 1))

    logger.warning(
        f"All search sources failed for query: {query} - {last_error}"
    )
    return []


# ========== 网页抓取 ==========

def fetch_page_content(url: str, max_chars: int = 1000) -> str:
    """获取网页内容。超时 8 秒。"""
    try:
        html = _fetch(url, timeout=_FETCH_TIMEOUT)
        text = re.sub(
            r'<(script|style)[^>]*>.*?</\1>',
            '',
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[网页内容已截断]"
        return text
    except Exception as e:
        logger.warning(f"Failed to fetch page: {url} - {e}")
        return "[无法获取网页内容]"


def _fetch_all_pages(urls: list[str], max_chars: int = 1000) -> dict[str, str]:
    contents: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=_MAX_FETCH_WORKERS) as executor:
        futures = {
            executor.submit(fetch_page_content, url, max_chars): url
            for url in urls
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                contents[url] = future.result()
            except Exception as e:
                logger.warning(f"Parallel fetch failed for {url}: {e}")
                contents[url] = "[无法获取网页内容]"
    return contents


def search_and_summarize(query: str, max_results: int = 2, fetch_content: bool = True) -> str:
    """搜索并返回格式化的结果文本（供 LLM 使用）。"""
    results = duckduckgo_search(query, max_results=max_results)
    if not results:
        return "[搜索未找到相关结果]"

    page_contents = {}
    if fetch_content:
        urls = [r["href"] for r in results]
        page_contents = _fetch_all_pages(urls, max_chars=1000)

    output_lines = [f"## 搜索结果：'{query}'\n"]
    for i, r in enumerate(results, 1):
        output_lines.append(f"**[{i}]** {r['title']}")
        output_lines.append(f"链接: {r['href']}")
        if fetch_content:
            content = page_contents.get(r["href"], "[无法获取网页内容]")
            output_lines.append(f"摘要: {content}\n")
        else:
            output_lines.append("")

    return "\n".join(output_lines)


# ========== 直接测试 ==========
if __name__ == "__main__":
    import asyncio

    async def _test():
        logging.basicConfig(level=logging.INFO)
        queries = [
            "今天沈阳的天气",
            "Python 编程语言介绍",
            "2024年最新科技新闻",
        ]
        for q in queries:
            print(f"\n{'='*60}")
            print(f"查询: {q}")
            print("=" * 60)
            result = search_and_summarize(q, max_results=2)
            print(result)
            time.sleep(1)

    asyncio.run(_test())
