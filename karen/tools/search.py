"""联网搜索工具 - 为 Researcher Agent 提供搜索能力。

搜索源优先级：
1. ddgs 库（最稳定，已安装）
2. 360 搜索（国内可访问，无需代理）
3. Bing 搜索（fallback）

代理配置：通过环境变量 HTTP_PROXY / HTTPS_PROXY 设置。
"""

import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)

# 超时配置
_SEARCH_TIMEOUT = 15
_FETCH_TIMEOUT = 8
_MAX_FETCH_WORKERS = 3
_MAX_RETRIES = 2

# 快速模式专用超时（ aggressive —— 快速模式追求速度而非 completeness ）
_FAST_SEARCH_TIMEOUT = 6
_FAST_FETCH_TIMEOUT = 3
_FAST_MAX_RETRIES = 1
_FAST_MAX_FETCH_WORKERS = 2

# CJK Unified Ideographs (U+4E00–U+9FFF)，用于判断 query 是否含中文
_CJK_RE = re.compile(r"[一-鿿]")

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


# 是否禁用 SSL 验证（仅在明确配置时允许，用于代理/自签名证书环境）
_VERIFY_SSL = os.environ.get("SEARCH_VERIFY_SSL", "1").lower() not in ("0", "false", "no")
if not _VERIFY_SSL:
    logger.warning("SSL verification is DISABLED for search requests. "
                   "Set SEARCH_VERIFY_SSL=1 to enable. This is insecure.")


def _fetch(url: str, timeout: int = _FETCH_TIMEOUT) -> str:
    """获取网页内容。优先用 requests，回退到 urllib。"""
    if _session:
        try:
            resp = _session.get(url, timeout=timeout, verify=_VERIFY_SSL)
            resp.raise_for_status()
            return resp.text
        except Exception:
            pass  # 回退到 urllib

    import ssl
    import urllib.request
    ssl_ctx = ssl.create_default_context()
    if not _VERIFY_SSL:
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


# ========== 缓存（带 TTL 和容量上限 + 线程安全锁） ==========

_search_cache: dict[str, tuple[float, list[dict]]] = {}
_SEARCH_CACHE_TTL = 300
_SEARCH_CACHE_MAX_SIZE = 200
_search_cache_lock = threading.Lock()


def _get_cached_search(query: str) -> list[dict] | None:
    with _search_cache_lock:
        if query in _search_cache:
            ts, results = _search_cache[query]
            if time.time() - ts < _SEARCH_CACHE_TTL:
                logger.debug(f"Search cache hit: '{query}'")
                return results
    return None


def _set_cached_search(query: str, results: list[dict]):
    with _search_cache_lock:
        # 超出容量时清理最旧的条目
        if len(_search_cache) >= _SEARCH_CACHE_MAX_SIZE:
            now = time.time()
            # 先清过期项
            expired = [k for k, (ts, _) in _search_cache.items() if now - ts > _SEARCH_CACHE_TTL]
            for k in expired:
                del _search_cache[k]
            # 仍超则清最早的
            if len(_search_cache) >= _SEARCH_CACHE_MAX_SIZE:
                oldest = min(_search_cache, key=lambda k: _search_cache[k][0])
                del _search_cache[oldest]
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
    """使用 ddgs 库搜索。"""
    try:
        from ddgs import DDGS
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

def duckduckgo_search(query: str, max_results: int = 2, fast_mode: bool = False) -> list[dict]:
    """搜索入口，带多源 fallback 和重试。

    搜索优先级：
    - 中文 query：360 → Bing → DDG（DDG 对中文时事查询常返回垃圾/无关结果）
    - 英文 query：DDG → 360 → Bing

    Args:
        fast_mode: 快速模式——减少超时和重试次数，优先返回结果速度。
    """
    cached = _get_cached_search(query)
    if cached is not None:
        return cached

    has_chinese = bool(_CJK_RE.search(query))
    if has_chinese:
        sources = [
            ("360_search", _so_search),
            ("bing", _bing_search),
            ("ddg_library", _try_ddg_library),
        ]
    else:
        sources = [
            ("ddg_library", _try_ddg_library),
            ("360_search", _so_search),
            ("bing", _bing_search),
        ]

    # 快速模式：减少重试，缩短超时
    max_retries = _FAST_MAX_RETRIES if fast_mode else _MAX_RETRIES
    search_timeout = _FAST_SEARCH_TIMEOUT if fast_mode else _SEARCH_TIMEOUT

    last_error = None

    for source_name, source_fn in sources:
        for attempt in range(max_retries):
            try:
                # 快速模式下给 source_fn 传更短超时（通过闭包捕获的全局变量控制）
                if fast_mode:
                    # 临时修改全局超时，source_fn 内部使用这些全局变量
                    orig_timeout = globals().get('_SEARCH_TIMEOUT')
                    globals()['_SEARCH_TIMEOUT'] = search_timeout
                    try:
                        results = source_fn(query, max_results)
                    finally:
                        if orig_timeout is not None:
                            globals()['_SEARCH_TIMEOUT'] = orig_timeout
                else:
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
                    f"attempt={attempt + 1}/{max_retries} "
                    f"failed for '{query}': {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(0.3 * (attempt + 1) if fast_mode else 0.5 * (attempt + 1))

    logger.warning(
        f"All search sources failed for query: {query} - {last_error}"
    )
    return []


# ========== 网页抓取 ==========

def fetch_page_content(url: str, max_chars: int = 1000, timeout: int = _FETCH_TIMEOUT) -> str:
    """获取网页内容。默认超时 8 秒，可通过 timeout 参数覆盖。"""
    try:
        html = _fetch(url, timeout=timeout)
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


def _fetch_all_pages(urls: list[str], max_chars: int = 1000, fast_mode: bool = False) -> dict[str, str]:
    """并行抓取多个页面。

    Args:
        fast_mode: 快速模式——减少 workers 和每个页面的超时，优先速度。
    """
    contents: dict[str, str] = {}
    workers = _FAST_MAX_FETCH_WORKERS if fast_mode else _MAX_FETCH_WORKERS
    fetch_timeout = _FAST_FETCH_TIMEOUT if fast_mode else _FETCH_TIMEOUT

    def _fetch_with_timeout(url: str) -> str:
        return fetch_page_content(url, max_chars=max_chars, timeout=fetch_timeout)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_with_timeout, url): url
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


def search_and_summarize(query: str, max_results: int = 2, fetch_content: bool = True, fast_mode: bool = False) -> str:
    """搜索并返回格式化的结果文本（供 LLM 使用）。

    Args:
        fast_mode: 快速模式——减少超时、重试、页面抓取数量和字符数。
    """
    # 快速模式：减少结果数，不抓取页面内容（只返回标题和链接节省 ~8-16s）
    effective_max = 1 if fast_mode else max_results
    effective_fetch = False if fast_mode else fetch_content

    results = duckduckgo_search(query, max_results=effective_max, fast_mode=fast_mode)
    if not results:
        return "[搜索未找到相关结果]"

    page_contents = {}
    if effective_fetch:
        urls = [r["href"] for r in results]
        max_chars = 500 if fast_mode else 1000
        page_contents = _fetch_all_pages(urls, max_chars=max_chars, fast_mode=fast_mode)

    output_lines = [f"## 搜索结果：'{query}'\n"]
    for i, r in enumerate(results, 1):
        output_lines.append(f"**[{i}]** {r['title']}")
        output_lines.append(f"链接: {r['href']}")
        if effective_fetch:
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
