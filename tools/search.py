"""联网搜索工具 - 为 Researcher Agent 提供搜索能力。"""

import json
import logging
import time
import urllib.parse
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def _fetch(url: str, timeout: int = 15) -> str:
    """发送 HTTP GET 请求并返回文本内容"""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


# 搜索结果缓存（query -> (timestamp, result)）
_search_cache: dict[str, tuple[float, list[dict]]] = {}
_SEARCH_CACHE_TTL = 300  # 5 分钟


def _get_cached_search(query: str) -> list[dict] | None:
    """获取缓存的搜索结果。"""
    import time
    if query in _search_cache:
        ts, results = _search_cache[query]
        if time.time() - ts < _SEARCH_CACHE_TTL:
            logger.debug(f"Search cache hit: '{query}'")
            return results
    return None


def _set_cached_search(query: str, results: list[dict]):
    """缓存搜索结果。"""
    import time
    _search_cache[query] = (time.time(), results)


def duckduckgo_search(query: str, max_results: int = 3) -> list[dict]:
    """使用 DuckDuckGo 搜索（无需 API Key）。

    返回结果列表，每项包含 title, href, snippet。
    - 默认返回 3 条（减少网页抓取耗时）
    - 启用 5 分钟缓存，避免重复搜索
    - 搜索超时 8 秒，避免长时间阻塞
    """
    import re

    # 检查缓存
    cached = _get_cached_search(query)
    if cached is not None:
        return cached

    try:
        # DuckDuckGo HTML 搜索
        encoded = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        html = _fetch(url, timeout=8)

        results = []
        # 尝试多种选择器匹配结果块
        patterns = [
            # 新版 DuckDuckGo HTML
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            # 备用模式
            r'<a[^>]+rel="nofollow"[^>]+class="[^"]*result[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            # 再备用
            r'<h[23][^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?</h[23]>',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, html, re.IGNORECASE | re.DOTALL):
                href = match.group(1)
                # DuckDuckGo 使用重定向链接，提取实际 URL
                redirect_match = re.search(r'uddg=([^&]+)', href)
                if redirect_match:
                    href = urllib.parse.unquote(redirect_match.group(1))

                title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
                if title and href and href.startswith("http"):
                    results.append({"title": title, "href": href})
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

        logger.info(f"DuckDuckGo search: '{query}' -> {len(results)} results")
        _set_cached_search(query, results)
        return results
    except Exception as e:
        logger.warning(f"Search failed for query: {query} - {e}")
        return []


def fetch_page_content(url: str, max_chars: int = 1200) -> str:
    """获取网页内容（简单文本提取）。

    超时缩短到 5 秒，避免单个网页阻塞整体搜索。
    """
    try:
        html = _fetch(url, timeout=5)
        # 移除 script/style 标签
        import re
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # 移除所有 HTML 标签
        text = re.sub(r'<[^>]+>', ' ', text)
        # 压缩空白
        text = re.sub(r'\s+', ' ', text).strip()
        # 截断
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[网页内容已截断]"
        return text
    except Exception as e:
        logger.warning(f"Failed to fetch page: {url} - {e}")
        return "[无法获取网页内容]"


def search_and_summarize(query: str, max_results: int = 2, fetch_content: bool = True) -> str:
    """搜索并返回格式化的结果文本（供 LLM 使用）。

    返回格式化的搜索结果文本，包含标题、链接和摘要。
    """
    results = duckduckgo_search(query, max_results=max_results)
    if not results:
        return "[搜索未找到相关结果]"

    output_lines = [f"## 搜索结果：'{query}'\n"]
    for i, r in enumerate(results, 1):
        output_lines.append(f"**[{i}]** {r['title']}")
        output_lines.append(f"链接: {r['href']}")
        if fetch_content:
            content = fetch_page_content(r['href'], max_chars=1500)
            output_lines.append(f"摘要: {content}\n")
        else:
            output_lines.append("")

    return "\n".join(output_lines)
