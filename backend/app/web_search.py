from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from .config import Settings


def search_web(query: str | list[str], settings: Settings) -> dict[str, Any]:
    """Return compact web snippets for follow-up QA.

    The default provider uses DuckDuckGo's public instant-answer endpoint so the
    prototype works without another API key. A configured JSON search endpoint
    can be introduced later through WEB_SEARCH_PROVIDER/WEB_SEARCH_BASE_URL.
    """

    queries = _clean_queries(query)
    if not settings.web_search_enabled or not queries:
        return {"enabled": False, "provider": settings.web_search_provider, "query": "", "queries": queries, "results": []}

    provider = settings.web_search_provider or "duckduckgo"
    all_results: list[dict[str, str]] = []
    errors: list[str] = []
    for clean_query in queries:
        try:
            if provider in {"duckduckgo", "ddg"}:
                results = _search_duckduckgo(clean_query, settings)
            elif settings.web_search_base_url:
                results = _search_configured_endpoint(clean_query, settings)
            else:
                raise RuntimeError(f"Web search provider '{provider}' needs WEB_SEARCH_BASE_URL.")
            for item in results:
                all_results.append({**item, "query": clean_query})
        except Exception as exc:
            errors.append(f"{clean_query}: {exc}")
        all_results = _dedupe_results(all_results)
        if len(all_results) >= settings.web_search_max_results:
            break

    payload: dict[str, Any] = {
        "enabled": True,
        "provider": provider,
        "query": queries[0],
        "queries": queries,
        "results": all_results[: settings.web_search_max_results],
    }
    if errors and not payload["results"]:
        payload["error"] = "; ".join(errors)
    elif errors:
        payload["errors"] = errors[:3]
    return payload


def _clean_queries(query: str | list[str]) -> list[str]:
    raw_items = query if isinstance(query, list) else [query]
    queries: list[str] = []
    for item in raw_items:
        clean = " ".join(str(item or "").split())
        if clean and clean not in queries:
            queries.append(clean)
    return queries[:5]


def _search_duckduckgo(query: str, settings: Settings) -> list[dict[str, str]]:
    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "no_redirect": "1",
        "kl": "cn-zh",
    }
    headers = {"User-Agent": "audio-first-video-agent/0.1"}
    timeout = httpx.Timeout(settings.web_search_timeout_seconds)
    results: list[dict[str, str]] = []
    errors: list[str] = []
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            response = client.get("https://api.duckduckgo.com/", params=params)
            response.raise_for_status()
            payload = response.json()
        _append_duckduckgo_result(
            results,
            title=payload.get("Heading") or payload.get("AbstractSource"),
            snippet=payload.get("AbstractText") or payload.get("Abstract"),
            url=payload.get("AbstractURL"),
        )
        for topic in payload.get("RelatedTopics") or []:
            if "Topics" in topic:
                for nested in topic.get("Topics") or []:
                    _append_duckduckgo_result(
                        results,
                        title=nested.get("Text"),
                        snippet=nested.get("Text"),
                        url=nested.get("FirstURL"),
                    )
            else:
                _append_duckduckgo_result(
                    results,
                    title=topic.get("Text"),
                    snippet=topic.get("Text"),
                    url=topic.get("FirstURL"),
                )
    except Exception as exc:
        errors.append(f"DuckDuckGo instant answer failed: {exc}")

    if len(results) < settings.web_search_max_results:
        try:
            results.extend(_search_duckduckgo_html(query, settings, settings.web_search_max_results - len(results)))
        except Exception as exc:
            errors.append(f"DuckDuckGo HTML search failed: {exc}")
    if len(results) < settings.web_search_max_results:
        try:
            results.extend(_search_bing_html(query, settings, settings.web_search_max_results - len(results)))
        except Exception as exc:
            errors.append(f"Bing HTML search failed: {exc}")
    results = _dedupe_results(results)
    if not results and errors:
        raise RuntimeError("; ".join(errors))
    return results


def _search_duckduckgo_html(query: str, settings: Settings, limit: int) -> list[dict[str, str]]:
    headers = {"User-Agent": "audio-first-video-agent/0.1"}
    timeout = httpx.Timeout(settings.web_search_timeout_seconds)
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        response = client.get("https://html.duckduckgo.com/html/", params={"q": query})
        response.raise_for_status()
        body = response.text
    results: list[dict[str, str]] = []
    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>(?P<tail>.*?)(?=<a[^>]+class="result__a"|</body>)',
        re.S | re.I,
    )
    snippet_pattern = re.compile(r'<(?:a|div)[^>]+class="result__snippet"[^>]*>(.*?)</(?:a|div)>', re.S | re.I)
    for match in pattern.finditer(body):
        tail = match.group("tail")
        snippet_match = snippet_pattern.search(tail)
        title = _strip_html(match.group("title"))
        snippet = _strip_html(snippet_match.group(1)) if snippet_match else ""
        url = _normalize_duckduckgo_url(html.unescape(match.group("url")))
        if title or snippet or url:
            results.append({"title": _clean_text(title or url, 120), "snippet": _clean_text(snippet, 320), "url": url})
        if len(results) >= limit:
            break
    return results


def _search_bing_html(query: str, settings: Settings, limit: int) -> list[dict[str, str]]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"}
    timeout = httpx.Timeout(settings.web_search_timeout_seconds)
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        response = client.get("https://www.bing.com/search", params={"q": query})
        response.raise_for_status()
        body = response.text
    results: list[dict[str, str]] = []
    pattern = re.compile(r'<li class="b_algo".*?<h2>\s*<a[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?</h2>(?P<tail>.*?)(?=<li class="b_algo"|</ol>)', re.S | re.I)
    snippet_pattern = re.compile(r"<p[^>]*>(.*?)</p>", re.S | re.I)
    for match in pattern.finditer(body):
        snippet_match = snippet_pattern.search(match.group("tail"))
        title = _strip_html(match.group("title"))
        snippet = _strip_html(snippet_match.group(1)) if snippet_match else ""
        url = html.unescape(match.group("url")).strip()
        if title or snippet or url:
            results.append({"title": _clean_text(title or url, 120), "snippet": _clean_text(snippet, 320), "url": url})
        if len(results) >= limit:
            break
    return results


def _search_configured_endpoint(query: str, settings: Settings) -> list[dict[str, str]]:
    assert settings.web_search_base_url is not None
    headers = {"User-Agent": "audio-first-video-agent/0.1"}
    if settings.web_search_api_key:
        headers["Authorization"] = f"Bearer {settings.web_search_api_key}"
    timeout = httpx.Timeout(settings.web_search_timeout_seconds)
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        response = client.get(settings.web_search_base_url, params={"q": query, "query": query})
        response.raise_for_status()
        payload = response.json()
    raw_items = payload.get("results") or payload.get("items") or payload.get("webPages", {}).get("value") or []
    results = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("name") or item.get("heading")
        snippet = item.get("snippet") or item.get("description") or item.get("content")
        url = item.get("url") or item.get("link")
        if title or snippet or url:
            results.append(
                {
                    "title": _clean_text(title or url or "Untitled result", 120),
                    "snippet": _clean_text(snippet or "", 320),
                    "url": str(url or ""),
                }
            )
    return _dedupe_results(results)


def _append_duckduckgo_result(results: list[dict[str, str]], *, title: object, snippet: object, url: object) -> None:
    title_text = _clean_text(title, 120)
    snippet_text = _clean_text(snippet, 320)
    url_text = str(url or "").strip()
    if not snippet_text and not url_text:
        return
    results.append({"title": title_text or url_text or "DuckDuckGo result", "snippet": snippet_text, "url": url_text})


def _dedupe_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen = set()
    for result in results:
        signature = (result.get("url") or "", result.get("title") or "")
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(result)
    return deduped


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    return html.unescape(" ".join(text.split()))


def _normalize_duckduckgo_url(value: str) -> str:
    url = value.strip()
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return url


def _clean_text(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."
