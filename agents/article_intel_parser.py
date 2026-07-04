from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Any

import requests


class _FallbackTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip_stack: list[str] = []
        self.in_title = False
        self.in_h1 = False
        self.in_article = 0
        self.in_main = 0
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []
        self.article_parts: list[str] = []
        self.main_parts: list[str] = []
        self.all_parts: list[str] = []
        self.meta_description = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        if tag in {"script", "style", "noscript", "nav", "footer", "aside", "svg"}:
            self.skip_stack.append(tag)
        if tag == "title":
            self.in_title = True
        if tag == "h1":
            self.in_h1 = True
        if tag == "article":
            self.in_article += 1
        if tag == "main":
            self.in_main += 1
        if tag == "meta":
            name = attrs_dict.get("name", "").casefold()
            prop = attrs_dict.get("property", "").casefold()
            if name == "description" or prop == "og:description":
                self.meta_description = attrs_dict.get("content", "").strip()

    def handle_endtag(self, tag: str) -> None:
        if self.skip_stack and self.skip_stack[-1] == tag:
            self.skip_stack.pop()
        if tag == "title":
            self.in_title = False
        if tag == "h1":
            self.in_h1 = False
        if tag == "article" and self.in_article:
            self.in_article -= 1
        if tag == "main" and self.in_main:
            self.in_main -= 1

    def handle_data(self, data: str) -> None:
        if self.skip_stack:
            return
        text = ArticleIntelParser.clean_text(data)
        if not text:
            return
        self.all_parts.append(text)
        if self.in_title:
            self.title_parts.append(text)
        if self.in_h1:
            self.h1_parts.append(text)
        if self.in_article:
            self.article_parts.append(text)
        if self.in_main:
            self.main_parts.append(text)


class ArticleIntelParser:
    """Fetches article pages and extracts match intelligence from article bodies."""

    INJURY_KEYWORDS = ["injury", "injured", "out", "misses", "doubtful", "returned from"]
    SUSPENSION_KEYWORDS = ["suspended", "suspension", "ban", "yellow card", "red card"]
    LINEUP_KEYWORDS = ["predicted lineup", "predicted lineups", "starting xi", "4-3-3", "4-2-3-1"]
    COACH_KEYWORDS = ["manager said", "coach said", "said:", "told reporters", "press conference"]

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    @staticmethod
    def clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", unescape(text or "")).strip()

    def fetch(self, url: str) -> dict[str, Any]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
        except Exception as exc:
            return self.empty_fetch(url, f"request_failed: {exc!r}")

        parsed = self.extract_body(response.text or "")
        parsed.update(
            {
                "url": url,
                "status_code": response.status_code,
                "html_length": len(response.text or ""),
                "fetch_success": response.status_code == 200 and parsed["parser_success"],
            }
        )
        if response.status_code != 200:
            parsed["failure_reason"] = f"http_{response.status_code}"
            parsed["fetch_success"] = False
        return parsed

    def empty_fetch(self, url: str, failure_reason: str) -> dict[str, Any]:
        return {
            "url": url,
            "status_code": "",
            "html_length": 0,
            "title": "",
            "meta_description": "",
            "article_body": "",
            "body_char_count": 0,
            "fetch_success": False,
            "parser_success": False,
            "failure_reason": failure_reason,
        }

    def extract_body(self, html: str) -> dict[str, Any]:
        if not html:
            return self.empty_fetch("", "empty_response")

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "nav", "footer", "aside", "svg"]):
                tag.decompose()
            title_node = soup.find("h1") or soup.find("title")
            meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            article = soup.find("article") or soup.find("main")
            body = article.get_text(" ", strip=True) if article else soup.get_text(" ", strip=True)
            title = title_node.get_text(" ", strip=True) if title_node else ""
            meta_description = meta.get("content", "").strip() if meta else ""
        except Exception:
            parser = _FallbackTextParser()
            parser.feed(html)
            title = self.clean_text(" ".join(parser.h1_parts or parser.title_parts))
            meta_description = self.clean_text(parser.meta_description)
            body = self.clean_text(" ".join(parser.article_parts or parser.main_parts or parser.all_parts))

        body = self.clean_text(body)
        parser_success = len(body) >= 500
        return {
            "title": self.clean_text(title),
            "meta_description": self.clean_text(meta_description),
            "article_body": body,
            "body_char_count": len(body),
            "parser_success": parser_success,
            "failure_reason": "" if parser_success else "parser_failed",
        }

    def extract_intel(self, article: dict[str, Any]) -> dict[str, Any]:
        body = str(article.get("article_body", ""))
        title = str(article.get("title", ""))
        meta = str(article.get("meta_description", ""))
        combined = self.clean_text(" ".join([title, meta, body]))
        lower = combined.casefold()

        injuries = self.sentences_for_keywords(combined, self.INJURY_KEYWORDS)
        suspensions = self.sentences_for_keywords(combined, self.SUSPENSION_KEYWORDS)
        lineups = self.sentences_for_keywords(combined, self.LINEUP_KEYWORDS)
        coach_comments = self.coach_comments(combined)

        has_evidence = bool(injuries or suspensions or lineups)
        confidence = 0.75 if article.get("fetch_success") else 0.4
        if lineups:
            confidence = 0.9

        return {
            "team_news": self.clean_text(" | ".join(filter(None, [title, meta]))) or "unknown",
            "injuries": injuries or "unknown",
            "suspensions": suspensions or "unknown",
            "expected_lineup": lineups or "unknown",
            "coach_comments": coach_comments or "unknown",
            "source_url": article.get("url") or "unknown",
            "confidence": confidence,
            "intel_has_content": has_evidence,
        }

    def snippets_to_intel(self, articles: list[dict[str, Any]]) -> dict[str, Any]:
        selected = articles[:3]
        titles = [article.get("title", "") for article in selected if article.get("title")]
        urls = [article.get("url", "") for article in selected if article.get("url")]
        text = self.clean_text(" | ".join(titles))
        lower = text.casefold()
        return {
            "team_news": text or "unknown",
            "injuries": text if any(self.keyword_found(lower, term) for term in self.INJURY_KEYWORDS) else "unknown",
            "suspensions": text if any(self.keyword_found(lower, term) for term in self.SUSPENSION_KEYWORDS) else "unknown",
            "expected_lineup": text if any(self.keyword_found(lower, term) for term in self.LINEUP_KEYWORDS + ["lineup", "squad"]) else "unknown",
            "coach_comments": text if any(self.keyword_found(lower, term) for term in ["coach", "manager", "press conference", "said"]) else "unknown",
            "source_url": "; ".join(urls) if urls else "unknown",
            "confidence": 0.4 if urls else 0.2,
            "intel_has_content": any(self.keyword_found(lower, term) for term in self.INJURY_KEYWORDS + self.SUSPENSION_KEYWORDS + self.LINEUP_KEYWORDS),
        }

    def best_intel_from_articles(self, articles: list[dict[str, Any]], limit: int = 2) -> dict[str, Any]:
        candidates = [self.snippets_to_intel(articles)]
        for article in articles[:limit]:
            url = str(article.get("url", "")).strip()
            if not url.startswith("http"):
                continue
            parsed = self.fetch(url)
            intel = self.extract_intel(parsed)
            if parsed.get("fetch_success") and intel["intel_has_content"]:
                candidates.append(intel)
        return sorted(candidates, key=lambda item: float(item.get("confidence", 0)), reverse=True)[0]

    def sentences_for_keywords(self, text: str, keywords: list[str]) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        matches = []
        for sentence in sentences:
            lower = sentence.casefold()
            if any(self.keyword_found(lower, keyword) for keyword in keywords):
                matches.append(sentence.strip())
        return self.clean_text(" ".join(matches[:3]))

    def keyword_found(self, lower_text: str, keyword: str) -> bool:
        escaped = re.escape(keyword.casefold()).replace(r"\ ", r"\s+")
        return re.search(rf"(?<![a-z]){escaped}(?![a-z])", lower_text) is not None

    def coach_comments(self, text: str) -> str:
        quoted = re.findall(r"[“\"]([^”\"]{30,240})[”\"]", text)
        if quoted:
            return self.clean_text(" | ".join(quoted[:2]))
        return self.sentences_for_keywords(text, self.COACH_KEYWORDS)
