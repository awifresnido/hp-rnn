"""Sanity-check the generated dependency-graph page."""

import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

HTML = Path("docs/dependency-graph.html")
MD = Path("docs/architecture.md")

VOID = {"br", "hr", "img", "meta", "link", "input", "path", "rect", "line", "circle", "use"}


class Checker(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.tags: Counter[str] = Counter()

    def handle_starttag(self, tag: str, attrs) -> None:
        self.tags[tag] += 1
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f"stray </{tag}>")
        elif self.stack[-1] != tag:
            self.errors.append(f"expected </{self.stack[-1]}> but found </{tag}>")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()


text = HTML.read_text(encoding="utf-8")
checker = Checker()
checker.feed(text)

print(f"html bytes    : {len(text):,}")
print(f"unclosed tags : {checker.stack or 'none'}")
print(f"nesting errors: {checker.errors or 'none'}")
print(f"tag counts    : {dict(checker.tags)}")
external = re.findall(r"https?://[^\s\"'<>]+", text)
print(f"external refs : {external or 'none (renders offline)'}")

markdown = MD.read_text(encoding="utf-8")
print(f"markdown lines: {len(markdown.splitlines()):,}")
print(f"module sections: {markdown.count('### `')}")
print(f"cross-module edges: {markdown.count('→ `src.')}")
