"""Shared fixtures: a tiny synthetic EPUB so tests never touch real book text."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

CONTAINER = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="id">urn:uuid:test</dc:identifier>
    <dc:title>Fixture Book</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="ch1"/>
    <itemref idref="ch2"/>
  </spine>
</package>
"""

CH1 = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>One</title></head>
<body>
<h1>Chapter One</h1>
<p>&#8220;Harry looked at Ron.&#8221; He didn&#8217;t speak.</p>
<p>A long dash &#8212; then silence.</p>
<p>9</p>
</body></html>
"""

CH2 = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Two</title></head>
<body>
<p>Hermione exam-
ined the map.</p>
<p>***</p>
</body></html>
"""

NAV = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<body><nav epub:type="toc"><ol><li><a href="ch1.xhtml">One</a></li></ol></nav></body></html>
"""


def write_epub(path: Path) -> Path:
    """Write a minimal two-chapter EPUB in spine order."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", CONTAINER)
        z.writestr("OEBPS/content.opf", OPF)
        z.writestr("OEBPS/nav.xhtml", NAV)
        z.writestr("OEBPS/ch1.xhtml", CH1)
        z.writestr("OEBPS/ch2.xhtml", CH2)
    return path


@pytest.fixture
def fixture_epub(tmp_path: Path) -> Path:
    return write_epub(tmp_path / "fixture.epub")


@pytest.fixture
def fixture_tokens() -> list[str]:
    return ["Harry", "looked", "at", "Ron", ".", "Harry", "smiled", "."]
