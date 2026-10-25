from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


@dataclass
class Section:
    text: str
    page: int | None = None      # 1-based page number (PDF), else None
    locator: str | None = None   # human label for citations: "p.3", "slide 4", "sheet Budget"


LEGACY_FORMATS = {".doc": ".docx", ".ppt": ".pptx", ".xls": ".xlsx"}


def load_file(path: Path) -> list[Section]:
    """Read a file into one or more text sections (page / slide / sheet aware)."""
    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader:
        return loader(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Section(text=text)]


# ---------------------------------------------------------------- PDF
def _load_pdf(path: Path) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            sections.append(Section(text=text, page=i, locator=f"p.{i}"))
    return sections


# ---------------------------------------------------------------- Word
def _load_docx(path: Path) -> list[Section]:
    """Paragraphs and tables, in document order."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    lines: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(child, doc).text.strip()
            if text:
                lines.append(text)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                cells = _dedupe_adjacent([c.text.strip() for c in row.cells])
                if any(cells):
                    lines.append(" | ".join(cells))
    text = "\n".join(lines)
    return [Section(text=text)] if text.strip() else []


def _dedupe_adjacent(cells: list[str]) -> list[str]:
    """Merged cells repeat in python-docx; drop consecutive duplicates."""
    out: list[str] = []
    for c in cells:
        if not out or c != out[-1]:
            out.append(c)
    return out


# ---------------------------------------------------------- PowerPoint
def _shape_text(shape) -> list[str]:
    lines: list[str] = []
    if getattr(shape, "shapes", None) is not None and shape.shape_type == 6:  # group
        for sub in shape.shapes:
            lines.extend(_shape_text(sub))
        return lines
    if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            t = "".join(run.text for run in para.runs).strip()
            if t:
                lines.append(t)
    if getattr(shape, "has_table", False) and shape.has_table:
        for row in shape.table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                lines.append(" | ".join(cells))
    return lines


def _load_pptx(path: Path) -> list[Section]:
    """One section per slide (including speaker notes) so citations say 'slide N'."""
    from pptx import Presentation

    prs = Presentation(str(path))
    sections = []
    for i, slide in enumerate(prs.slides, start=1):
        lines: list[str] = []
        for shape in slide.shapes:
            lines.extend(_shape_text(shape))
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                lines.append(f"Speaker notes: {notes}")
        text = "\n".join(lines)
        if text.strip():
            sections.append(Section(text=text, page=i, locator=f"slide {i}"))
    return sections


# --------------------------------------------------------------- Excel
def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _load_xlsx(path: Path) -> list[Section]:
    """One section per sheet. If row 1 looks like headers, rows become 'col: value' pairs,
    which embeds far better than bare numbers."""
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    sections = []
    try:
        for ws in wb.worksheets:
            rows = [[_fmt(c) for c in row] for row in ws.iter_rows(values_only=True)]
            rows = [r for r in rows if any(r)]
            if not rows:
                continue
            header = rows[0]
            has_header = len(rows) > 1 and all(header) and all(
                not h.replace(".", "", 1).replace("-", "", 1).isdigit() for h in header
            )
            lines = []
            if has_header:
                lines.append(" | ".join(header))
                for r in rows[1:]:
                    pairs = [f"{h}: {v}" for h, v in zip(header, r) if v]
                    if pairs:
                        lines.append("; ".join(pairs))
            else:
                lines = [" | ".join(c for c in r if c) for r in rows]
            sections.append(
                Section(text=f"Sheet: {ws.title}\n" + "\n".join(lines), locator=f"sheet {ws.title}")
            )
    finally:
        wb.close()
    return sections


# ---------------------------------------------------------------- HTML
class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "template", "head", "svg"}
    _BLOCK = {
        "p", "div", "br", "li", "tr", "section", "article", "header", "footer",
        "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol", "blockquote", "pre",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._skip_depth == 0:
            self.parts.append(data)


def _load_html(path: Path) -> list[Section]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    parser = _TextExtractor()
    parser.feed(raw)
    body = "".join(parser.parts)
    title = parser.title.strip()
    text = f"{title}\n\n{body}" if title else body
    return [Section(text=text)] if text.strip() else []


_LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".pptx": _load_pptx,
    ".xlsx": _load_xlsx,
    ".xlsm": _load_xlsx,
    ".html": _load_html,
    ".htm": _load_html,
}
