import hashlib
import re
from pathlib import Path
from typing import List

from config.settings import get_settings
from core.deps import fitz
from core.text import count_tokens
from core.types import DocumentChunk


def compute_file_fingerprint(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_document(file_path: Path) -> List[DocumentChunk]:
    if file_path.suffix.lower() == ".pdf":
        return _parse_pdf(file_path)
    if file_path.suffix.lower() == ".md":
        return _parse_markdown(file_path)
    return []


def _parse_pdf(file_path: Path) -> List[DocumentChunk]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed")
    settings = get_settings()
    doc = fitz.open(file_path)
    chunks: List[DocumentChunk] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        tables = page.find_tables()
        table_texts = []
        for tab in tables:
            table_texts.append(tab.to_pandas().to_csv(index=False))
        blocks = page.get_text("dict")["blocks"]
        text_buffer = ""
        current_section = ""
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                font_size = spans[0].get("size", 0)
                font_name = spans[0].get("font", "")
                is_bold = "Bold" in font_name
                text = " ".join([s.get("text", "") for s in spans]).strip()
                if not text:
                    continue
                if font_size > 14 or is_bold:
                    current_section = text
                    chunks.append(
                        DocumentChunk(
                            chunk_id=hashlib.md5(text.encode()).hexdigest(),
                            text=text,
                            source_document=file_path.name,
                            page_number=page_num + 1,
                            section_title=current_section,
                            token_count=count_tokens(text),
                        )
                    )
                else:
                    text_buffer += " " + text
                    if count_tokens(text_buffer) >= settings.target_chunk_tokens:
                        sentences = re.split(r"(?<=[.!?]) +", text_buffer)
                        for sent in sentences:
                            chunk_id = hashlib.md5(sent.encode()).hexdigest()
                            chunks.append(
                                DocumentChunk(
                                    chunk_id=chunk_id,
                                    text=sent,
                                    source_document=file_path.name,
                                    page_number=page_num + 1,
                                    section_title=current_section,
                                    token_count=count_tokens(sent),
                                )
                            )
                        text_buffer = ""
        if text_buffer.strip():
            chunk_id = hashlib.md5(text_buffer.encode()).hexdigest()
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=text_buffer,
                    source_document=file_path.name,
                    page_number=page_num + 1,
                    section_title=current_section,
                    token_count=count_tokens(text_buffer),
                )
            )
    doc.close()
    return chunks


def _parse_markdown(file_path: Path) -> List[DocumentChunk]:
    text = file_path.read_text(encoding="utf-8")
    chunk_id = hashlib.md5(text.encode()).hexdigest()
    return [
        DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            source_document=file_path.name,
            page_number=1,
            token_count=count_tokens(text),
        )
    ]
