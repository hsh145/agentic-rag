"""
PDF 解析器 - 文字提取 + 表格识别

改进说明（2026-07-18）:
  1. 多模式文本提取：按 text → dict → 原始排序降级
  2. 布局感知：用 blocks 模式保持段落顺序
  3. 图片页检测：文字不足时标记而非留空
  4. 跨页合并：短页自动与相邻页合并
"""
import logging
import io
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger("parser.pdf")


class PDFParser:
    """PDF 解析器，支持文字提取和表格识别"""

    def parse(self, file_path: str, extract_tables: bool = True) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        documents = []

        if ext == ".pdf":
            documents = self._parse_pdf(path, extract_tables)
        else:
            raise ValueError(f"不支持的 PDF 格式: {ext}")

        return documents

    def _extract_text_pymupdf(self, path: Path) -> List[dict]:
        """使用 PyMuPDF 多模式提取文本

        Returns:
            [{"text": str, "page": int, "images": int}, ...]
            每页一个条目，无文字的页也会保留（标记 images > 0）
        """
        import fitz
        doc = fitz.open(str(path))
        results = []

        for page_num, page in enumerate(doc):
            page_result = {"page": page_num + 1, "images": len(page.get_images()), "text": ""}

            # 模式1: 标准文本提取
            text = page.get_text("text").strip()
            if len(text) > 30:
                page_result["text"] = text
                results.append(page_result)
                continue

            # 模式2: blocks 模式（保留布局顺序）
            if len(text) <= 30:
                blocks = page.get_text("dict")
                block_texts = []
                for block in blocks.get("blocks", []):
                    if block.get("type") == 0:  # 文本块
                        for line in block.get("lines", []):
                            line_text = "".join(
                                span.get("text", "") for span in line.get("spans", [])
                            )
                            if line_text.strip():
                                block_texts.append(line_text)
                    elif block.get("type") == 1:  # 图片块
                        page_result["images"] += 1
                merged = "\n".join(block_texts).strip()
                if len(merged) > len(text):
                    page_result["text"] = merged
                    results.append(page_result)
                    continue

            # 模式3: 原始字符串（兜底）
            if not page_result["text"]:
                raw = page.get_text("raw").strip()
                # raw 模式可能包含乱码，过滤掉不可打印字符
                if raw:
                    clean = "".join(c for c in raw if c.isprintable() or c in "\n\t").strip()
                    if len(clean) > 20:
                        page_result["text"] = clean
                        results.append(page_result)
                        continue

            # 无文字 → 保留空记录（标记为图片页）
            results.append(page_result)

        doc.close()
        return results

    def _merge_short_pages(self, pages: List[dict], min_chars: int = 50) -> List[dict]:
        """将文字少的页合并到前一项"""
        if not pages:
            return []
        merged = [pages[0]]
        for p in pages[1:]:
            # 如果是图片页或文字极少，合并到上一页
            if len(p["text"].strip()) < min_chars and merged:
                if merged[-1]["text"]:
                    merged[-1]["text"] += "\n\n" + p["text"]
                    merged[-1]["images"] += p["images"]
                else:
                    merged.append(p)
            else:
                merged.append(p)
        # 过滤纯图片页（仅在有文字页时剔除）
        text_pages = [p for p in merged if p["text"].strip()]
        return text_pages if text_pages else merged

    def _parse_pdf(self, path: Path, extract_tables: bool) -> List[Document]:
        docs = []
        page_tables = []

        # ---- 主提取：PyMuPDF 多模式 ----
        page_texts = self._extract_text_pymupdf(path)

        # ---- 降级：PyPDF2（如果 PyMuPDF 完全失败）----
        if not any(p["text"].strip() for p in page_texts):
            logger.warning("PyMuPDF 提取不足，尝试 PyPDF2 降级")
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(path))
                page_texts = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    page_texts.append({
                        "text": text.strip() if text else "",
                        "page": i + 1,
                        "images": 0,
                    })
            except Exception as e:
                logger.warning(f"PyPDF2 降级也失败: {e}")

        # ---- 短页合并 ----
        page_texts = self._merge_short_pages(page_texts)

        # ---- 表格提取 ----
        if extract_tables:
            try:
                import camelot
                tables = camelot.read_pdf(str(path), pages="all", flavor="lattice")
                for i, table in enumerate(tables):
                    md_table = table.df.to_markdown(index=False)
                    page_tables.append({
                        "text": md_table,
                        "page": table.page,
                        "type": "table",
                    })
                if tables:
                    logger.info(f"camelot 提取了 {len(tables)} 个有线表格")
            except Exception:
                try:
                    import camelot
                    tables = camelot.read_pdf(str(path), pages="all", flavor="stream")
                    for i, table in enumerate(tables):
                        md_table = table.df.to_markdown(index=False)
                        page_tables.append({
                            "text": md_table,
                            "page": table.page,
                            "type": "table",
                        })
                    if tables:
                        logger.info(f"camelot 提取了 {len(tables)} 个无线表格")
                except Exception as e:
                    logger.debug(f"camelot 表格提取跳过: {e}")

        # ---- 生成 Document ----
        for item in page_texts:
            docs.append(Document(
                page_content=item["text"],
                metadata={
                    "source": str(path),
                    "source_type": "pdf",
                    "page": item["page"],
                    "file_name": path.name,
                },
            ))

        for item in page_tables:
            docs.append(Document(
                page_content=item["text"],
                metadata={
                    "source": str(path),
                    "source_type": "table",
                    "page": item["page"],
                    "file_name": path.name,
                    "format": "markdown_table",
                },
            ))

        logger.info(f"PDF 解析完成: {path.name} → {len(docs)} 个文档 ({len(page_texts)} 文字页, {len(page_tables)} 表格)")
        return docs
