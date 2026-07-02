"""
PDF 解析器 - 文字提取 + 表格识别
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

    def _parse_pdf(self, path: Path, extract_tables: bool) -> List[Document]:
        docs = []
        page_texts = []
        page_tables = []

        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            for page_num, page in enumerate(doc):
                text = page.get_text().strip()
                if text:
                    page_texts.append({
                        "text": text,
                        "page": page_num + 1,
                    })
            doc.close()
        except Exception as e:
            logger.warning(f"PyMuPDF 解析失败: {e}")

        if not page_texts:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(path))
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        page_texts.append({
                            "text": text.strip(),
                            "page": i + 1,
                        })
            except Exception as e:
                logger.warning(f"PyPDF2 解析失败: {e}")

        # 提取表格（使用 camelot）
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
                    logger.info(f"camelot 提取了 {len(tables)} 个无线表格")
                except Exception as e:
                    logger.warning(f"camelot 表格提取失败: {e}")

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

        logger.info(f"PDF 解析完成: {path.name} → {len(docs)} 个文档")
        return docs
