"""
纯文本/代码解析器
"""
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger("parser.text")


class TextParser:
    CODE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".go", ".rs", ".sh", ".bat", ".sql", ".html", ".css", ".vue", ".jsx", ".tsx"}

    def parse(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = path.read_text(encoding="gbk")
            except Exception as e:
                logger.error(f"读取文件失败: {e}")
                return []

        ext = path.suffix.lower()
        source_type = "code" if ext in self.CODE_EXTENSIONS else ("markdown" if ext == ".md" else "text")

        lang_map = {".py": "python", ".js": "javascript", ".ts": "typescript",
                    ".java": "java", ".go": "go", ".rs": "rust", ".sql": "sql",
                    ".html": "html", ".css": "css", ".sh": "bash", ".json": "json"}
        language = lang_map.get(ext, "")

        return [Document(
            page_content=content,
            metadata={
                "source": str(path),
                "source_type": source_type,
                "file_name": path.name,
                "file_ext": ext,
                "language": language,
                "size": len(content),
            },
        )]
