"""
文件类型自动检测
"""
import logging
from pathlib import Path

logger = logging.getLogger("parser.detector")


class FileTypeDetector:
    EXTENSION_MAP = {
        ".txt": "text",
        ".md": "markdown",
        ".json": "text",
        ".csv": "text",
        ".log": "text",
        ".yaml": "text",
        ".yml": "text",
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "docx",
        ".xlsx": "xlsx",
        ".xls": "xls",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".gif": "image",
        ".bmp": "image",
        ".webp": "image",
    }

    @classmethod
    def detect(cls, file_path: str) -> str:
        """返回文件类型标识：text|markdown|pdf|docx|xlsx|image|unknown"""
        ext = Path(file_path).suffix.lower()
        return cls.EXTENSION_MAP.get(ext, "unknown")

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        return cls.detect(file_path) != "unknown"
