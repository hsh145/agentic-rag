"""
文档解析器包
"""
from .detector import FileTypeDetector
from .pdf_parser import PDFParser
from .office_parser import OfficeParser
from .image_parser import ImageParser
from .text_parser import TextParser

__all__ = ["FileTypeDetector", "PDFParser", "OfficeParser", "ImageParser", "TextParser"]
