"""
图片解析器 - OCR + VLM 描述（复用 local-multimodal-rag 和 eval-dataset 思路）
"""
import logging
import base64
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger("parser.image")


class ImageParser:
    """图片解析器：OCR 提取文字（如有）+ VLM 生成描述"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def parse(self, file_path: str, generate_description: bool = True) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ocr_text = self._ocr_text(path)
        description = ""

        if generate_description:
            description = self._vlm_describe(path, ocr_text)

        # 构建文档内容
        content_parts = []
        if ocr_text:
            content_parts.append(f"[OCR 文字]\n{ocr_text}")
        if description:
            content_parts.append(f"[图片描述]\n{description}")

        content = "\n\n".join(content_parts) if content_parts else f"[图片文件: {path.name}]"

        return [Document(
            page_content=content,
            metadata={
                "source": str(path),
                "source_type": "image_description",
                "file_name": path.name,
                "file_ext": path.suffix.lower(),
                "image_path": str(path),
                "has_ocr": bool(ocr_text),
                "has_vlm": bool(description),
            },
        )]

    def _ocr_text(self, path: Path) -> str:
        """用 PaddleOCR/tesseract 提取图片文字"""
        # PaddleOCR
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False)
            result = ocr.ocr(str(path))
            texts = []
            for line_group in result:
                if line_group:
                    for line in line_group:
                        texts.append(line[1][0])
            return "\n".join(texts)
        except Exception as e:
            logger.warning(f"PaddleOCR 不可用或失败: {e}")

        # fallback: tesseract
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(str(path))
            text = pytesseract.image_to_string(img, lang="chi_sim+eng")
            return text.strip()
        except Exception as e:
            logger.warning(f"Tesseract 失败: {e}")

        return ""

    def _vlm_describe(self, path: Path, ocr_text: str) -> str:
        """用 Qwen-VL 生成图片描述"""
        api_key = self.api_key
        if not api_key:
            import os
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")

        if not api_key:
            logger.warning("未配置 API Key，跳过 VLM 描述")
            return ""

        try:
            from openai import OpenAI
            # 图片转 Base64
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            ext = path.suffix.lower().lstrip(".")
            if ext == "jpg":
                ext = "jpeg"
            data_uri = f"data:image/{ext};base64,{img_b64}"

            # DashScope 兼容 OpenAI SDK
            client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            prompt = (
                "请详细描述这张图片的内容。如果包含文字、图表、流程、数据，请完整复述。"
                "如果是截图，请说明截图中的关键信息。"
            )

            resp = client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"VLM 描述生成失败: {e}")
            # fallback 描述
            return f"[图片: {path.name}]"
