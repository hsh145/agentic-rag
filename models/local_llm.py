"""
本地模型加载器 — 支持 transformers 格式的微调模型

用法：
    from models.local_llm import LocalLLM
    llm = LocalLLM("./models/qwen-judge", device="cuda")
    response = llm.generate("用户问题是什么意图？")
"""

import logging
import json
from typing import Optional

logger = logging.getLogger("models.local_llm")


class LocalLLM:
    """本地小模型推理封装（用于替代 API 调用）

    支持的模型：
      - Qwen2.5-1.5B / 3B / 7B（推荐蒸馏后的判断模型）
      - Llama-3.2-3B
      - 任何 HuggingFace transformers 支持的 CausalLM
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipe = None

    def _lazy_load(self):
        """延迟加载模型（首次调用时加载）"""
        if self._pipe is not None:
            return

        logger.info(f"加载本地模型: {self.model_path} (device={self.device})")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
            )
            if self.device == "cpu":
                self._model = self._model.to(self.device)

            logger.info(f"模型加载完成: {self.model_path}")
        except ImportError as e:
            logger.error(f"请安装 transformers: pip install transformers torch")
            raise ImportError(f"缺少依赖: {e}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """生成回答

        Args:
            prompt: 用户输入
            system_prompt: 系统提示词（Qwen 格式）

        Returns:
            生成的文本
        """
        self._lazy_load()

        # 构建 Qwen 格式的对话
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        import torch
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # 尝试提取 JSON 内容（如果输出包含 JSON）
        return response

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        """生成并解析 JSON 输出（用于判断节点）"""
        response = self.generate(prompt, system_prompt)
        # 清理可能的 markdown 格式
        content = response.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"JSON 解析失败，原始输出: {content[:200]}")
            return {}

    @classmethod
    def from_config(cls, config) -> Optional["LocalLLM"]:
        """从配置对象创建实例"""
        if not config.use_local_judge or not config.local_judge_model_path:
            return None
        return cls(
            model_path=config.local_judge_model_path,
            device=config.local_judge_device,
        )
