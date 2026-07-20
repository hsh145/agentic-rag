"""
Agent 工具函数 — 重试、降级等工程组件
"""
import time
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger("agent.utils")


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """带指数退避的重试装饰器

    用法:
        @retry(max_attempts=3, base_delay=1.0)
        def unstable_api_call():
            ...

    参数:
        max_attempts: 最大重试次数
        base_delay: 首次重试延迟（秒）
        max_delay: 最大延迟（秒）
        backoff: 退避倍数（每次延迟 * backoff）
        exceptions: 哪些异常需要重试
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts:
                        delay = min(base_delay * (backoff ** (attempt - 1)), max_delay)
                        logger.warning(
                            f"{func.__name__} 第{attempt}次失败: {e}, "
                            f"{delay:.1f}s 后重试 ({max_attempts - attempt}次剩余)"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} 重试{max_attempts}次均失败: {e}")
            raise last_error
        return wrapper
    return decorator


def safe_call(func: Callable, default_return: Any = None, *args, **kwargs) -> Any:
    """安全调用：函数失败时返回默认值而非抛异常"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"{func.__name__} 调用失败，走降级: {e}")
        return default_return
