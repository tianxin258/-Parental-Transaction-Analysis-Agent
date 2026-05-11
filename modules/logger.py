"""日志工具：同时输出到控制台和日志文件

Windows 终端 GBK 编码无法输出 emoji，所以控制台 Handler 会自动
将不可编码字符替换为 ?，而日志文件（UTF-8）完整保留所有字符。
"""

import logging
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


class _ConsoleHandler(logging.StreamHandler):
    """控制台 Handler：自动替换 GBK 无法编码的字符，避免 UnicodeEncodeError。"""

    def __init__(self):
        super().__init__(sys.stdout)

    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            msg = self.format(record)
            try:
                sys.stdout.write(msg.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding) + self.terminator)
            except Exception:
                self.handleError(record)


def setup_logger(name: str = "AI家长成交分析Agent") -> logging.Logger:
    """初始化日志系统：日志文件 + 控制台输出。"""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 防止重复添加 handler
    if logger.handlers:
        return logger

    # ── 文件 Handler（追加模式，UTF-8）──
    log_file = _LOG_DIR / "runtime.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # ── 控制台 Handler（兼容 GBK 编码）──
    console_handler = _ConsoleHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    logger.info(f"[日志文件] {log_file}")
    return logger
