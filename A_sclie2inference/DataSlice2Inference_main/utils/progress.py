#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简单的终端进度条工具，避免 tqdm 滚动输出"""

from __future__ import annotations

import shutil
import sys
import time
from typing import Optional


class ProgressBar:
    """单行进度条，支持覆盖刷新"""

    def __init__(
        self,
        total: int,
        desc: str = "",
        unit: str = "",
        width: Optional[int] = None,
    ):
        self.total = max(total, 1)
        self.desc = desc.strip()
        self.unit = unit.strip()
        self.start_time = time.time()
        self.last_rendered = ""
        self.finished = False

        if width is None:
            term_width = shutil.get_terminal_size((80, 20)).columns
            width = max(20, term_width - 40)
        self.bar_width = width
        
        # 获取终端宽度，用于清除整行
        try:
            self.term_width = shutil.get_terminal_size((80, 20)).columns
        except Exception:
            self.term_width = 80

    def _format_rate(self, current: int, elapsed: float) -> str:
        if elapsed <= 0:
            return "0.00"
        return f"{current / elapsed:5.2f}"

    def update(self, current: int):
        current = min(max(current, 0), self.total)
        elapsed = time.time() - self.start_time
        progress = current / self.total

        filled = int(self.bar_width * progress)
        bar = "█" * filled + " " * (self.bar_width - filled)
        percent = f"{progress * 100:6.2f}%"
        rate = self._format_rate(current, elapsed)
        unit = f"{self.unit}" if self.unit else ""

        desc_prefix = f"{self.desc} " if self.desc else ""
        # 构建进度条内容
        content = f"{desc_prefix}[{bar}] {percent} ({current}/{self.total}{unit}) {rate}{unit}/s"
        
        # 计算需要填充的空格数，确保清除到行尾
        padding = max(0, self.term_width - len(content) - 1)
        
        # 使用 \r 回到行首，空格填充清除旧内容，确保不换行
        line = f"\r{content}{' ' * padding}"
        
        # 确保输出立即刷新，不使用缓冲
        sys.stdout.write(line)
        sys.stdout.flush()
        self.last_rendered = line

        if current >= self.total:
            self.finish()

    def finish(self):
        if self.finished:
            return
        self.finished = True
        sys.stdout.write("\n")
        sys.stdout.flush()


