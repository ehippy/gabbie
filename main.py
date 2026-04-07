#!/usr/bin/env python3
"""
Gabbie - Voice Gateway for Raspberry Pi 3

Entry point for CLI. Use:
  gabbie          - Launch TUI (auto-starts daemon)
  gabbie daemon   - Daemon management
  gabbie tui      - Launch TUI only
  gabbie config   - Configuration management
  gabbie devices  - List audio devices
"""

from src.cli import run

if __name__ == "__main__":
    run()
