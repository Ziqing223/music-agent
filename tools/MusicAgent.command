#!/bin/zsh
# MusicAgent.command — P17-B minimal local product shell launcher.
#
# Double-click to start: this Terminal window hosts the local service and the
# UI opens as a browser tab (http://127.0.0.1:<port>). Exit either from the
# browser's 退出应用 button or here with Ctrl+C; on startup failure the window
# stays open so the error is readable.
#
# The launcher supplies the product's default live store (the same convention
# as the tools/ repair scripts: $HOME/MusicAgent/music_agent.db). Override with
# MUSIC_AGENT_DB, or pass your own --db after the defaults — the last --db wins.
# All other real switches (--provider codex, --model, --mode attach/embed,
# --port, ...) are the production CLI's own; pass them straight through.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "音乐助手：找不到虚拟环境 $PY —— 请先在仓库根目录搭建环境。"
  read -n 1 -s '?' >/dev/null 2>&1 || true
  exit 1
fi

DB="${MUSIC_AGENT_DB:-$HOME/MusicAgent/music_agent.db}"
if [[ ! -f "$DB" ]]; then
  echo "音乐助手：找不到音乐数据库 $DB —— 请先完成首次数据同步，"
  echo "或用环境变量 MUSIC_AGENT_DB 指定数据库路径。"
  read -n 1 -s '?' >/dev/null 2>&1 || true
  exit 1
fi

echo "音乐助手：正在启动本地服务，随后会自动打开浏览器…（数据库：$DB）"
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "注意：未设置 DEEPSEEK_API_KEY，对话功能将不可用（播放器与控制仍然可用）。"
fi

PYTHONPATH="$ROOT/src" "$PY" -m music_agent.cli web --db "$DB" "$@"
STATUS=$?
if (( STATUS != 0 )); then
  echo ""
  echo "音乐助手：启动失败（退出码 $STATUS）——错误信息在上方。按任意键关闭窗口。"
  read -n 1 -s '?' >/dev/null 2>&1 || true
fi
exit $STATUS