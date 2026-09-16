# Music Agent V1 使用教程

> 适用版本：Music Agent V1 Public Snapshot
> 平台：macOS + Apple Music / Music.app
> 状态：Demo implementation；不宣称 production-ready。

> 只想快速启动：先看仓库根目录 `README.md` 的 **Quick start — 5 minutes**。本文件提供完整操作说明和故障排查。

Music Agent 是一个运行在 Mac 本机的 AI 音乐伴侣。它可以把自然语言请求转成推荐、试听、正式播放、播放控制和反馈学习等音乐任务，并在真实外部动作后通过 readback / reconciliation 再决定是否向用户报告成功。

---

## 1. 你可以用它做什么

当前 V1 适合这些场景：

- “推荐几首适合晚上散步的歌。”
- “今天想听安静一点的音乐。”
- “推荐类似《Spring Thief》的歌。”
- “优先推荐一些我最近没怎么接触过的歌。”
- “播放第二首。”
- “试听第四首。”
- “随便播放一首。”
- “暂停 / 继续播放 / 下一首 / 上一首。”
- “我喜欢这首歌。”
- “我不喜欢这首歌。”
- “太吵了，再舒缓一点。”
- “再换一首。”

V1 的重点是：

```text
自然语言
→ 理解当前意图
→ 使用当前会话与长期偏好
→ 推荐 / 选择 / 调用音乐工具
→ 真实执行
→ 读取实际结果
→ 再回复用户
```

---

# 2. 最推荐的体验方式：浏览器界面

如果你只是想把项目跑起来，推荐先使用 Web UI。它比命令行更接近完整产品体验，而且在没有独立后台 runtime 时会自动使用 embedded authority。

完整首次启动流程：

```text
安装 Python 环境
↓
初始化本地数据库
↓
同步 Apple Music Library
↓
配置 DeepSeek API Key
↓
启动 Web UI
↓
开始自然语言交互
```

---

# 3. 环境要求

需要：

- macOS
- Python 3.12+
- Music.app / Apple Music
- DeepSeek API Key（使用默认对话 Provider 时）
- 如果要构建原生 App：Swift 5.10+ / Xcode 或兼容 Command Line Tools

进入项目目录：

```bash
cd /path/to/music-agent
```

---

# 4. 安装 Python 环境

## 方式 A：使用 uv

```bash
uv sync --python 3.12
```

之后可以直接使用：

```bash
uv run music-agent --help
```

## 方式 B：标准 venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

检查安装：

```bash
music-agent --help
```

如果你没有激活虚拟环境，也可以：

```bash
PYTHONPATH=src .venv/bin/python -m music_agent --help
```

---

# 5. 第一次创建本地数据库

Music Agent 默认推荐把本地数据库放在：

```text
~/MusicAgent/music_agent.db
```

先建立目录：

```bash
mkdir -p ~/MusicAgent
```

然后从 Music.app 执行第一次 Library discovery / sync：

```bash
music-agent library-sync \
  --db ~/MusicAgent/music_agent.db
```

如果使用 uv：

```bash
uv run music-agent library-sync \
  --db ~/MusicAgent/music_agent.db
```

首次访问 Music.app 时，如果 macOS 弹出自动化/控制权限提示，请按系统提示允许当前 Terminal、Python 或 Music Agent 访问 Music.app。没有相应系统权限时，与 Music.app 有关的读取和播放功能可能失败。

同步完成后检查数据库状态：

```bash
music-agent status \
  --db ~/MusicAgent/music_agent.db
```

---

# 6. 配置 DeepSeek

默认 conversational provider 是 DeepSeek。

在当前 Terminal 中设置：

```bash
export DEEPSEEK_API_KEY='你的 DeepSeek API Key'
```

不要把真实 API Key 写进 repository、README 或 shell script。

可以先测试一个单轮请求：

```bash
music-agent chat \
  --db ~/MusicAgent/music_agent.db \
  --provider deepseek \
  --message "推荐几首适合晚上听的歌"
```

---

# 7. 启动 Music Agent Web UI

推荐命令：

```bash
music-agent web \
  --db ~/MusicAgent/music_agent.db \
  --provider deepseek
```

正常情况下会：

1. 在 `127.0.0.1` 启动本地服务；
2. 自动选择可用端口；
3. 自动打开浏览器；
4. 显示 Music Agent 主界面。

如果不希望自动打开浏览器：

```bash
music-agent web \
  --db ~/MusicAgent/music_agent.db \
  --provider deepseek \
  --no-browser
```

Web 模式默认会检查是否已有独立 runtime：

```text
有 runtime
→ attach

没有 runtime
→ embed authority 到当前 Web 进程
```

所以第一次体验 Web UI **不需要先单独运行 `music-agent run`**。

退出时可以使用界面的退出功能，或者在启动它的 Terminal 中按：

```text
Ctrl + C
```

---

# 8. 打开以后怎么用

## 8.1 场景推荐

可以直接输入：

```text
推荐几首适合晚上散步的歌。
今天想听安静一点的音乐。
来点适合工作时听的。
```

Music Agent 会优先理解你这一轮的 **Current Intent**，而不是只根据历史偏好机械推荐。

---

## 8.2 相似音乐

例如：

```text
推荐类似《Spring Thief》的歌。
这首不错，再来几首类似的。
```

系统会使用当前歌曲或明确指定的歌曲作为上下文继续推荐。

---

## 8.3 扩展 Discovery

例如：

```text
优先推荐一些没收藏、近期没推荐过的音乐。
想听一些我平时不太会主动找的。
```

Discovery 的目标是扩大候选范围，但 Music Agent 不会仅因为自己的记录中没见过一首歌，就声称“你一定从来没听过”。

---

# 9. 推荐结果怎么操作

推荐结果可能根据真实可执行能力提供不同动作。

## 正式播放

如果歌曲可以通过你的 Apple Music Library 正式播放：

```text
播放第二首。
播放这首。
随便播放一首。
```

Music Agent 会调用真实播放能力，并根据实际播放状态做 readback 后再报告结果。

## Preview 试听

如果歌曲只能试听：

```text
试听第四首。
先试听一下。
```

Preview 是独立于正式播放的试听路径。

当正式音乐正在播放时启动 Preview，Music Agent 的 runtime 会尽量遵循：

```text
正式播放
→ 暂停
→ Preview
→ Preview 结束
→ 恢复之前正式播放状态
```

## 打开 Apple Music

如果存在有效 Apple Music 链接，也可以使用打开 Apple Music 的路径。

---

# 10. 播放控制

可以直接用自然语言：

```text
暂停
继续播放
下一首
上一首
```

也可以通过 CLI：

```bash
music-agent playback pause
music-agent playback play
music-agent playback next
music-agent playback previous
```

查看当前播放状态：

```bash
music-agent playback now-playing
```

具体参数可查看：

```bash
music-agent playback --help
```

---

# 11. 反馈与个性化

Music Agent 支持显式反馈，例如：

```text
我喜欢这首歌。
我不喜欢这首歌。
我喜欢这个方向。
这个方向不太对。
```

其中要区分：

```text
当前会话调整
≠
长期偏好
```

例如：

```text
今晚不要摇滚，换安静一点。
```

通常首先影响当前 Session。

而更明确的长期表达，例如：

```text
我一直很喜欢这种音乐。
以后可以多推荐这种风格。
```

才更接近长期 Preference Evidence。

V1 不会简单使用这种错误等价关系：

```text
Skip = Dislike        ❌
听完 = Like           ❌
Preview 完成 = 长期喜欢 ❌
```

---

# 12. 连续对话

Music Agent 支持 Current Session Context，所以不用每次重新把完整需求说一遍。

例如：

```text
你：推荐几首适合晚上散步的歌。

Music Agent：返回推荐结果

你：再安静一点。

Music Agent：沿当前方向调整

你：试听第二首。

Music Agent：执行第二首 Preview

你：不用，播放第四首。

Music Agent：按当前推荐上下文处理新的明确选择
```

如果“这首”“第二首”“再换一首”等指代无法安全确定，系统应该要求澄清，而不是猜一个对象直接执行。

---

# 13. 命令行聊天模式

如果不想使用 Web UI，可以启动连续 Terminal 会话。

如果只进行普通查询/部分本地工具：

```bash
music-agent chat-session \
  --db ~/MusicAgent/music_agent.db \
  --provider deepseek
```

结束：

```text
/exit
/quit
```

或者发送 EOF。

### Preview / runtime 类工具

`chat-session` 本身不会自动创建 preview/device-safety 的独立 runtime authority。

如果看到：

```text
agent_runtime_offline
```

可以在另一个 Terminal 启动：

```bash
music-agent run \
  --db ~/MusicAgent/music_agent.db
```

然后保持这个 Terminal 开着，再在另一个 Terminal 使用 `chat-session`。

Web UI 一般不需要这一步，因为 Web 默认可以 embed authority。

---

# 14. 后台 Runtime

需要长期运行共享 runtime、周期 refresh 或 audio-safety 时：

```bash
music-agent run \
  --db ~/MusicAgent/music_agent.db
```

默认 runtime 包含 audio-safety monitor。

如果只是调试、明确不需要它：

```bash
music-agent run \
  --db ~/MusicAgent/music_agent.db \
  --no-audio-safety
```

完整参数：

```bash
music-agent run --help
```

---

# 15. 双击启动 Web 版

Repository 内提供：

```text
tools/MusicAgent.command
```

它默认寻找：

```text
<repo>/.venv/bin/python
~/MusicAgent/music_agent.db
```

所以使用前要确保：

1. `.venv` 已创建；
2. 项目已安装；
3. `~/MusicAgent/music_agent.db` 已经完成第一次 Library Sync。

然后可以在 Finder 中双击 `MusicAgent.command`。

如果没有设置：

```text
DEEPSEEK_API_KEY
```

launcher 会提示对话功能不可用。

---

# 16. Native macOS App

> 当前 V1 的 Native App 是开发版 App Shell，不是独立发行的 DMG/App Store 应用。

进入：

```bash
cd app/MusicAgent
```

构建：

```bash
./build-app.sh
```

生成：

```text
app/MusicAgent/dist/Music Agent.app
```

安装到当前用户：

```bash
./install-app.sh
```

安装位置：

```text
~/Applications/Music Agent.app
```

Native App 当前仍依赖本地 repository runtime。它会按下面的顺序寻找 repository：

1. `MUSIC_AGENT_REPO_ROOT`；
2. 兼容变量 `MUSIC_AGENT_REPO`；
3. 默认 checkout：`~/Documents/music-agent`；
4. 如果直接从 repository 中运行未安装的 App，则继续从 App bundle 的上级目录向上寻找。

因此公开仓库可以 clone 到其他位置；如果安装后的 App 无法自动找到 repository，可以在启动环境中显式设置：

```bash
export MUSIC_AGENT_REPO_ROOT='/path/to/music-agent'
```

目标 repository 需要存在：

```text
.venv/bin/python
src/music_agent/
pyproject.toml
```

数据库默认仍为：

```text
~/MusicAgent/music_agent.db
```

如果首次打开 Native App 没有 DeepSeek API Key，它会要求输入，并将凭据保存在 macOS Keychain；不会把 API Key 写进 repository。

App 菜单中也提供：

```text
DeepSeek API Key…
```

用于之后更新凭据。

---

# 17. 常见问题

## 找不到数据库

错误类似：

```text
找不到音乐数据库
```

先执行：

```bash
mkdir -p ~/MusicAgent
music-agent library-sync \
  --db ~/MusicAgent/music_agent.db
```

---

## 没有设置 DeepSeek API Key

Terminal：

```bash
export DEEPSEEK_API_KEY='...'
```

Native App：首次启动时按提示输入，凭据会保存到 Keychain。

---

## Preview 报 `agent_runtime_offline`

这通常出现在直接使用 `chat` / `chat-session`，而独立 runtime 没有运行时。

另开一个 Terminal：

```bash
music-agent run \
  --db ~/MusicAgent/music_agent.db
```

Web UI 默认 embed 模式一般不需要额外启动 runtime。

---

## Music.app 播放/读取失败

检查：

1. Music.app 是否可正常使用；
2. macOS 是否阻止当前 Terminal / Python / Music Agent 控制 Music.app；
3. 当前目标歌曲是否确实存在可执行路径；
4. 是否存在有效 Library binding 或 Preview resource。

Music Agent 在无法确认真实状态时会优先 fail closed，不应该把失败伪装成成功。

---

## Native App 提示找不到开发运行环境

Native App V1 不是完全自包含应用。

最简单的开发布局是：

```text
~/Documents/music-agent
├── .venv/bin/python
├── pyproject.toml
└── src/music_agent/
```

也可以通过开发环境变量覆盖 repository 或 database 路径，但普通体验建议先使用默认布局。

---

# 18. V1 暂时不做什么

V1 的产品核心不包括：

- Apple Music Like 持久写入；
- Add to Library；
- Rating 写入；
- Create Playlist / Edit Playlist；
- 完整 Playlist 管理；
- 手机 App；
- 云同步；
- 跨设备 Playback；
- 完整 Apple Music Library 浏览器；
- 社区 / 评论；
- 商业化功能。

内部 repository 可能保留部分 capability / safety 基础设施，但这不代表这些能力属于 V1 对用户开放的产品功能。

---

# 19. 最快 5 分钟上手清单

如果你只想尽快看到 Music Agent 跑起来：

```bash
# 1. 进入项目
cd /path/to/music-agent

# 2. 建环境
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# 3. 建数据库 + 同步 Library
mkdir -p ~/MusicAgent
music-agent library-sync --db ~/MusicAgent/music_agent.db

# 4. 配置模型
export DEEPSEEK_API_KEY='你的 Key'

# 5. 启动产品 UI
music-agent web \
  --db ~/MusicAgent/music_agent.db \
  --provider deepseek
```

然后在界面里输入：

```text
推荐几首适合晚上散步的歌。
```

如果能正常返回推荐，再测试：

```text
试听第二首。
播放第一首。
暂停。
继续播放。
我喜欢这首歌。
再安静一点。
```

这基本就覆盖了 V1 最核心的一条产品闭环：

```text
自然语言需求
→ 推荐
→ 试听 / 播放
→ 反馈
→ 连续调整
```
