# Music Agent

Music Agent is a local-first AI music companion for Apple Music on macOS. It turns natural-language requests into structured music workflows, executes real playback or preview actions, and verifies external state before reporting success.

> **V1 status:** demo implementation complete and engineering-validated on macOS.
> **Production-ready:** not claimed.

## What V1 can do

- natural-language music requests through DeepSeek or Codex-oriented provider paths;
- personalized recommendation batches;
- Apple Music library search and formal playback;
- catalog preview playback;
- play, pause, resume, next and previous controls;
- explicit recommendation-item selection and structured follow-up actions;
- durable like/dislike feedback and preference learning;
- current playback and recommendation context;
- fail-closed action execution with readback/reconciliation;
- a local browser UI;
- a native macOS app shell;
- structured Product Eval contracts, evidence, scorecards and runner tooling.

The core reliability rule is simple:

```text
LLM understands the request
        ↓
Code owns workflow and critical state
        ↓
Tools execute
        ↓
Readback verifies external reality
        ↓
Presentation reports the verified result
```

## Repository layout

```text
src/music_agent/   Python runtime, workflows, providers and local web UI
app/MusicAgent/    Native macOS app shell
tests/             Automated regression and contract tests
eval/              Product acceptance contract, evidence and scorecards
tools/             Evaluation, maintenance and local utility scripts
docs/decisions/    Curated architecture decision records
docs/USER_GUIDE.zh-CN.md  Complete Chinese user guide
```

This public snapshot intentionally omits private project-history material, local host configuration, local databases and development-agent configuration.

The `docs/decisions/` directory contains only the architecture decisions that are useful for understanding the public V1 implementation. Internal phase closeouts, handoffs, capability-probe history and private project notes are intentionally excluded.

## Requirements

- macOS
- Python 3.12+
- Music.app / Apple Music for Apple Music integration
- Swift 5.10+ / Xcode or compatible Command Line Tools for the native app
- a supported model provider for conversational features

The Python package has one declared runtime dependency:

```text
jsonschema >= 4.23, < 5
```

The Swift package targets macOS 14 or newer.

## Quick start — 5 minutes

For a first run, the browser UI is the easiest path. A more detailed Chinese walkthrough is available in [docs/USER_GUIDE.zh-CN.md](docs/USER_GUIDE.zh-CN.md).

```bash
# 1. Create and activate the environment
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# 2. Create the local data directory and sync your Music.app library
mkdir -p ~/MusicAgent
music-agent library-sync --db ~/MusicAgent/music_agent.db

# 3. Configure the default conversational provider
export DEEPSEEK_API_KEY='YOUR_KEY_HERE'

# 4. Start the local product UI
music-agent web \
  --db ~/MusicAgent/music_agent.db \
  --provider deepseek
```

Then try a request such as:

```text
推荐几首适合晚上散步的歌。
```

Useful follow-ups include:

```text
试听第二首。
播放第一首。
暂停。
继续播放。
我喜欢这首歌。
再安静一点。
```

On first access to Music.app, macOS may ask for Automation permissions. Music-related reading or playback can fail until those permissions are granted.

## Setup

Using `uv`:

```bash
uv sync --python 3.12
```

Or with a standard virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The package exposes the `music-agent` command. You can also run the module directly:

```bash
music-agent --help
# or
PYTHONPATH=src python -m music_agent --help
```

## Local database

Most commands require an explicit SQLite store path. The bundled local launcher and native shell use this default convention:

```text
~/MusicAgent/music_agent.db
```

Create the directory if needed, then run a Music.app library discovery/sync:

```bash
mkdir -p ~/MusicAgent
music-agent library-sync --db ~/MusicAgent/music_agent.db
```

Inspect the resulting store/runtime status with:

```bash
music-agent status --db ~/MusicAgent/music_agent.db
```

Local databases are intentionally ignored by Git.

## Provider configuration

### DeepSeek

DeepSeek is the default conversational provider. Supply its credential through the environment rather than committing it:

```bash
export DEEPSEEK_API_KEY='...'
```

Then inspect or run an interactive session:

```bash
music-agent chat-session --help
music-agent chat-session \
  --db ~/MusicAgent/music_agent.db \
  --provider deepseek
```

### Codex

Codex is also supported as a provider path. Exact runtime options are exposed by the CLI:

```bash
music-agent chat-session --help
```

Provider selection is explicit; there is no automatic provider fallback.

## Run the local product shell

The browser product shell runs on loopback and supports chat, now-playing state, playback controls, recommendation cards, preview playback and library playback.

```bash
music-agent web \
  --db ~/MusicAgent/music_agent.db \
  --provider deepseek
```

Use `--no-browser` to avoid opening a tab automatically, or inspect all options with:

```bash
music-agent web --help
```

The repository also contains a double-clickable local launcher:

```text
tools/MusicAgent.command
```

It uses `~/MusicAgent/music_agent.db` by default and accepts `MUSIC_AGENT_DB` as an override.

## Run the foreground runtime

The foreground runtime hosts shared agent state, periodic refresh work and audio-safety behavior:

```bash
music-agent run --db ~/MusicAgent/music_agent.db
```

Useful options include refresh intervals, client permissions and `--no-audio-safety`. See:

```bash
music-agent run --help
```

## Native macOS app

The Swift app shell lives in `app/MusicAgent/`.

Build the development app bundle:

```bash
cd app/MusicAgent
./build-app.sh
```

The bundle is assembled at:

```text
app/MusicAgent/dist/Music Agent.app
```

Install it for the current user:

```bash
./install-app.sh
```

The install helper places the app in `~/Applications/Music Agent.app`.

Signing, notarization and DMG packaging are not part of this V1 repository workflow.

## Product behavior and safety

Music Agent distinguishes between formal library playback, catalog previews and control actions instead of treating every request as a generic model response.

For critical external actions the intended contract is:

```text
user intent
→ authorized target/action
→ execution
→ readback
→ reconciliation
→ verified response
```

Ambiguous outcomes fail closed rather than being presented as success.

The project also includes runtime audio-safety logic designed to avoid unintended playback through built-in speakers when a private output device disappears.

## Feedback and personalization

V1 has a durable feedback/learning loop:

```text
Recommendation / playback
        ↓
Explicit feedback
        ↓
Feedback interpretation
        ↓
Learning application
        ↓
Updated preference state
        ↓
Future recommendation context
```

Long-term personalization is stored as structured state rather than relying on an indefinitely growing chat transcript.

## Product Eval

The Product Eval assets live under `eval/`:

```text
eval/product_acceptance_v1.yaml
eval/product_eval_run.schema.json
eval/product_scorecard_v1.md
eval/evidence/
eval/runs/
```

Runner tooling lives under `tools/`:

```text
tools/product_eval_runner.py
tools/run_product_eval.py
tools/record_product_uat.py
```

The current structured scorecard records:

- corpus size: **15** top-level tasks;
- execution coverage: **2/15 (13.33%)**;
- acceptance on the executed slice: **2/2 (100%)**;
- wrong-action rate on the recorded slice: **0/6**;
- the recorded EVAL-05 engineering regression: **5421/5421 PASS**.

The 100% acceptance figure is **not** whole-corpus coverage. Thirteen top-level Product Eval cases remain `NOT_RUN` in the recorded scorecard.

## Tests

Run the Python test suite from the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

The Product Eval suite is intentionally separate from ordinary engineering regression so that test-suite success is not presented as complete product acceptance coverage.

## Privacy and repository hygiene

Do not commit:

- API keys or other credentials;
- local SQLite databases;
- `.mcp.json` or other host-specific model-tool configuration;
- virtual environments or build products;
- private project-history documents;
- machine-specific logs or traces.

The repository `.gitignore` includes the common local/runtime cases.

## V1 boundaries

Music Agent V1 is a demo implementation, not a production-readiness claim. Important boundaries include:

- Apple Music integration is macOS-specific;
- local persistence and local runtime assumptions remain part of the design;
- conversational features require a configured provider;
- Product Eval execution coverage is still partial;
- signing/notarization/distribution packaging is not included;
- future recommendation and continuation work may extend beyond the V1 contract.

## License

The package metadata currently declares the project as **Proprietary**. Publishing the source repository does not grant an open-source license unless that licensing decision is changed explicitly.
