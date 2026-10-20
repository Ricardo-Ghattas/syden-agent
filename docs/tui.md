# SYDEN terminal application

## Install and launch

Python 3.12+ and uv are required. From this checkout:

```bash
uv sync --locked
uv run syden --help
uv tool install --editable .
syden --help
```

The tool installs into uv's own environment; neither activating `.venv` nor setting
`PYTHONPATH` is necessary. If `syden` is not on PATH, run `uv tool update-shell`,
then open a new terminal. The executable is normally in `~/.local/bin` on Linux;
`uv tool dir --bin` prints the actual directory. Editable installs require keeping
the checkout at its installed location. For an independent installation, use
`uv tool install .` instead.

With an editable installation, plain `syden` automatically reuses the installed
checkout's existing `.env` and `data/processed`, even when launched from another
directory. You do not need to copy credentials or pass configuration flags.
`syden ingest FILE` also defaults to that checkout's `data` directory.

Configuration file selection, in order:

1. Explicit `--env-file PATH`.
2. `~/.config/syden/.env` (or `$XDG_CONFIG_HOME/syden/.env`), if present.
3. The editable checkout's `.env`.

Existing environment variables always take precedence over values in the selected
file. SYDEN locates its checkout from the installed package location and verifies
its project identity; it does not implicitly load arbitrary `.env` files from the
current working directory. Credentials are never printed or copied.

For a standalone installation without a source checkout, configure the user-level
file with `OPENAI_API_KEY` and `OPENAI_MODEL`; `.env.example` lists provider and
budget options. Keep this credential file private, for example with
`chmod 600 ~/.config/syden/.env`.

`SYDEN_DATA_DIR` overrides the default data directory for both installation types.
Without it, editable installations use their checkout's `data` directory;
standalone installations use `~/.local/share/syden`. `--processed-dir PATH`
overrides the collection for the TUI or a question. Explicit relative paths
resolve against the invoking working directory; automatic checkout paths do not
change when you run from a different directory.

```bash
syden                                   # interactive TUI
syden ingest ~/Downloads/boq.xlsx        # default data collection
syden ingest ./spec.pdf --data-dir ./data
syden --processed-dir ./data/processed   # choose an explicit collection
syden ask "What is the quantity of item 001?" --json
syden --processed-dir ./data/processed ask "Compare these items"
syden eval --mode offline
syden chat                              # legacy line-oriented chat
```

Global options (`--env-file`, `--processed-dir`, `--debug`) go before the
subcommand. `ask --processed-dir PATH` also remains available through the legacy
question parser. Use `syden COMMAND --help` for command-specific options.
Legacy `python main.py FILE`, `main.py ask`, `main.py chat`, and `main.py eval`
commands retain their existing behavior and relative-path defaults.

## Keyboard and conversation

| Action | Shortcut or command |
| --- | --- |
| Submit question | Enter |
| Insert newline | Ctrl+O or F4 |
| Start a new session | Ctrl+N or `/new` |
| Clear display and model context, keeping session ID | Ctrl+L or `/clear` |
| Cancel current request | Esc |
| Exit safely | Ctrl+Q or `/exit` |
| Command help | `/help` |
| References from latest answer | `/sources` |
| Focus history / composer | Tab / Shift+Tab |
| Scroll focused history | PageUp / PageDown, mouse wheel |

Ctrl+O sends a distinct control byte on Linux terminals; F4 is a separate
function-key fallback. Neither relies on distinguishing Shift+Enter or Ctrl+J
from Enter. Text editing keeps its usual bindings, including Ctrl+A for line
start and Ctrl+C to copy selected composer text. Select answer text with the
mouse; terminal Shift+selection provides another copying route. Clipboard
integration depends on terminal support. A real process interrupt also restores
the terminal and closes the client.

The composer stays editable while a request runs, so you can draft a follow-up.
Submitting another question is blocked until that request finishes. Local slash
commands and shortcuts remain available. New messages scroll into view; previous
messages remain available until a reset. Memory is in-process only, and the
existing session turn/context budgets remain enforced. The index is loaded once;
after ingesting additional documents, reopen SYDEN to reload the collection.

## Execution and cancellation

Textual thread workers load the collection and call the existing `ask` function.
Workers post thread-safe messages; only the UI thread updates widgets. The header
starts with **Not checked**, then **Response received** after an actual provider
response. It never claims an untested connection is online. Activity descriptions
come from explicit orchestration hooks, not model reasoning or inferred prose.
`--debug` shows sanitized activity names, status, and session identifiers.

Each turn executes against a deep copy of conversation state. Only an accepted,
uncancelled completion replaces the UI session. Esc sets a cooperative stop
signal: the agent checks it between calls and before accepting an answer. A
currently blocking HTTP/tool call cannot be forcibly stopped safely. Cancellation
and safe exit wait for it to return or time out (bounded by the configured API and
agent timeouts). The UI stays responsive and shows that it is waiting. No further
question starts during that wait. `/new` or `/clear` resets active memory
immediately; the old worker's eventual result is discarded. Exit then closes the
shared OpenAI client. Remote work may already have been processed or billed;
cancellation does not retract requests sent to the provider.

## Rendering and streaming

The TUI and noninteractive terminal commands share Rich Markdown, responsive
comparison tables, and source grouping. Filenames appear once per document;
adjacent worksheet rows merge while noncontiguous ranges stay separate. All
uncertainty and completeness warnings remain visible under Notes. Full exact
citation objects remain in `AgentResponse` and `--json` output.

The OpenAI adapter currently uses synchronous `responses.create`. Tool rounds,
citation checks, and strict JSON validation require the assembled response. Real
streaming would require a separate SDK stream accumulator with validated tool
calls and final JSON handling, including provider compatibility tests. It is
intentionally deferred: the TUI shows real activity and renders only validated
final answers. It never simulates token streaming or displays partial JSON.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run syden eval --mode offline
uv build --wheel
```

Headless Textual tests exercise keyboard submission, real orchestration with a
mocked SDK, multi-turn context, blocked workers, duplicate prevention, cancellation,
session resets, errors, Markdown, citations, long content, resizing, and narrow
terminals. CLI tests cover routing, configuration precedence, entry point discovery,
client cleanup, and JSON output. No paid requests are required. Linux pseudoterminal
smoke checks verify installed-command launch, Ctrl+O/Ctrl+N/Ctrl+Q handling, and
restoration of terminal settings after exit.
