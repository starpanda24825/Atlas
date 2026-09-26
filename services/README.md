# Atlas as systemd user services

Three units, plus one installer. They are **user** units (they live in
`~/.config/systemd/user/` and are driven by `systemctl --user`), which means
they already run as *you* — `User=`/`Group=` are not just unnecessary there,
systemd rejects them.

| Unit | Runs | Enabled by the installer? |
|---|---|---|
| `searxng.service` | the SearXNG docker compose stack | yes |
| `atlas.service` | `{venv}/bin/python main.py` | yes |
| `atlas-ui.service` | the built Tauri app | **no** — optional |

## Install

```bash
bash services/install.sh            # copy units, enable searxng + atlas, print status
bash services/install.sh --start    # ...and start them now
```

The installer is idempotent, checks the things that most often break
(`.venv/bin/python`, `.env`, the UI build, `docker`) before installing, and
prints status at the end. `atlas-ui.service` is installed but never enabled —
turn it on with `systemctl --user enable --now atlas-ui.service` once you have
built the app.

Everyday commands:

```bash
systemctl --user status atlas searxng          # current state
journalctl --user -u atlas -f                  # follow the daemon log
journalctl --user -u atlas -u searxng -n 50    # last 50 lines of each
systemctl --user restart atlas                 # after a code change
systemctl --user disable --now atlas searxng   # stop starting them at login
```

## Notes on the units

- **Paths are absolute.** systemd does not expand `~`, so the units use the
  `%h` specifier (`/home/<you>`) with the repo at
  `%h/Documents/Projects/AI/Atlas`. If you clone Atlas somewhere else, the
  installer detects the mismatch and rewrites the installed copies to the real
  path — but re-run it after any move.
- **`EnvironmentFile=-.../.env`** — the leading `-` makes the file optional. If
  it were required, `atlas.service` would refuse to start until you created
  `.env`. `core/config.py` loads `.env` itself as well; the directive is there
  so values also arrive as real environment variables, and so you can override
  anything without editing the file.
- **`After=network.target searxng.service`** — `searxng.service` is the
  operative dependency (`network.target` is a system target and is inert from
  the user manager, but is accepted; `systemd-analyze --user verify` is silent
  on it). Atlas' quick search talks to SearXNG over HTTP at
  `core/config.py:SEARXNG_URL`.
- **`atlas.service` has a 45s stop timeout.** The daemon traps SIGTERM and
  stops the wake word listener and the llama.cpp servers, and flushes pending
  memory writes, so it needs more than the default to shut down cleanly.
- **`searxng.service` wraps `docker compose up` in the foreground** (no `-d`),
  which is what makes `Type=simple` correct and streams container logs into the
  journal. It needs your user in the `docker` group. Note the compose file
  itself sets `restart: unless-stopped` on the container, which is redundant now
  that systemd owns the lifecycle — harmless (that policy still respects an
  explicit stop), just duplicated.
- **`atlas-ui.service` runs a graphical app**, so it needs the session
  environment. Started from a login session that is normally inherited; if no
  window appears, run
  `systemctl --user import-environment DISPLAY WAYLAND_DISPLAY XAUTHORITY` and
  restart it. It must be built first — `cd ui && npm run tauri build` — and the
  binary is `ui/src-tauri/target/release/tauri-app`: Tauri names it after the
  Cargo package, not `productName`, because `mainBinaryName` is unset (its
  config schema documents the default as "use the output binary from cargo").

## Running without a graphical session

A user manager is normally started at login. To keep these running on a box you
never log into graphically:

```bash
sudo loginctl enable-linger $USER
```
