#!/usr/bin/env bash
#
# Install Atlas' systemd *user* units.
#
#   bash services/install.sh            # copy units, enable searxng + atlas
#   bash services/install.sh --start    # ...and start them now
#
# What it does:
#   1. copies the unit files into ~/.config/systemd/user/
#   2. runs `systemctl --user daemon-reload`
#   3. enables searxng.service and atlas.service
#   4. prints status
#
# atlas-ui.service is installed but intentionally NOT enabled — it is optional
# and needs a release build of the Tauri app first. Enable it with:
#
#   systemctl --user enable --now atlas-ui.service
#
set -euo pipefail

UNITS=(searxng.service atlas.service atlas-ui.service)
ENABLE=(searxng.service atlas.service)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
# The unit files are written against this path (via systemd's %h specifier).
EXPECTED_ROOT="${HOME}/Documents/Projects/AI/Atlas"

START=0
for arg in "$@"; do
    case "$arg" in
        --start) START=1 ;;
        -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

say() { printf '%s\n' "$*"; }
hr()  { printf '%s\n' "-----------------------------------------------------------------"; }

hr
say "Installing Atlas systemd user units"
hr
say "repo root : ${REPO_ROOT}"
say "unit dir  : ${UNIT_DIR}"

# --- sanity checks ----------------------------------------------------------
# These are the two things that most often make atlas.service fail to start, so
# report them up front rather than letting systemd explain later.
say ""
say "Checks:"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    say "  [ok]   .venv/bin/python"
else
    say "  [WARN] .venv/bin/python is missing or not executable — atlas.service"
    say "         will fail. Create it with: python3 -m venv .venv && \\"
    say "         .venv/bin/pip install -r requirements.txt"
fi

if [[ -f "${REPO_ROOT}/.env" ]]; then
    say "  [ok]   .env present (loaded by atlas.service)"
else
    say "  [note] no .env — fine, all values have defaults. Copy .env.example"
    say "         to .env to set API keys."
fi

if [[ -x "${REPO_ROOT}/ui/src-tauri/target/release/tauri-app" ]]; then
    say "  [ok]   UI release binary built"
else
    say "  [note] UI release binary not built yet — atlas-ui.service will not"
    say "         start until you run: cd ui && npm run tauri build"
fi

if command -v docker >/dev/null 2>&1; then
    say "  [ok]   docker on PATH ($(command -v docker))"
else
    say "  [WARN] docker not found — searxng.service will fail"
fi

# --- path mismatch ---------------------------------------------------------
# The checked-in units hardcode %h/Documents/Projects/AI/Atlas. If this copy of
# the repo lives elsewhere, rewrite the paths on the way in so the installed
# units actually point at it.
RELOCATED=0
if [[ "${REPO_ROOT}" != "${EXPECTED_ROOT}" ]]; then
    RELOCATED=1
    say ""
    say "  [WARN] repo is not at the path the unit files assume:"
    say "           expected: ${EXPECTED_ROOT}"
    say "           actual  : ${REPO_ROOT}"
    say "         Rewriting the installed copies to use the actual path."
fi

# --- install ---------------------------------------------------------------
say ""
mkdir -p "${UNIT_DIR}"

for unit in "${UNITS[@]}"; do
    src="${SCRIPT_DIR}/${unit}"
    if [[ ! -f "${src}" ]]; then
        say "  [FAIL] ${src} not found"
        exit 1
    fi
    if [[ ${RELOCATED} -eq 1 ]]; then
        # %h expands to $HOME at unit-load time; substitute the concrete root.
        sed "s|%h/Documents/Projects/AI/Atlas|${REPO_ROOT}|g" "${src}" > "${UNIT_DIR}/${unit}"
        chmod 0644 "${UNIT_DIR}/${unit}"
    else
        install -m 0644 "${src}" "${UNIT_DIR}/${unit}"
    fi
    say "  installed ${unit}"
done

say ""
say "Reloading the user manager..."
systemctl --user daemon-reload

say "Enabling: ${ENABLE[*]}"
# shellcheck disable=SC2068
systemctl --user enable "${ENABLE[@]}"

if [[ ${START} -eq 1 ]]; then
    say ""
    say "Starting: ${ENABLE[*]}"
    systemctl --user start "${ENABLE[@]}"
fi

# --- status ----------------------------------------------------------------
say ""
hr
say "Status"
hr
systemctl --user --no-pager --full --lines=0 status "${ENABLE[@]}" 2>&1 || true

say ""
say "Enabled units:"
systemctl --user list-unit-files --no-pager 2>/dev/null \
    | grep -Ei 'atlas|searxng' || say "  (none found)"

say ""
hr
if [[ ${START} -eq 1 ]]; then
    say "Done. Follow the logs with:"
else
    say "Done — enabled, but not started yet. Start them with:"
    say ""
    say "  systemctl --user start searxng.service atlas.service"
    say ""
    say "or re-run this script with --start. Follow the logs with:"
fi
say ""
say "  journalctl --user -u atlas -f"
say "  journalctl --user -u searxng -f"
say ""
say "To run without ever logging in graphically, enable lingering once:"
say ""
say "  sudo loginctl enable-linger ${USER}"
say ""
