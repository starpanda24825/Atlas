"""
Atlas — wake word sample collection and training prerequisites.

This is the tooling around custom "hey atlas" training: recording reference
clips, scoring them against the current detector, and reporting what the
training run still needs. The training itself is a separate, heavy step (see
below); the listening daemon is entirely unaware of this module.

Three subcommands::

    # Record training clips into models/wake_word/
    .venv/bin/python -m voice.wake_word_samples record --kind positive
    .venv/bin/python -m voice.wake_word_samples record --kind negative

    # Score the clips you just recorded against the current model
    .venv/bin/python -m voice.wake_word_samples score

    # What is present, what is missing, and the exact commands to run
    .venv/bin/python -m voice.wake_word_samples doctor

Why recording is its own helper
-------------------------------
openWakeWord consumes single-channel **16-bit PCM, 16 kHz** audio. A float32
recording silently scores wrong, and the directory has to exist before
``soundfile.write`` is called. Both are easy to get wrong in a one-liner, so
they are handled once, here.

The honest state of custom training
-----------------------------------
The installed openWakeWord (0.6.x) has **no** CLI that takes reference clips and
emits a wake word model. There are two real routes:

1. **Standalone model** — ``python -m openwakeword.train --training_config ...``
   in three stages (``--generate_clips``, ``--augment_clips``, ``--train_model``).
   This synthesises its own positives with Piper and *does not* consume your own
   recordings directly; it needs a YAML config, a Piper checkout, and room/IR +
   background-noise corpora. It emits ``hey_atlas.onnx`` (and attempts a
   ``.tflite`` conversion). This is the route that produces a file the listening
   daemon loads.

2. **Speaker verifier** — ``openwakeword.custom_verifier_model.train_custom_verifier``
   is a *function* (no CLI) that trains a small logistic-regression verifier from
   **your own** positive and negative clips, saved as a ``.joblib``. It gates an
   existing base model rather than replacing it, and the current daemon does not
   consume it. Useful for speaker-specific rejection later, not for "hey atlas".

Until a custom model exists, the daemon runs a pre-built stand-in and says so.
Because the daemon loads ONNX (tflite-runtime is broken against NumPy 2.x here),
copy the **``hey_atlas.onnx``** — not the ``.tflite`` — into
``WAKE_WORD_MODEL_DIR``.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from core.config import (
    SAMPLE_RATE,
    WAKE_WORD_MODEL_DIR,
    WAKE_WORD_MODEL_NAME,
    WAKE_WORD_SAMPLE_SECONDS,
    WAKE_WORD_THRESHOLD,
)

logger = logging.getLogger(__name__)

DEFAULT_COUNT: int = 5
DEFAULT_PREFIX: str = "positive"
WAV_SUBTYPE: str = "PCM_16"

#: Python modules the standalone training pipeline imports. None are needed to
#: *listen*; they are only required to train, which is why they are absent from
#: requirements.txt.
TRAINING_PYTHON_DEPS: tuple[str, ...] = (
    "torch",
    "torchinfo",
    "torchmetrics",
    "onnx",
    "piper",  # imported from a checkout, not pip — checked as a hint
)

#: The three stages of the official training CLI, in order.
TRAINING_STAGES: tuple[tuple[str, str], ...] = (
    ("--generate_clips", "synthesise positive clips with Piper"),
    ("--augment_clips", "mix in room impulse responses and background noise"),
    ("--train_model", "train and export hey_atlas.onnx / .tflite"),
)


# ---------------------------------------------------------------------------
# Paths and dependencies
# ---------------------------------------------------------------------------


def ensure_model_dir(directory: str | Path | None = None) -> Path:
    """Return ``directory`` (default the configured model dir), created."""
    target = Path(directory) if directory else WAKE_WORD_MODEL_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def custom_model_path(directory: str | Path | None = None) -> Path | None:
    """The trained model the daemon would load, if it exists (ONNX first)."""
    target = Path(directory) if directory else WAKE_WORD_MODEL_DIR
    for suffix in (".onnx", ".tflite"):
        candidate = target / f"{WAKE_WORD_MODEL_NAME}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _import_sounddevice() -> Any:
    """sounddevice, with the actionable PortAudio message from the daemon."""
    from voice.wake_word import WakeWordDaemon

    return WakeWordDaemon._import_sounddevice()


def _import_soundfile() -> Any:
    try:
        import soundfile as sf
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"soundfile could not be imported ({exc}). Install it with "
            "`.venv/bin/pip install soundfile`."
        ) from exc
    return sf


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def record_samples(
    count: int = DEFAULT_COUNT,
    *,
    seconds: float = WAKE_WORD_SAMPLE_SECONDS,
    directory: str | Path | None = None,
    prefix: str = DEFAULT_PREFIX,
    prompt: str | None = None,
    input_fn: Any = input,
    sample_rate: int = SAMPLE_RATE,
    device: Any = None,
) -> list[Path]:
    """Record ``count`` clips into ``directory`` and return their paths.

    Each clip is ``seconds`` long, mono, 16 kHz, 16-bit PCM — the format
    openWakeWord expects. The directory is created first, and the file is only
    written after the recording finishes, so a failed take does not leave a
    corrupt WAV behind.
    """
    target = ensure_model_dir(directory)
    sd = _import_sounddevice()
    sf = _import_soundfile()

    frames = int(round(max(0.5, float(seconds)) * sample_rate))
    say = prompt or f"say '{WAKE_WORD_MODEL_NAME.replace('_', ' ')}'"
    written: list[Path] = []

    for index in range(1, max(1, int(count)) + 1):
        path = target / f"{prefix}_{index - 1}.wav"
        input_fn(f"[{index}/{count}] Press Enter, then {say} ...")
        try:
            audio = sd.rec(frames, samplerate=sample_rate, channels=1,
                           dtype="int16", device=device)
            sd.wait()
        except Exception as exc:
            raise RuntimeError(f"recording failed ({exc})") from exc
        sf.write(str(path), audio, sample_rate, subtype=WAV_SUBTYPE)
        logger.info("recorded %s", path)
        print(f"    saved {path.name}")
        written.append(path)

    print(f"\n{len(written)} clip(s) in {target}")
    return written


def mic_check(
    seconds: float = 3.0,
    *,
    device: Any = None,
    play: bool = True,
    sample_rate: int = SAMPLE_RATE,
    input_fn: Any = input,
    max_rounds: int | None = None,
) -> dict[str, Any]:
    """Record, play the take straight back, and loop until you are happy.

    This is the "is my microphone any good?" step. It records ``seconds`` of
    audio in the exact format the wake word daemon uses, prints the peak/RMS
    level so a silent or clipping input is obvious, then plays it back through
    the speakers. Press Enter to record again, or answer ``y``/``q`` to finish.

    Returns ``{rounds, peak_dbfs, rms_dbfs, verdict}``.
    """
    import numpy as np

    sd = _import_sounddevice()
    frames = int(round(max(1.0, float(seconds)) * sample_rate))
    rounds = 0
    peak_dbfs = float("-inf")
    rms_dbfs = float("-inf")

    while True:
        rounds += 1
        input_fn(
            f"[take {rounds}] Press Enter, then talk normally for "
            f"{seconds:.1f}s..."
        )
        try:
            audio = sd.rec(frames, samplerate=sample_rate, channels=1,
                           dtype="int16", device=device)
            sd.wait()
        except Exception as exc:
            raise RuntimeError(f"recording failed ({exc})") from exc

        samples = audio.reshape(-1).astype("float32") / 32768.0
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
        peak_dbfs = 20 * float(np.log10(max(peak, 1e-6)))
        rms_dbfs = 20 * float(np.log10(max(rms, 1e-6)))

        if play:
            print("    playing it back...")
            sd.play(audio, sample_rate, device=device)
            sd.wait()

        verdict = "good"
        if peak_dbfs < -45:
            verdict = "far too quiet — check the right input device is selected/plugged in"
        elif peak >= 0.999:
            verdict = "clipping — lower the mic gain in your system settings"
        elif peak_dbfs < -25:
            verdict = "quiet but usable — move closer or raise the gain"
        print(
            f"    level: peak {peak_dbfs:.1f} dBFS, average {rms_dbfs:.1f} dBFS "
            f"-> {verdict}"
        )

        if not play:
            break
        answer = input_fn(
            "    Did that sound like you? [Enter = record again, y = happy, q = quit] "
        ).strip().lower()
        if answer in ("", "r"):
            if max_rounds is not None and rounds >= max_rounds:
                break
            continue
        break

    return {
        "rounds": rounds,
        "peak_dbfs": round(peak_dbfs, 1),
        "rms_dbfs": round(rms_dbfs, 1),
        "verdict": verdict,
    }


def list_samples(
    directory: str | Path | None = None, glob: str = "positive_*.wav"
) -> list[Path]:
    """The sample WAVs in ``directory`` matching ``glob``, sorted."""
    target = Path(directory) if directory else WAKE_WORD_MODEL_DIR
    if not target.is_dir():
        return []
    return sorted(target.glob(glob))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def load_detector() -> Any:
    """Load the same detector the daemon uses (custom if present, else standby)."""
    from voice.wake_word import WakeWordDaemon

    return WakeWordDaemon()


def score_wav(detector: Any, path: str | Path) -> dict[str, Any]:
    """Frame-level max/mean wake score for one clip."""
    key = getattr(detector, "model_key", None)
    try:
        predictions = detector.model.predict_clip(str(path))
    except Exception as exc:
        return {"file": Path(path).name, "error": f"{type(exc).__name__}: {exc}"}

    scores: list[float] = []
    for entry in predictions or []:
        if isinstance(entry, dict):
            if key in entry:
                scores.append(float(entry[key]))
            elif entry:
                scores.append(float(max(entry.values())))
        elif entry is not None:
            scores.append(float(entry))
    if not scores:
        return {"file": Path(path).name, "error": "no prediction frames"}
    return {
        "file": Path(path).name,
        "max": round(max(scores), 4),
        "mean": round(sum(scores) / len(scores), 4),
        "frames": len(scores),
    }


def score_samples(
    paths: Sequence[str | Path] | None = None,
    *,
    directory: str | Path | None = None,
    glob: str = "positive_*.wav",
    detector: Any = None,
) -> dict[str, Any]:
    """Score every matching clip and summarise how strongly it fires.

    Returns ``{count, clips, max, mean_of_max, min_of_max, suggested_threshold,
    model_source, custom}``. ``suggested_threshold`` is derived from the weakest
    clip so a threshold below it would accept all of your own takes.
    """
    clips = [Path(p) for p in paths] if paths else list_samples(directory, glob)
    detector = detector or load_detector()

    results = [score_wav(detector, clip) for clip in clips]
    scored = [r for r in results if "max" in r]
    maxima = [r["max"] for r in scored]

    suggestion: float | None = None
    if maxima:
        weakest = min(maxima)
        suggestion = round(min(max(weakest * 0.9, 0.1), 0.9), 2)

    return {
        "count": len(results),
        "scored": len(scored),
        "clips": results,
        "max": round(max(maxima), 4) if maxima else None,
        "mean_of_max": round(sum(maxima) / len(maxima), 4) if maxima else None,
        "min_of_max": round(min(maxima), 4) if maxima else None,
        "suggested_threshold": suggestion,
        "model_source": getattr(detector, "model_source", None),
        "custom": bool(getattr(detector, "using_custom_model", False)),
    }


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


def training_status(directory: str | Path | None = None) -> dict[str, Any]:
    """Report what is present and what the training run still needs."""
    target = Path(directory) if directory else WAKE_WORD_MODEL_DIR
    recorded = [
        path.name for path in sorted(target.glob("*.wav"))
    ] if target.is_dir() else []
    missing_deps = [name for name in TRAINING_PYTHON_DEPS if name != "piper" and not _module_available(name)]

    model = custom_model_path(target)
    verifier = target / f"{WAKE_WORD_MODEL_NAME}.joblib"
    return {
        "model_dir": str(target),
        "model_dir_exists": target.is_dir(),
        "model_name": WAKE_WORD_MODEL_NAME,
        "custom_model": str(model) if model else None,
        "custom_model_present": model is not None,
        "verifier": str(verifier) if verifier.is_file() else None,
        "verifier_present": verifier.is_file(),
        "recorded_clips": recorded,
        "positive_clips": [n for n in recorded if n.startswith("positive_")],
        "negative_clips": [n for n in recorded if n.startswith("negative_")],
        "missing_python_deps": missing_deps,
        "piper_present": _module_available("piper"),
        "threshold": WAKE_WORD_THRESHOLD,
    }


def verifier_commands() -> list[str]:
    """The recommended local two-step path (record/test, then train)."""
    return [
        ".venv/bin/python -m voice.wake_word_samples mic-check",
        ".venv/bin/python -m voice.wake_word_train --record",
    ]


def training_commands(config_path: str = "train_hey_atlas.yaml") -> list[str]:
    """The three official ``openwakeword.train`` stage commands, in order."""
    return [
        f".venv/bin/python -m openwakeword.train --training_config {config_path} {flag}"
        for flag, _ in TRAINING_STAGES
    ]


def format_status(status: dict[str, Any]) -> str:
    """Human-readable rendering of :func:`training_status`."""
    lines = [
        "Wake word training status",
        f"  model dir      : {status['model_dir']}",
        f"  model name     : {status['model_name']}",
        f"  custom model   : {status['custom_model'] or 'not trained yet'}",
        f"  positive clips : {len(status['positive_clips'])}",
        f"  negative clips : {len(status['negative_clips'])}",
        f"  missing deps   : {', '.join(status['missing_python_deps']) or 'none'}",
        f"  piper checkout : {'present' if status['piper_present'] else 'not installed'}",
    ]
    if status.get("verifier_present"):
        lines += ["", f"  verifier       : {status['verifier']} (in use)"]
    if status["custom_model_present"]:
        lines += [
            "",
            f"{Path(status['custom_model']).name} is in place and takes priority.",
            "Restart Atlas (or the wake word daemon) to pick it up.",
        ]
    elif status.get("verifier_present"):
        lines += [
            "",
            "A trained verifier is installed — the daemon hears your (trained)",
            f"'{status['model_name'].replace('_', ' ')}'. Retrain any time with:",
            *[f"  {cmd}" for cmd in verifier_commands()],
        ]
    else:
        lines += [
            "",
            "Nothing trained yet — the daemon is running a pre-built stand-in.",
            "",
            f"Train '{status['model_name'].replace('_', ' ')}' from your own recordings:",
            *[f"  {cmd}" for cmd in verifier_commands()],
            "",
            "That fits a small verifier over openWakeWord's own features; the",
            "daemon runs it on top of a base model and uses its score directly.",
            "No downloads, no Piper, and it takes about a minute.",
            "",
            "(The alternative is a full standalone network via the heavy pipeline:",
            *[f"  {cmd}" for cmd in training_commands()],
            "  which needs a Piper checkout, room/IR + background-noise corpora",
            "  and the missing deps above.)",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voice.wake_word_samples",
        description="Record, score and check prerequisites for a custom wake word.",
    )
    sub = parser.add_subparsers(dest="command")

    record = sub.add_parser("record", help="record training clips")
    record.add_argument("--kind", choices=("positive", "negative"), default="positive")
    record.add_argument("--count", type=int, default=DEFAULT_COUNT)
    record.add_argument("--seconds", type=float, default=WAKE_WORD_SAMPLE_SECONDS)
    record.add_argument("--dir", default=None)

    mic = sub.add_parser("mic-check", help="record and play back to test your microphone")
    mic.add_argument("--seconds", type=float, default=3.0)
    mic.add_argument("--no-play", action="store_true", help="record without playing back")

    score = sub.add_parser("score", help="score recorded clips against the model")
    score.add_argument("--dir", default=None)
    score.add_argument("--glob", default="positive_*.wav")

    sub.add_parser("doctor", help="show prerequisites and the correct commands")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = args.command or "doctor"

    if command == "record":
        if args.kind == "positive":
            prompt = f"say '{WAKE_WORD_MODEL_NAME.replace('_', ' ')}'"
            prefix = "positive"
        else:
            prompt = "say something else (a full sentence that is NOT the wake word)"
            prefix = "negative"
        record_samples(
            args.count,
            seconds=args.seconds,
            directory=args.dir,
            prefix=prefix,
            prompt=prompt,
        )
        return 0

    if command == "mic-check":
        try:
            report = mic_check(args.seconds, play=not args.no_play)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"\nFinished after {report['rounds']} take(s). "
              f"Final level: peak {report['peak_dbfs']} dBFS, "
              f"average {report['rms_dbfs']} dBFS ({report['verdict']}).")
        return 0

    if command == "score":
        report = score_samples(directory=args.dir, glob=args.glob)
        if not report["count"]:
            print("No clips found. Record some first:")
            print("  python -m voice.wake_word_samples record --kind positive")
            return 1
        if not report["custom"]:
            print(f"WARNING: scoring against a stand-in ({report['model_source']}), "
                  "so these numbers are not about your trained model.\n")
        for clip in report["clips"]:
            if "error" in clip:
                print(f"  {clip['file']:<20} ERROR {clip['error']}")
            else:
                print(f"  {clip['file']:<20} max={clip['max']:.3f} mean={clip['mean']:.3f}")
        print(
            f"\nlowest max={report['min_of_max']}, highest max={report['max']}, "
            f"suggested threshold={report['suggested_threshold']}"
        )
        return 0

    print(format_status(training_status()))
    return 0


# ---------------------------------------------------------------------------
# Self-test (no microphone required)
# ---------------------------------------------------------------------------


def _self_test() -> int:
    import tempfile
    import shutil

    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        mark = "ok  " if condition else "FAIL"
        print(f"  [{mark}] {label}{f' — {detail}' if detail else ''}")
        if not condition:
            failures.append(label)

    work = Path(tempfile.mkdtemp(prefix="atlas-wake-samples-"))
    try:
        target = ensure_model_dir(work / "wake_word")
        check("the model dir is created", target.is_dir(), str(target))

        # A fake detector so no real model is needed.
        class FakeModel:
            def predict_clip(self, path: str) -> list[dict[str, float]]:
                return [{"hey": 0.1}, {"hey": 0.7}, {"hey": 0.4}]

        class FakeDetector:
            model = FakeModel()
            model_key = "hey"
            model_source = "fake"
            using_custom_model = True

        import soundfile as sf

        import numpy as np

        paths = []
        for index, _ in enumerate(range(3)):
            path = target / f"positive_{index}.wav"
            sf.write(str(path), np.zeros(16000, dtype="int16"), SAMPLE_RATE, subtype=WAV_SUBTYPE)
            paths.append(path)

        listed = list_samples(target, "positive_*.wav")
        check("recorded clips are discovered", len(listed) == 3, str(len(listed)))

        report = score_samples(paths, detector=FakeDetector())
        check("scoring finds every clip", report["count"] == 3 and report["scored"] == 3)
        check("scoring reports the peak frame", report["max"] == 0.7, str(report["max"]))
        check("a threshold is suggested", isinstance(report["suggested_threshold"], float),
              str(report["suggested_threshold"]))
        check("the model provenance is reported", report["custom"] is True)

        single = score_wav(FakeDetector(), paths[0])
        check("score_wav returns max/mean/frames",
              single["max"] == 0.7 and single["frames"] == 3, str(single))

        status = training_status(target)
        check("status reports the model dir", status["model_dir_exists"])
        check("status lists positive clips", len(status["positive_clips"]) == 3)
        check("status flags no custom model yet", status["custom_model_present"] is False)
        check("the recommended path is the local trainer",
              len(verifier_commands()) == 2
              and any("wake_word_train" in c for c in verifier_commands()))
        check("the heavy pipeline commands are still produced",
              all("--training_config" in c for c in training_commands())
              and len(training_commands()) == len(TRAINING_STAGES))
        rendered = format_status(status)
        check("the report explains what is missing", "stand-in" in rendered)
        check("the report recommends the local trainer",
              "wake_word_train" in rendered)

        # A trained verifier must be reported as installed.
        (target / f"{WAKE_WORD_MODEL_NAME}.joblib").write_bytes(b"junk")
        verifier_status = training_status(target)
        check("an installed verifier is detected", verifier_status["verifier_present"] is True)
        check("the report says the verifier is in use",
              "verifier" in format_status(verifier_status).lower())

        # --- mic-check (fake device, no hardware) ------------------------
        global _import_sounddevice
        real_importer = _import_sounddevice

        class FakeSD:
            def __init__(self) -> None:
                self.played = 0

            def rec(self, frames: int, **kwargs: Any) -> Any:
                rng = np.random.default_rng(0)
                return (rng.normal(0, 0.2, (frames, 1)) * 32767).astype("int16")

            def wait(self) -> None:
                pass

            def play(self, data: Any, rate: int, **kwargs: Any) -> None:
                self.played += 1

        fake = FakeSD()
        _import_sounddevice = lambda: fake  # type: ignore[assignment]
        try:
            mic = mic_check(1.0, input_fn=lambda prompt: "y")
        finally:
            _import_sounddevice = real_importer  # type: ignore[assignment]
        check("mic-check records and plays back once", fake.played == 1 and mic["rounds"] == 1)
        check("mic-check reports a level verdict", mic["verdict"] == "good", str(mic))
        check("mic-check levels are sane dBFS", -40 < mic["peak_dbfs"] <= 0,
              f"{mic['peak_dbfs']} dBFS")

        # Simulate a trained model appearing.
        (target / f"{WAKE_WORD_MODEL_NAME}.onnx").write_bytes(b"not-a-real-model")
        check("a present custom model is detected",
              training_status(target)["custom_model_present"] is True)
    except Exception as exc:  # pragma: no cover - the test reports its own failure
        import traceback

        traceback.print_exc()
        failures.append(f"crashed: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for label in failures:
            print(f"  - {label}")
        return 1
    print("\nall checks passed.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        raise SystemExit(_self_test())
    raise SystemExit(main(sys.argv[1:]))
