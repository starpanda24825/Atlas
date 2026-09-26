"""
Atlas — train a personal "Hey Atlas" wake word from your own recordings.

Why this file exists
--------------------
openWakeWord ships no ``hey_atlas`` model, and its own trainer
(``python -m openwakeword.train``) is a heavyweight three-stage pipeline: it
*synthesises* positives with Piper, needs background-noise and room-impulse
corpora plus ``torch``/``onnx``, and it does not consume your own recordings
directly. That is not what we want here.

What we want is: *the user says "Hey Atlas" a dozen times, and Atlas hears it.*
openWakeWord supports exactly that through its **custom verifier** hook. A
verifier is a small scikit-learn logistic-regression classifier trained on
openWakeWord's own audio features. At runtime the library runs it on every frame
and **replaces the base model's score with the verifier's probability**
(``Model(..., custom_verifier_models={...}, custom_verifier_threshold=0.0)``).
So a handful of your clips is enough to turn a stand-in model into a detector
for your phrase, in your voice — and nothing leaves this machine.

How the pieces fit together:

* this module records your clips, extracts features and fits the verifier,
* it writes ``models/wake_word/hey_atlas.joblib`` (the verifier) and
  ``models/wake_word/hey_atlas.json`` (the recommended threshold),
* :class:`voice.wake_word.WakeWordDaemon` picks both up automatically on the
  next start — there is nothing to edit and no daemon to rebuild.

The ``.onnx`` route (full custom network) is still possible later; the daemon
prefers a real ``hey_atlas.onnx`` if you ever produce one.

Usage::

    # 1. check the microphone (records and plays your voice back)
    python -m voice.wake_word_samples mic-check

    # 2. record samples and train + install the verifier
    python -m voice.wake_word_train --record

    # or, if the positives/negatives are already recorded:
    python -m voice.wake_word_train
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from core.config import WAKE_WORD_MODEL_DIR, WAKE_WORD_MODEL_NAME

logger = logging.getLogger(__name__)

# --- Audio geometry (must match the daemon) ---------------------------------
FRAME_SAMPLES = 1280
SAMPLE_RATE = 16000

# openWakeWord's feature buffer is pre-seeded with random embeddings, so early
# windows mix real audio with noise. The feature window is 16 frames long, so
# once 16 real frames have been appended the last window is entirely real —
# hence the warmup, sized to the window rather than guessed.
FEATURE_WINDOW_FRAMES = 16
FEATURE_WARMUP_FRAMES = FEATURE_WINDOW_FRAMES

# Fraction of the quietest windows to drop, so silence is not taught as a
# positive example. 0.15 keeps ~85% of a clean clip.
DEFAULT_GATE_RATIO = 0.15

# How many extra augmented copies of each positive clip to add.
DEFAULT_AUGMENT = 3

POSITIVE_GLOB = "positive_*.wav"
NEGATIVE_GLOB = "negative_*.wav"


# ---------------------------------------------------------------------------
# Dependencies / helpers
# ---------------------------------------------------------------------------


def _import_openwakeword() -> Any:
    try:
        import openwakeword
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The 'openwakeword' package is required — install it with "
            "`.venv/bin/pip install openwakeword`."
        ) from exc
    return openwakeword


def resolve_base_model_name() -> str:
    """The base model the verifier will ride on top of.

    Must match what :class:`voice.wake_word.WakeWordDaemon` resolves when no
    custom ``.onnx`` exists, otherwise the verifier key would not match.
    """
    from voice.wake_word import resolve_base_model_name as _resolve

    return _resolve()


def find_clips(directory: str | Path | None = None, glob: str = POSITIVE_GLOB) -> list[Path]:
    target = Path(directory) if directory else WAKE_WORD_MODEL_DIR
    if not target.is_dir():
        return []
    return sorted(target.glob(glob))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def load_wav(path: str | Path) -> "Any":
    """Read a WAV as mono int16 at its native sample rate (16 kHz expected)."""
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="int16", always_2d=True)
    if rate != SAMPLE_RATE:
        logger.warning("%s is %d Hz, expected %d Hz — resampling", path, rate, SAMPLE_RATE)
        # Linear resample is good enough for feature harvesting.
        target_len = int(round(len(data) * SAMPLE_RATE / rate))
        idx = np.linspace(0, len(data) - 1, target_len)
        data = np.stack(
            [np.interp(idx, np.arange(len(data)), data[:, ch]) for ch in range(data.shape[1])],
            axis=1,
        )
    mono = data.astype("float32").mean(axis=1)
    return np.clip(mono, -32768, 32767).astype("int16")


def _perturb(clip: "Any", rng: "Any", *, gain_range: tuple[float, float] = (0.7, 1.3)) -> "Any":
    """A cheap augmentation: random gain + a sub-frame time shift."""
    import numpy as np

    gain = float(rng.uniform(*gain_range))
    shifted = clip[int(rng.integers(0, FRAME_SAMPLES)):]
    scaled = shifted.astype("float32") * gain
    return np.clip(scaled, -32768, 32767).astype("int16")


def clip_features(
    oww_model: Any,
    model_name: str,
    clip: "Any",
    *,
    gate_ratio: float = DEFAULT_GATE_RATIO,
) -> list["Any"]:
    """Harvest ``(1, 16, 96)`` feature windows from one clip.

    Mirrors what openWakeWord calls ``get_features(model_inputs[name])`` after
    every ``predict``; the daemon's verifier sees exactly these windows.
    """
    import numpy as np

    pre = oww_model.preprocessor
    try:
        pre.reset()
    except Exception:  # pragma: no cover - depends on version
        logger.debug("preprocessor.reset() unavailable", exc_info=True)

    windows: list[Any] = []
    energies: list[float] = []
    for index in range(0, len(clip) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        frame = clip[index:index + FRAME_SAMPLES]
        oww_model.predict(frame)  # advances the mel/embedding buffers
        if index // FRAME_SAMPLES < FEATURE_WARMUP_FRAMES:
            continue
        features = pre.get_features(oww_model.model_inputs[model_name])
        windows.append(np.asarray(features)[0])
        energies.append(float(np.mean(np.abs(frame.astype("float32")))))

    if len(windows) <= 1 or gate_ratio <= 0:
        return windows

    cutoff = float(np.quantile(energies, gate_ratio))
    return [w for w, energy in zip(windows, energies) if energy > 0 and energy >= cutoff]


def extract_dataset(
    positives: Sequence[str | Path],
    negatives: Sequence[str | Path],
    *,
    base_model: str | None = None,
    augment: int = DEFAULT_AUGMENT,
    gate_ratio: float = DEFAULT_GATE_RATIO,
    seed: int = 0,
) -> dict[str, Any]:
    """Build the ``(X, y, groups)`` feature matrix from clip paths.

    ``groups`` holds the source clip index for each window, so the train/test
    split can keep windows from the same recording together.
    """
    import numpy as np

    oww = _import_openwakeword()
    Model = oww.Model
    model_name = base_model or resolve_base_model_name()

    detector = Model(wakeword_models=[model_name], inference_framework="onnx")
    key = list(detector.models.keys())[0]

    rng = np.random.default_rng(seed)
    features: list[Any] = []
    labels: list[int] = []
    groups: list[int] = []

    for index, path in enumerate(positives):
        clip = load_wav(path)
        windows = clip_features(detector, key, clip, gate_ratio=gate_ratio)
        if not windows:
            logger.warning("no usable frames in %s (too short or silent)", path)
            continue
        for window in windows:
            features.append(window)
            labels.append(1)
            groups.append(index)
        for _ in range(max(0, augment)):
            variant = _perturb(clip, rng)
            for window in clip_features(detector, key, variant, gate_ratio=gate_ratio):
                features.append(window)
                labels.append(1)
                groups.append(index)

    offset = len(positives)
    for index, path in enumerate(negatives):
        clip = load_wav(path)
        for window in clip_features(detector, key, clip, gate_ratio=gate_ratio):
            features.append(window)
            labels.append(0)
            groups.append(offset + index)

    # ``stack`` (not ``vstack``): each window is (16, 96) and must stay a single
    # row, so the matrix is (n_windows, 16, 96) — exactly what the verifier's
    # ``flatten_features`` step expects at runtime.
    return {
        "X": (
            np.stack(features, axis=0)
            if features
            else np.empty((0, FEATURE_WINDOW_FRAMES, 96), dtype="float32")
        ),
        "y": np.asarray(labels, dtype=int),
        "groups": np.asarray(groups, dtype=int),
        "model_name": key,
    }


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def build_pipeline() -> Any:
    """The same estimator structure openWakeWord's own verifier uses.

    Reusing ``flatten_features`` from the library is deliberate: the pickled
    object is unpickled by openWakeWord at runtime, so it must reference a
    function that exists there.
    """
    from openwakeword.custom_verifier_model import flatten_features
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    return make_pipeline(
        FunctionTransformer(flatten_features),
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced", random_state=0),
    )


def _split_indices(groups: "Any", *, test_fraction: float = 0.25, seed: int = 0) -> tuple["Any", "Any"]:
    import numpy as np

    unique = np.unique(groups)
    if len(unique) < 4:
        # Too few clips to hold anything out; evaluate on the training data.
        idx = np.arange(len(groups))
        return idx, idx
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_test = max(1, int(round(len(unique) * test_fraction)))
    test_groups = set(shuffled[:n_test].tolist())
    test = np.array([i for i, g in enumerate(groups) if g in test_groups])
    train = np.array([i for i, g in enumerate(groups) if g not in test_groups])
    return train, test


def choose_threshold(probs: "Any", labels: "Any", *, min_recall: float = 0.7) -> dict[str, Any]:
    """Pick the detection threshold that best separates the two classes.

    Preference: the *highest* threshold that still keeps recall at
    ``min_recall`` (fewer false wakes), falling back to the best
    recall-minus-false-positive-rate trade-off when recall is poor.
    """
    import numpy as np

    positives = probs[labels == 1]
    negatives = probs[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return {"threshold": 0.5, "recall": None, "fpr": None}

    candidates = np.round(np.linspace(0.05, 0.95, 91), 4)
    best_by_margin: dict[str, Any] | None = None

    ok: list[tuple[float, float]] = []  # (threshold, fpr)
    for t in candidates:
        recall = float((positives >= t).mean())
        fpr = float((negatives >= t).mean())
        margin = recall - fpr
        if best_by_margin is None or margin > best_by_margin["margin"]:
            best_by_margin = {"threshold": float(t), "recall": recall, "fpr": fpr, "margin": margin}
        if recall >= min_recall:
            ok.append((float(t), fpr))

    if ok:
        threshold, fpr = max(ok, key=lambda item: item[0])
        return {
            "threshold": round(threshold, 4),
            "recall": round(float((positives >= threshold).mean()), 4),
            "fpr": round(fpr, 4),
            "margin": round(float((positives >= threshold).mean() - fpr), 4),
        }
    assert best_by_margin is not None
    return {
        "threshold": round(best_by_margin["threshold"], 4),
        "recall": round(best_by_margin["recall"], 4),
        "fpr": round(best_by_margin["fpr"], 4),
        "margin": round(best_by_margin["margin"], 4),
    }


def train(
    positives: Sequence[str | Path],
    negatives: Sequence[str | Path],
    *,
    base_model: str | None = None,
    augment: int = DEFAULT_AUGMENT,
    gate_ratio: float = DEFAULT_GATE_RATIO,
    output_dir: str | Path | None = None,
    name: str = WAKE_WORD_MODEL_NAME,
    save: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Train a verifier from recorded clips and (optionally) install it.

    Returns a report dict including the fitted pipeline, the resolved
    ``threshold`` and the evaluation metrics.
    """
    import numpy as np

    target = Path(output_dir) if output_dir else WAKE_WORD_MODEL_DIR
    if not positives:
        raise ValueError("no positive clips found")
    if not negatives:
        raise ValueError(
            "no negative clips found — the verifier needs examples of ordinary "
            "speech to reject (record some with "
            "`python -m voice.wake_word_samples record --kind negative`)"
        )

    data = extract_dataset(
        positives, negatives, base_model=base_model, augment=augment,
        gate_ratio=gate_ratio, seed=seed,
    )
    X, y, groups = data["X"], data["y"], data["groups"]
    if X.shape[0] == 0:
        raise ValueError("feature extraction produced no windows — are the clips too short?")

    train_idx, test_idx = _split_indices(groups, seed=seed)
    pipeline = build_pipeline()
    pipeline.fit(X[train_idx], y[train_idx])

    train_probs = pipeline.predict_proba(X[train_idx])[:, 1]
    threshold_info = choose_threshold(train_probs, y[train_idx])

    test_probs = pipeline.predict_proba(X[test_idx])[:, 1]
    test_pos = test_probs[y[test_idx] == 1]
    test_neg = test_probs[y[test_idx] == 0]
    threshold = float(threshold_info["threshold"])
    metrics = {
        "train_windows": int(len(train_idx)),
        "test_windows": int(len(test_idx)),
        "positive_windows": int((y == 1).sum()),
        "negative_windows": int((y == 0).sum()),
        "threshold": threshold,
        "train_recall": threshold_info["recall"],
        "train_fpr": threshold_info["fpr"],
        "test_recall": round(float((test_pos >= threshold).mean()), 4) if test_pos.size else None,
        "test_fpr": round(float((test_neg >= threshold).mean()), 4) if test_neg.size else None,
        "mean_positive_prob": round(float(test_pos.mean()), 4) if test_pos.size else None,
        "mean_negative_prob": round(float(test_neg.mean()), 4) if test_neg.size else None,
    }

    report: dict[str, Any] = {
        "pipeline": pipeline,
        "threshold": threshold,
        "metrics": metrics,
        "base_model": data["model_name"],
        "saved_verifier": None,
        "saved_sidecar": None,
    }

    if save:
        target.mkdir(parents=True, exist_ok=True)
        verifier_path = target / f"{name}.joblib"
        sidecar_path = target / f"{name}.json"
        with verifier_path.open("wb") as handle:
            pickle.dump(pipeline, handle)
        sidecar_path.write_text(
            json.dumps(
                {
                    "threshold": threshold,
                    "base_model": data["model_name"],
                    "created": int(time.time()),
                    "positive_clips": [Path(p).name for p in positives],
                    "negative_clips": [Path(p).name for p in negatives],
                    "metrics": metrics,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        report["saved_verifier"] = str(verifier_path)
        report["saved_sidecar"] = str(sidecar_path)

    return report


# ---------------------------------------------------------------------------
# Interactive end-to-end run
# ---------------------------------------------------------------------------


def record_training_data(
    *,
    directory: str | Path | None = None,
    positives: int = 12,
    negatives: int = 10,
    positive_seconds: float = 3.0,
    negative_seconds: float = 5.0,
    input_fn: Any = input,
) -> dict[str, list[Path]]:
    """Record positives (the wake word) and negatives (any other speech)."""
    from voice.wake_word_samples import record_samples

    phrase = WAKE_WORD_MODEL_NAME.replace("_", " ")
    print(
        "\n=== Positive clips ===\n"
        f"You will record {positives} takes. Say \"{phrase}\" at a natural pace\n"
        "each time — vary your distance and volume slightly.\n"
    )
    pos = record_samples(
        positives, seconds=positive_seconds, directory=directory,
        prefix="positive", prompt=f'say "{phrase}"', input_fn=input_fn,
    )

    print(
        "\n=== Negative clips ===\n"
        f"Now {negatives} takes of ordinary speech that is NOT the wake word.\n"
        "Read these out (or anything similar) — about a sentence each:\n"
        '  "okay, what is the weather looking like tomorrow morning"\n'
        '  "remind me to call my sister when I get home this evening"\n'
        '  "can you summarise the last email I got from the bank"\n'
        '  "let us go over the schedule for next week real quick"\n'
        '  "I think we should stop by the shop on the way back"\n'
        '  "hey, could you turn the music down a little please"\n'
        '  "add milk and eggs to the shopping list for me"\n'
        '  "what time does the film start on saturday afternoon"\n'
        '  "start a new note about the project deadline changes"\n'
        '  "how many calories were in that lunch I logged"\n'
    )
    neg = record_samples(
        negatives, seconds=negative_seconds, directory=directory,
        prefix="negative", prompt="say a full sentence that is NOT the wake word",
        input_fn=input_fn,
    )
    return {"positives": pos, "negatives": neg}


def format_report(report: dict[str, Any]) -> str:
    m = report["metrics"]
    lines = [
        "",
        "================= training result =================",
        f"base model         : {report['base_model']}",
        f"windows (pos/neg)  : {m['positive_windows']} / {m['negative_windows']}",
        f"held-out windows   : {m['test_windows']}",
        f"threshold chosen   : {m['threshold']:.2f}",
        f"recall (your voice): {_pct(m['test_recall'])}  "
        f"[train {_pct(m['train_recall'])}]",
        f"false wakes        : {_pct(m['test_fpr'])}  [train {_pct(m['train_fpr'])}]",
        f"mean prob (pos/neg): {_num(m['mean_positive_prob'])} / {_num(m['mean_negative_prob'])}",
        "---------------------------------------------------",
    ]
    if report["saved_verifier"]:
        lines.append(f"verifier written   : {report['saved_verifier']}")
        lines.append(f"threshold sidecar  : {report['saved_sidecar']}")
    lines.append("===================================================")
    return "\n".join(lines)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.0f}%"


def _num(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def run(
    *,
    directory: str | Path | None = None,
    record: bool = False,
    positives: int = 12,
    negatives: int = 10,
    augment: int = DEFAULT_AUGMENT,
    name: str = WAKE_WORD_MODEL_NAME,
    input_fn: Any = input,
) -> int:
    """The whole flow: record (optionally), train, install, print next steps."""
    target = Path(directory) if directory else WAKE_WORD_MODEL_DIR
    positives_paths = find_clips(target, POSITIVE_GLOB)
    negatives_paths = find_clips(target, NEGATIVE_GLOB)

    if record or not positives_paths or not negatives_paths:
        if not record:
            print(
                f"Found {len(positives_paths)} positive and {len(negatives_paths)} "
                f"negative clip(s) in {target}."
            )
        recorded = record_training_data(
            directory=target, positives=positives, negatives=negatives, input_fn=input_fn
        )
        positives_paths = sorted(recorded["positives"])
        negatives_paths = sorted(recorded["negatives"])

    print(
        f"\nTraining on {len(positives_paths)} positive and "
        f"{len(negatives_paths)} negative clip(s)..."
    )
    try:
        report = train(
            positives_paths, negatives_paths, augment=augment, output_dir=target, name=name
        )
    except ValueError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print(format_report(report))
    print(
        "\nDone. The wake word daemon will pick this up automatically on its next\n"
        "start — no config change, no rebuild. Test it with:\n\n"
        "    .venv/bin/python -m voice.wake_word\n\n"
        "Say \"Hey Atlas\" a few times. If it fires too often, raise the threshold\n"
        f"in {target / (name + '.json')} (or set WAKE_WORD_THRESHOLD in core/config.py).\n"
        "If it misses you, lower it.\n"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voice.wake_word_train",
        description='Train a personal "Hey Atlas" wake word from your own recordings.',
    )
    parser.add_argument("--record", action="store_true",
                        help="record samples first (recommended on the first run)")
    parser.add_argument("--dir", default=None, help="where clips live (default: the model dir)")
    parser.add_argument("--positives", type=int, default=12, help="how many wake-word takes")
    parser.add_argument("--negatives", type=int, default=10, help="how many ordinary-speech takes")
    parser.add_argument("--augment", type=int, default=DEFAULT_AUGMENT,
                        help="extra augmented copies of each positive clip")
    parser.add_argument("--name", default=WAKE_WORD_MODEL_NAME, help="wake word name to write")
    parser.add_argument("--self-test", action="store_true", help="run offline checks and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    if args.self_test:
        return _self_test()
    try:
        return run(
            directory=args.dir,
            record=args.record,
            positives=args.positives,
            negatives=args.negatives,
            augment=args.augment,
            name=args.name,
        )
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Self-test (no microphone, no model weights required)
# ---------------------------------------------------------------------------


def _self_test() -> int:
    import tempfile
    import shutil

    import numpy as np

    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        mark = "ok  " if condition else "FAIL"
        print(f"  [{mark}] {label}{f' — {detail}' if detail else ''}")
        if not condition:
            failures.append(label)

    class FakePreprocessor:
        def reset(self) -> None:
            pass

        def get_features(self, n: int) -> "Any":
            return self._frame[None, :, :].astype("float32")

    class FakeModel:
        def __init__(self) -> None:
            self.preprocessor = FakePreprocessor()
            self.model_inputs = {"hey": 16}
            self.models = {"hey": object()}
            self._step = 0

        def predict(self, frame: "Any") -> dict[str, float]:
            self._step += 1
            rng = np.random.default_rng(self._step)
            self.preprocessor._frame = rng.random((16, 96))
            return {"hey": 0.01}

    work = Path(tempfile.mkdtemp(prefix="atlas-wake-train-"))
    try:
        # --- geometry -----------------------------------------------------
        audio = (np.sin(np.arange(4 * SAMPLE_RATE, dtype="float32") / 40) * 8000).astype("int16")
        check("clips are long enough to train on",
              len(audio) // FRAME_SAMPLES > FEATURE_WARMUP_FRAMES)

        # --- feature harvesting ------------------------------------------
        windows = clip_features(FakeModel(), "hey", audio, gate_ratio=0.0)
        expected = (len(audio) // FRAME_SAMPLES) - FEATURE_WARMUP_FRAMES
        check("windows skip the warmup frames", len(windows) == expected,
              f"{len(windows)} vs {expected}")
        check("each window is (16, 96)", windows and windows[0].shape == (16, 96))

        gated = clip_features(FakeModel(), "hey", audio, gate_ratio=0.5)
        check("gating drops quiet windows", 0 < len(gated) <= len(windows),
              f"{len(gated)} of {len(windows)}")

        # --- dataset & split ---------------------------------------------
        import soundfile as sf

        n_pos, n_neg = 6, 4
        for i in range(n_pos):
            sf.write(str(work / f"positive_{i}.wav"), audio, SAMPLE_RATE, subtype="PCM_16")
        for i in range(n_neg):
            sf.write(str(work / f"negative_{i}.wav"), audio, SAMPLE_RATE, subtype="PCM_16")
        check("clips are found by glob",
              len(find_clips(work, POSITIVE_GLOB)) == n_pos
              and len(find_clips(work, NEGATIVE_GLOB)) == n_neg)

        # --- threshold logic (pure numpy, no model) ----------------------
        probs = np.concatenate([np.linspace(0.8, 0.99, 50), np.linspace(0.0, 0.25, 50)])
        labels = np.array([1] * 50 + [0] * 50)
        info = choose_threshold(probs, labels)
        check("a threshold is chosen", 0.05 <= info["threshold"] <= 0.95, str(info["threshold"]))
        check("chosen threshold keeps recall and rejects negatives",
              info["recall"] >= 0.7 and info["fpr"] == 0.0, str(info))
        check("overlapping classes still yield a threshold",
              "threshold" in choose_threshold(np.array([0.5, 0.5]), np.array([1, 0])))

        # --- pipeline round-trips through pickle -------------------------
        # Windows must be stacked, not concatenated: the matrix has to stay
        # (n_windows, 16, 96) or the verifier trains on the wrong feature count.
        X = np.stack(windows, axis=0)
        check("the feature matrix keeps (n, 16, 96)",
              X.ndim == 3 and X.shape[1:] == (16, 96), str(X.shape))

        y = np.array([1, 0] * (len(X) // 2) + [1] * (len(X) % 2))
        pipeline = build_pipeline()
        pipeline.fit(X, y)
        check("windows flatten to 16x96 features, what the runtime feeds in",
              pipeline.named_steps["standardscaler"].n_features_in_ == 16 * 96,
              str(pipeline.named_steps["standardscaler"].n_features_in_))

        blob = pickle.dumps(pipeline)
        restored = pickle.loads(blob)
        before = pipeline.predict_proba(X)[:, 1]
        after = restored.predict_proba(X)[:, 1]
        check("the verifier survives pickling (what the daemon loads)",
              np.allclose(before, after), f"{before[:2]} vs {after[:2]}")
        check("predictions are probabilities", bool(np.all((before >= 0) & (before <= 1))))

        # The runtime calls predict_proba on a single (1, 16, 96) window.
        single = restored.predict_proba(np.asarray(windows[0])[None, :, :])[0][-1]
        check("a single window scores like the batch",
              abs(float(single) - float(before[0])) < 1e-9, f"{single} vs {before[0]}")
    except Exception as exc:  # pragma: no cover - reported as a failure
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
    raise SystemExit(main())
