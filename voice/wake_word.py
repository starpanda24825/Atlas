"""
Atlas — wake word daemon.

The always-on ear of Atlas. A single, tiny process whose only job is to answer
one question: *did the user just say "Hey Atlas"?* When the answer is yes it
fires a callback and goes straight back to listening. It never transcribes
speech, never calls an LLM and never touches the GPU — it exists so the
expensive parts of Atlas can stay asleep until they are actually needed.

Why this stays cheap:

* openWakeWord runs on the **ONNX** runtime, which is pinned to
  ``CPUExecutionProvider``. Importing this module pulls in no CUDA, Torch,
  TensorFlow or JAX code at all (verified), so it cannot disturb a game or
  compete with the llama.cpp servers that do own the GPU.
* Audio arrives as 1280-sample (80 ms) frames at 16 kHz — one small model
  invocation per frame. ``stream.read()`` blocks for the frame's duration, so
  the loop is paced by the microphone rather than spinning, keeping
  steady-state CPU well under 2%.
* The two heavyweight dependencies (``openwakeword``, ``sounddevice``) are
  imported lazily, so importing this module is nearly free and a missing
  PortAudio library produces one actionable error instead of an ImportError
  at module load.

Deviation from the brief — ``inference_framework``
--------------------------------------------------
The brief asked for ``openwakeword.Model(..., inference_framework="tflite")``.
That does not work in this environment: the installed ``tflite-runtime`` 2.14
wheel is ABI-incompatible with NumPy 2.x, and building an interpreter raises

    SystemError: <built-in method CreateWrapperFromFile of PyCapsule object
    ...> returned a result with an exception set

ONNX is the supported alternative, is already installed, and is equally
CPU-only, so :data:`INFERENCE_FRAMEWORK` is ``"onnx"``. Set it back to
``"tflite"`` only if tflite-runtime is replaced by a NumPy 2 compatible build
(e.g. ``ai-edge-litert``).

Usage::

    from voice.wake_word import WakeWordDaemon

    daemon = WakeWordDaemon()
    try:
        daemon.listen_forever(on_wake)      # blocks until stop()
    finally:
        daemon.stop()

Run it standalone (logs every detection, drives no LLM)::

    python3 -m voice.wake_word
"""

# ---------------------------------------------------------------------------
# Training a custom "Hey Atlas" model
# ---------------------------------------------------------------------------
# openWakeWord ships no "hey atlas" model, so by default this daemon falls back
# to a pre-built stand-in and warns about it. To train the real thing:
#
# The trainer is a three-stage CLI. Note what it actually does: it
# *synthesises* its own positive samples with Piper TTS and needs background
# noise and room-impulse-response corpora. It does not consume your own
# recordings directly — the official notebook workflow is where real recordings
# are folded in alongside the synthetic ones.
#
#   1. Install the training extras. These are deliberately NOT in
#      requirements.txt: they pull multi-GB ML dependencies that the listening
#      daemon must never import.
#
#          .venv/bin/python -m pip install torchinfo torchmetrics
#
#   2. Clone Piper (for synthetic positives) and openWakeWord (for the
#      reference training config and notebooks).
#
#          git clone https://github.com/rhasspy/piper.git
#          git clone https://github.com/dscripka/openWakeWord.git
#
#   3. Write a training config YAML, modelled on
#      openWakeWord/notebooks/train_custom_model.ipynb. The keys the CLI reads
#      are: model_name, model_type, target_phrase, output_dir, n_samples,
#      n_samples_val, tts_batch_size, augmentation_rounds,
#      augmentation_batch_size, total_length, batch_n_per_class, steps,
#      max_negative_weight, target_false_positives_per_hour, layer_size,
#      feature_data_files, custom_negative_phrases, rir_paths, background_paths,
#      background_paths_duplication_rate, false_positive_validation_data_path
#      and piper_sample_generator_path. Minimum viable version:
#
#          model_name: hey_atlas
#          model_type: dnn
#          target_phrase: ["hey atlas"]
#          output_dir: /home/starpanda/Documents/Projects/AI/Atlas/models
#          n_samples: 5000
#          piper_sample_generator_path: /path/to/piper/src/python
#          rir_paths: [/path/to/mit_ir_survey]
#          background_paths: [/path/to/background_noise]
#          # ...plus the remaining keys from the notebook
#
#   4. Run the three stages in order (each reads the previous stage's output):
#
#          .venv/bin/python -m openwakeword.train --training_config train_hey_atlas.yaml --generate_clips
#          .venv/bin/python -m openwakeword.train --training_config train_hey_atlas.yaml --augment_clips
#          .venv/bin/python -m openwakeword.train --training_config train_hey_atlas.yaml --train_model
#
#   5. Copy the resulting hey_atlas.onnx into models/wake_word/ (see
#      WAKE_WORD_MODEL_DIR in core/config.py). This daemon picks it up
#      automatically on the next start and stops warning.
#
# Tip for a quick sanity check before committing to a full training run: record
# a handful of 16 kHz mono 16-bit WAVs of yourself saying "Hey Atlas" and score
# them with Model.predict_clip() — it runs the same streaming pipeline on a file.

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from core.config import (
    CHANNELS,
    SAMPLE_RATE,
    WAKE_WORD_MODEL_DIR,
    WAKE_WORD_THRESHOLD,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import numpy as np

logger = logging.getLogger(__name__)

# --- Audio format -----------------------------------------------------------
# openWakeWord consumes fixed 80 ms frames: 1280 samples at 16 kHz, and the
# input length must be a multiple of that. Longer chunks cut the per-sample
# overhead at the cost of detection latency. Measured here on the ONNX runtime
# (model inference only, one core, 200 frames averaged):
#
#     1280 samples ( 80 ms audio) -> 1.67 ms/frame -> ~2.1% of one core
#     2560 samples (160 ms audio) -> 2.91 ms/frame -> ~1.8% of one core
#     3840 samples (240 ms audio) -> 4.16 ms/frame -> ~1.7% of one core
#
# 1280 stays the default: it is the lowest-latency option and its cost is a
# couple of percent of a *single* core, i.e. far under 2% of total CPU on a
# multi-core desktop. Pass a larger frame_samples if that trade looks wrong.
FRAME_MS = 80
FRAME_SAMPLES = 1280
# openWakeWord expects raw 16-bit PCM, not floats.
FRAME_DTYPE = "int16"

# --- Detection --------------------------------------------------------------
# The model we actually want, and the pre-built models used as stand-ins until
# it has been trained. Order matters: the first one that exists wins.
WAKE_WORD_MODEL_NAME = "hey_atlas"
FALLBACK_MODEL_NAMES = ("alexa", "hey_mycroft")

# See the module docstring: tflite-runtime is broken against NumPy 2.x here.
INFERENCE_FRAMEWORK = "onnx"

# After a detection, ignore everything for this long so the tail of the phrase
# (and any immediate repeat) cannot fire a second callback.
COOLDOWN_SECONDS = 2.0

# How long to wait before reopening the microphone after a device error.
RECONNECT_DELAY_SECONDS = 5.0


class WakeWordDaemon:
    """Listens for a wake word and calls back when it hears it.

    Loads the OpenWakeWord model in :meth:`__init__`, then blocks in
    :meth:`listen_forever`. Audio device failures are retried forever rather
    than raised: this process is expected to outlive HDMI switches, sleep/wake
    cycles and USB microphones being unplugged.

    All tunables default to :mod:`core.config`, but each is also a constructor
    argument so it can be overridden (and tested) without editing config.
    """

    def __init__(
        self,
        wake_word: str = WAKE_WORD_MODEL_NAME,
        threshold: float = WAKE_WORD_THRESHOLD,
        model_dir: Path = WAKE_WORD_MODEL_DIR,
        fallback_names: tuple[str, ...] = FALLBACK_MODEL_NAMES,
        frame_samples: int = FRAME_SAMPLES,
    ) -> None:
        self.wake_word: str = wake_word
        self.threshold: float = threshold
        self.model_dir: Path = model_dir
        self.fallback_names: tuple[str, ...] = fallback_names

        # Must be a multiple of 80 ms (1280 samples); see FRAME_SAMPLES.
        self.frame_samples: int = frame_samples

        self._stop_event = threading.Event()
        self._stream: Any | None = None

        # Populated by _load_model().
        self.model: Any | None = None
        self.model_key: str | None = None
        self.model_source: str | None = None
        self.using_custom_model: bool = False

        self._load_model()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        """True between :meth:`listen_forever` and :meth:`stop`."""
        return not self._stop_event.is_set()

    def __repr__(self) -> str:
        return (
            f"<WakeWordDaemon model={self.model_source!r} "
            f"custom={self.using_custom_model} threshold={self.threshold}>"
        )

    # ------------------------------------------------------------------
    # Lazy dependencies
    # ------------------------------------------------------------------

    @staticmethod
    def _import_model_class() -> Any:
        """Import openWakeWord's Model lazily and explain any failure."""
        try:
            from openwakeword.model import Model
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "The 'openwakeword' package is required for wake word "
                "detection — install it with `pip install openwakeword`."
            ) from exc
        return Model

    @staticmethod
    def _import_sounddevice() -> Any:
        """Import sounddevice lazily, naming the system package on failure.

        sounddevice raises ``OSError`` (not ImportError) when the PortAudio
        shared library is missing, so both are handled here.
        """
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "sounddevice could not load the PortAudio library. Install the "
                "system package, then retry: "
                "`sudo apt install -y portaudio19-dev libportaudio2` "
                f"(original error: {exc})"
            ) from exc
        return sd

    @staticmethod
    def _pretrained_model_paths(framework: str) -> list[str]:
        try:
            import openwakeword
        except ImportError:  # pragma: no cover - depends on environment
            return []
        try:
            return list(openwakeword.get_pretrained_model_paths(framework))
        except Exception:  # pragma: no cover - defensive
            logger.debug("could not enumerate pre-trained models", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _custom_model_path(self) -> Path | None:
        """Return the trained ``hey_atlas`` model in the config dir, if any.

        ``models/wake_word`` is created lazily, so a missing directory simply
        means "not trained yet" — not an error.
        """
        if not self.model_dir.is_dir():
            return None
        preferred = ".onnx" if INFERENCE_FRAMEWORK == "onnx" else ".tflite"
        for suffix in (preferred, ".onnx", ".tflite"):
            candidate = self.model_dir / f"{self.wake_word}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def _fallback_model_name(self) -> str:
        """First pre-built model available in this installation."""
        available = self._pretrained_model_paths(INFERENCE_FRAMEWORK)
        for name in self.fallback_names:
            if any(name in Path(path).stem for path in available):
                return name
        raise RuntimeError(
            f"No '{self.wake_word}' model in {self.model_dir} and none of the "
            f"pre-built stand-ins {self.fallback_names} are installed. Run "
            "`python -c \"import openwakeword; "
            "openwakeword.utils.download_models()\"` first."
        )

    def _load_model(self) -> None:
        """Load the custom model, or fall back to a pre-built one with a warning."""
        Model = self._import_model_class()

        source, is_custom = self._custom_model_path(), True
        if source is None:
            source, is_custom = self._fallback_model_name(), False
            logger.warning(
                "Custom '%s' model not found in %s — falling back to the "
                "pre-built '%s' model as a placeholder. Say the stand-in phrase "
                "instead until a custom model is trained (see the training notes "
                "in voice/wake_word.py).",
                self.wake_word,
                self.model_dir,
                source,
            )

        try:
            model = self._build_model(Model, str(source))
        except Exception:
            if not is_custom:
                raise
            # A half-trained or wrong-framework custom file must not take the
            # daemon down; degrade to the stand-in and say so loudly.
            logger.exception(
                "could not load custom model %s — falling back to a pre-built "
                "stand-in so wake word detection keeps working",
                source,
            )
            source, is_custom = self._fallback_model_name(), False
            model = self._build_model(Model, str(source))

        keys = list(model.models.keys())
        if not keys:
            raise RuntimeError("openWakeWord loaded no models")
        if len(keys) > 1:  # pragma: no cover - we always pass a single model
            logger.warning("expected one wake word model, got %s — using %s", keys, keys[0])

        self.model = model
        self.model_key = keys[0]
        self.model_source = str(source)
        self.using_custom_model = is_custom

        logger.info(
            "wake word model ready: %s (custom=%s, threshold=%.2f)",
            self.model_source,
            is_custom,
            self.threshold,
        )

    @staticmethod
    def _build_model(Model: Any, source: str) -> Any:
        # enable_speex_noise_suppression stays off: it is tflite-only and adds a
        # dependency the CUDA-free path does not need.
        return Model(
            wakeword_models=[source],
            inference_framework=INFERENCE_FRAMEWORK,
            enable_speex_noise_suppression=False,
        )

    # ------------------------------------------------------------------
    # Listening
    # ------------------------------------------------------------------

    def listen_forever(self, callback: Callable[[], None]) -> None:
        """Block, calling ``callback()`` each time the wake word is heard.

        The microphone loop is wrapped so that *any* audio failure — device
        unplugged, PortAudio internal error, overflow — is logged and retried
        after :data:`RECONNECT_DELAY_SECONDS` instead of propagating. This is a
        daemon: it must survive whatever the desktop does to the audio stack.

        Returns only once :meth:`stop` has been called.
        """
        self._stop_event.clear()
        logger.info(
            "listening for '%s' (model=%s, threshold=%.2f, %s frames)",
            self.wake_word,
            self.model_source,
            self.threshold,
            f"{FRAME_MS}ms",
        )

        while not self._stop_event.is_set():
            try:
                self._stream_until_stopped(callback)
            except Exception as exc:
                # Expected during sleep/wake, device switches and unplugs.
                logger.warning(
                    "wake word audio stream failed (%s: %s) — reopening in %.0fs",
                    type(exc).__name__,
                    exc,
                    RECONNECT_DELAY_SECONDS,
                )
                # wait() doubles as an interruptible sleep: a stop() during the
                # backoff should not have to wait out the full delay.
                if self._stop_event.wait(RECONNECT_DELAY_SECONDS):
                    break
            except KeyboardInterrupt:  # pragma: no cover - interactive use
                logger.info("interrupted — stopping wake word daemon")
                break

        logger.info("wake word daemon stopped")

    def _stream_until_stopped(self, callback: Callable[[], None]) -> None:
        """Open the microphone and process frames until ``stop`` is called."""
        sd = self._import_sounddevice()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=FRAME_DTYPE,
            blocksize=self.frame_samples,
        ) as stream:
            self._stream = stream
            try:
                while not self._stop_event.is_set():
                    chunk, overflowed = stream.read(self.frame_samples)
                    if overflowed:
                        logger.debug("input overflow — frames were dropped")

                    if self._score(chunk) < self.threshold:
                        continue

                    self._on_detection(callback)

                    # Cooldown: keep *reading* and discarding frames rather than
                    # sleeping. A paused InputStream overflows, which would cost
                    # us audio the moment listening resumes.
                    deadline = time.monotonic() + COOLDOWN_SECONDS
                    while not self._stop_event.is_set() and time.monotonic() < deadline:
                        stream.read(self.frame_samples)
            finally:
                self._stream = None

    def _score(self, chunk: "np.ndarray") -> float:
        """Wake word confidence for one frame, in the 0..1 range.

        Falls back to the highest score returned if the model's internal key
        does not match what was requested — cheap insurance against openWakeWord
        renaming keys between versions, and harmless because exactly one model
        is loaded.
        """
        assert self.model is not None, "wake word model not loaded"
        audio = chunk.reshape(-1)
        scores = self.model.predict(audio)
        try:
            return float(scores[self.model_key])
        except KeyError:  # pragma: no cover - defensive
            return float(max(scores.values()))

    def _on_detection(self, callback: Callable[[], None]) -> None:
        """Reset detector state, notify the caller, then cool down."""
        logger.info("wake word '%s' detected", self.wake_word)

        # Drop the buffered audio behind the phrase so its own tail cannot
        # immediately re-trigger the model.
        if self.model is not None:
            try:
                self.model.reset()
            except Exception:  # pragma: no cover - defensive
                logger.debug("could not reset wake word model state", exc_info=True)

        try:
            callback()
        except Exception:
            # A broken handler must not kill the ear — log and keep listening.
            logger.exception("wake word callback raised — continuing to listen")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Ask :meth:`listen_forever` to return. Safe from any thread."""
        self._stop_event.set()


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    daemon = WakeWordDaemon()
    print(f"model: {daemon.model_source} (custom={daemon.using_custom_model})")
    print("Say the wake word, or press Ctrl-C to quit.")

    def on_wake() -> None:
        print(">>> wake word detected")

    try:
        daemon.listen_forever(on_wake)
    except KeyboardInterrupt:
        pass
    finally:
        daemon.stop()
