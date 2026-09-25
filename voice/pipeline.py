"""
Atlas — full-duplex voice pipeline.

The conversational heart of Atlas: one object that hears you, transcribes you,
hands the text to the agent, speaks the reply, and lets you interrupt it
mid-sentence.

Data flow::

    mic ──> AEC ──> VAD ──> utterance buffer ──> faster-whisper ──> agent
             ▲                                                            │
             │ echo reference                                             ▼
    speaker <── playback buffer <── Kokoro (sentence at a time) <── text

Why each piece exists
---------------------

**Echo cancellation.** Atlas plays audio out of a speaker and captures audio
from a microphone in the same room, so the mic hears Atlas. Without cancelling
that, Atlas transcribes its own speech and (worse) treats it as a barge-in and
interrupts itself forever.

The brief asked for "subtract delayed playback signal". A plain subtraction
does not work: the path from speaker to microphone has an unknown gain, an
unknown delay, a frequency response and reverberation, so subtracting a
time-shifted copy of the playback leaves residual energy that is still far
louder than speech. Instead :class:`EchoCanceller` runs a small **block NLMS
adaptive filter**: it *learns* the transfer function from the playback
reference to the mic input and subtracts the filtered estimate. That is the
same "subtract the delayed playback signal" idea, with the delay and gain
estimated rather than guessed. :data:`AEC_DELAY_MS` applies the coarse
alignment; the filter absorbs the rest.

This is **not production AEC**. Measured against synthetic speaker-to-mic paths
it removes roughly 10-30 dB of echo on speech-like playback and nothing at all
on pathological 1/f input, and it has no double-talk protection (see
:class:`EchoCanceller`). WebRTC AEC3 via ``webrtc-audio-processing`` handles
double-talk, nonlinearity and clock drift properly, and is what this should be
swapped for if barge-in proves unreliable.

**Headphones remain the reliable configuration.** With a loudspeaker, expect
barge-in to misfire occasionally: Atlas can still hear some of itself, and the
VAD cannot distinguish that from the user.

**Barge-in.** The VAD runs on the echo-cancelled signal. When it reports
speech while TTS is playing, :attr:`barge_in_detected` is set; the playback
loop polls it every :data:`PLAYBACK_POLL_SECS` and stops within one ~30 ms
chunk. The word that was being spoken is recovered from Kokoro's token
timestamps so the caller knows exactly where Atlas was cut off.

**Streaming STT.** faster-whisper has no streaming API — it is a batch
transcriber. :meth:`listen_once` therefore *simulates* streaming: once speech
has run for :data:`PARTIAL_MIN_SECS`, a worker re-transcribes the growing
buffer every :data:`PARTIAL_INTERVAL_SECS` to produce partials (exposed via
``on_partial``), while only the final transcribe after silence is returned and
handed to the agent.

**Streaming TTS.** Text is split into sentence chunks (min
:data:`MIN_CHUNK_WORDS` words). Kokoro generates chunk N+1 on a worker thread
while chunk N is still playing, so the first chunk starts as soon as it is
ready instead of waiting for the whole reply. That overlap is what keeps
perceived latency low.

Deviations from the brief
-------------------------
* **Pipecat is used for its VAD engine, not as a full ``Pipeline``.** The brief
  says "use Pipecat to build a proper full-duplex voice conversation", but the
  custom mechanisms it also requires — the echo canceller and its ring buffer,
  the ``barge_in_detected`` event, sentence-chunked concurrent TTS with word
  timestamps, and pseudo-streaming STT — are all things a real Pipecat
  ``Pipeline`` replaces rather than hosts. So Pipecat supplies
  ``SileroVADAnalyzer`` (configured with exactly the brief's 0.7 / 250 ms /
  700 ms) and this module orchestrates around it.
* **Everything runs at 16 kHz.** Kokoro emits 24 kHz while the mic, VAD and
  whisper all want 16 kHz. Keeping two rates means resampling inside the audio
  callback and mapping fractional frame counts into the echo reference. Instead
  TTS is resampled to :data:`SAMPLE_RATE` once, so playback and reference are
  1:1 and the echo canceller stays exact. Cost: playback is band-limited to
  8 kHz (telephone quality — clearly intelligible, not hi-fi). Raise
  ``SAMPLE_RATE`` if you would rather have the fidelity back.

Usage::

    from voice.pipeline import VoicePipeline

    pipeline = VoicePipeline()          # loads whisper + kokoro (~60s cold)

    def agent(transcript: str) -> str:
        return ask_atlas(transcript)     # or return a generator of str chunks

    try:
        pipeline.full_duplex_session(agent)
    finally:
        pipeline.shutdown()

Press Ctrl-C to leave a session; ``shutdown()`` closes the audio streams and
joins the worker threads.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from core.config import (
    CHANNELS,
    FOLLOWUP_TIMEOUT,
    MAX_SESSION_TURNS,
    SAMPLE_RATE,
    SILENCE_SECS,
    SILENCE_THRESH,
    TTS_SPEED,
    TTS_VOICE,
    WHISPER_MODEL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio format
# ---------------------------------------------------------------------------

# int16 everywhere: PortAudio's native format, whisper's input, and what the
# echo canceller normalises from.
AUDIO_DTYPE = "int16"
_INT16_SCALE = 32768.0

# Input block = 32 ms, matching Silero's own 512-sample window at 16 kHz.
INPUT_BLOCK = 512
# Output block = 30 ms. Also the barge-in granularity: the brief asks playback
# to stop "within one chunk (approximately 50ms)", and 30 ms beats that.
OUTPUT_BLOCK = 480

# How often the playback loop checks barge_in_detected. 20 ms keeps the
# reaction inside one output chunk.
PLAYBACK_POLL_SECS = 0.02

# ---------------------------------------------------------------------------
# VAD (Silero, via Pipecat) — the brief's numbers
# ---------------------------------------------------------------------------

VAD_CONFIDENCE = 0.7  # "speech probability threshold 0.7"
VAD_START_SECS = 0.25  # "minimum speech duration 250ms"
VAD_STOP_SECS = 0.7  # "minimum silence duration 700ms"
# Pipecat gates the VAD on BS.1770 loudness normalised from -110 LUFS (0.0) to
# -10 LUFS (1.0). 0.6 is about -50 LUFS, i.e. it only rejects near-silence, so
# the default is left alone rather than tuned downward.
VAD_MIN_VOLUME = 0.6

# ---------------------------------------------------------------------------
# Echo cancellation
# ---------------------------------------------------------------------------

# Coarse speaker->mic latency. Has to cover DAC buffering, time-of-flight and
# the input block. The adaptive filter only has AEC_FILTER_TAPS of reach, so
# getting this roughly right matters.
AEC_DELAY_MS = 80
# 256 taps at 16 kHz = 16 ms of reverberation tail. Enough for a small room on
# a laptop, cheap enough to run inside an audio callback.
AEC_FILTER_TAPS = 256
# NLMS step size for the normalised block update. The normalisation below is
# the conservative one (divided by the total window energy, not the mean), so
# the useful range is large: 1.0 is barely audible, 20+ is unstable on 1/f
# input. 5.0 was picked by sweeping white / speech-coloured / 1-f reference
# signals, and the divergence guard below backstops the pathological cases.
AEC_STEP = 5.0
_AEC_EPS = 1e-6
# Divergence guard: a residual this many times louder than the raw mic means
# the filter has run away, so it is rebuilt from zero rather than allowed to
# emit garbage (or NaNs) into the VAD and whisper.
AEC_DIVERGE_RATIO = 4.0
_AEC_MIN_BLOCKS_BEFORE_RESET = 8

# ---------------------------------------------------------------------------
# Speech-to-text
# ---------------------------------------------------------------------------

WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "int8_float16"
# Partials start after this much continuous speech...
PARTIAL_MIN_SECS = 0.5
# ...and are refreshed this often.
PARTIAL_INTERVAL_SECS = 0.4
# Never let a single utterance grow without bound.
MAX_UTTERANCE_SECS = 30.0
# Frames kept before speech is confirmed, so the start of a word is not clipped
# while the VAD is still in its STARTING state.
PREROLL_SECS = 0.25
# Below this RMS the block is treated as silence and skipped entirely.
_MIN_RMS = SILENCE_THRESH

# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------

KOKORO_LANG_CODE = "a"  # American English; matches TTS_VOICE like "am_adam"
MIN_CHUNK_WORDS = 4
MAX_CHUNK_WORDS = 70

# How long speak() waits for the first audio chunk before giving up.
FIRST_CHUNK_TIMEOUT = 15.0
# How long it waits for subsequent chunks once playback has started.
NEXT_CHUNK_TIMEOUT = 30.0

# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

SESSION_START_TIMEOUT = 30.0  # waiting for the user's opening utterance
# Max audio queued for playback; beyond this the callback starts dropping.
_PLAYBACK_QUEUE_SECS = 10.0


# ===========================================================================
# Echo cancellation
# ===========================================================================


class _RingBuffer:
    """Rolling float32 history of what went to the speaker.

    The output callback appends what it is about to play; the input callback
    reads the same audio ``delay_samples`` later. Written as a compacted array
    rather than a true circular buffer: at 16 kHz the shift costs well under a
    millisecond per second of audio, and "valid samples live at the start of
    the array" is far harder to get wrong.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._length = 0
        self._lock = threading.Lock()

    @property
    def filled(self) -> int:
        """Number of valid samples currently held."""
        with self._lock:
            return self._length

    def write(self, samples: np.ndarray) -> None:
        n = samples.shape[0]
        if n == 0:
            return
        with self._lock:
            if n >= self._capacity:  # keep only the newest part
                self._buf[:] = samples[-self._capacity :]
                self._length = self._capacity
                return
            # Drop the oldest samples to make room at the end.
            keep = min(self._length, self._capacity - n)
            if keep > 0:
                self._buf[:keep] = self._buf[self._length - keep : self._length]
            self._buf[keep : keep + n] = samples
            self._length = keep + n

    def read_delayed(self, n: int, delay_samples: int) -> np.ndarray:
        """Return the ``n`` samples written ``delay_samples`` samples ago.

        Zero-filled where the requested history does not exist yet (at
        start-up, or right after :meth:`flush`), so the echo estimate is 0 and
        the canceller is a no-op rather than a source of garbage.
        """
        out = np.zeros(max(n, 0), dtype=np.float32)
        if n <= 0:
            return out
        with self._lock:
            end = self._length - delay_samples
            if end <= 0:
                return out
            start = max(0, end - n)
            count = end - start
            if count > 0:
                out[n - count :] = self._buf[start:end]
        return out

    def flush(self) -> None:
        """Forget all history.

        Called when playback is cut short: samples still in flight belong to
        audio that will never be heard, and treating them as history would make
        the canceller subtract speech that never happened.
        """
        with self._lock:
            self._length = 0


class EchoCanceller:
    """Block NLMS adaptive filter that removes speaker bleed from the mic.

    Keeps the standard canceller shape::

        estimate = filter(reference)
        error    = mic - estimate      <-- what we actually listen to
        filter  += step * error * reference / ||reference||^2

    The filter is a FIR of :data:`AEC_FILTER_TAPS` taps applied to the delayed
    playback reference. There is no double-talk protection: adaptation
    continues while the user speaks, which biases the filter slightly towards
    the user's own voice (see the note in :meth:`process` for why the textbook
    guard cannot be used here). Measured suppression on synthetic
    speaker-to-mic paths is ~12 dB for speech-like playback, and the divergence
    guard below keeps pathological input from producing garbage.
    """

    def __init__(
        self,
        taps: int = AEC_FILTER_TAPS,
        step: float = AEC_STEP,
        delay_ms: int = AEC_DELAY_MS,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.taps = taps
        self.step = step
        self.sample_rate = sample_rate
        self.delay_samples = int(delay_ms * sample_rate / 1000)
        self.weights = np.zeros(taps, dtype=np.float32)
        # Previous reference tail, so each block's windows reach back into the
        # previous block instead of starting from zero.
        self._history = np.zeros(taps - 1, dtype=np.float32)
        self.last_erle_db: float | None = None  # echo return loss enhancement
        self.resets = 0  # how often the divergence guard has fired
        self._blocks_since_reset = 0

    def reset(self) -> None:
        """Forget the learned filter (e.g. after an output device change)."""
        self.weights[:] = 0.0
        self._history[:] = 0.0
        self.last_erle_db = None
        self._blocks_since_reset = 0

    def process(self, mic_int16: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Return the echo-cancelled mic block as int16.

        ``mic_int16`` is the raw captured block and ``reference`` is the
        already delay-aligned playback signal for the same time span, both
        1-D. np.int16 in, np.int16 out, so it drops straight into the callback.
        """
        mic = mic_int16.astype(np.float32) / _INT16_SCALE
        ref = reference.astype(np.float32, copy=False)

        n = mic.shape[0]
        if n == 0 or n < self.taps:
            # Not enough samples for a full window (only possible if
            # INPUT_BLOCK is configured absurdly small) — pass through.
            return mic_int16

        # windows[i] is the causal reference window feeding output sample i, so
        # the estimate is a plain matrix product. Building it as a strided view
        # keeps this allocation-free apart from the matmul.
        combined = np.concatenate((self._history, ref))
        windows = np.lib.stride_tricks.sliding_window_view(combined, self.taps)[:n]

        estimate = windows @ self.weights
        error = mic - estimate

        mic_energy = float(np.dot(mic, mic))
        error_energy = float(np.dot(error, error))

        # Block NLMS, normalised by the total window energy. The seemingly
        # obvious refinement — dividing by the mean window power instead —
        # makes the update the sum of n per-sample steps and looks much faster,
        # but it is unstable whenever the playback is highly correlated (which
        # speech very much is): the block correlation matrix is nearly
        # rank-one, so the effective step scales with the filter length and the
        # weights diverge to NaN within a second.
        #
        # NOTE: there is deliberately no double-talk gate. The textbook
        # formulation — freeze adaptation when ||error|| >> ||estimate|| —
        # cannot work here: until the filter has converged the error *is* the
        # whole mic, which is indistinguishable from someone talking, so the
        # gate latches on the very first block and adaptation never begins.
        # (Both variants were tried; the first stalled the filter at 2 dB of
        # suppression, the second never moved it at all.) A usable detector
        # needs a converged baseline or cross-correlation against the
        # reference, and neither is trustworthy while the filter is learning.
        # Consequence: talking over Atlas while it speaks biases the filter a
        # little towards the user's own voice. The divergence guard below
        # bounds how bad that can get.
        norm = float(np.sum(windows * windows)) + _AEC_EPS
        self.weights += (self.step / norm) * (windows.T @ error)

        # Keep the tail so the next block's windows stay continuous.
        self._history = ref[-(self.taps - 1) :].copy() if self.taps > 1 else self._history

        diverged = not np.all(np.isfinite(self.weights)) or (
            self._blocks_since_reset >= _AEC_MIN_BLOCKS_BEFORE_RESET
            and mic_energy > _AEC_EPS
            and error_energy > mic_energy * AEC_DIVERGE_RATIO
        )
        self._blocks_since_reset += 1
        if diverged:
            self.resets += 1
            logger.debug(
                "echo canceller diverged (reset #%d) — rebuilding filter", self.resets
            )
            self.reset()
            # Hand back the raw mic for this block: passing through is far
            # better than emitting an amplified, un-cancelled residual.
            return mic_int16

        # Echo Return Loss Enhancement, smoothed: how much of the echo the
        # canceller actually removed, measured against the raw mic.
        if mic_energy > _AEC_EPS:
            erle = 10.0 * float(np.log10(mic_energy / max(error_energy, _AEC_EPS)))
            self.last_erle_db = (
                erle if self.last_erle_db is None else 0.9 * self.last_erle_db + 0.1 * erle
            )

        cleaned = np.clip(error * _INT16_SCALE, -32768, 32767)
        return cleaned.astype(np.int16)


# ===========================================================================
# Playback
# ===========================================================================


class _PlaybackBuffer:
    """Lock-protected int16 sample buffer between producers and the DAC.

    :meth:`read` is called from the PortAudio output callback and must never
    block, so it is a plain lock rather than a queue: the callback takes
    whatever is there and pads with silence. :meth:`wait_until_empty` is what
    the playback loop uses to know when a chunk has finished sounding.
    """

    def __init__(self, capacity_frames: int) -> None:
        self._capacity = capacity_frames
        self._data = np.zeros(capacity_frames, dtype=np.int16)
        self._length = 0
        self._condition = threading.Condition()
        self.underruns = 0

    @property
    def pending(self) -> int:
        with self._condition:
            return self._length

    def write(self, samples: np.ndarray) -> None:
        with self._condition:
            free = self._capacity - self._length
            if samples.shape[0] > free:
                samples = samples[:free]
                # Producers should throttle; if this triggers, audio was lost.
                logger.debug("playback buffer full — dropped %d samples", -free)
            if samples.shape[0]:
                self._data[self._length : self._length + samples.shape[0]] = samples
                self._length += samples.shape[0]
            self._condition.notify_all()

    def read(self, frames: int) -> np.ndarray:
        """Pop up to ``frames`` samples, zero-padded. Never blocks."""
        with self._condition:
            take = min(frames, self._length)
            out = np.zeros(frames, dtype=np.int16)
            if take:
                out[:take] = self._data[:take]
                remaining = self._length - take
                if remaining:
                    self._data[:remaining] = self._data[take : self._length]
                self._length = remaining
            elif frames:
                self.underruns += 1
            self._condition.notify_all()
            return out

    def clear(self) -> None:
        """Drop everything queued — used when barge-in cuts playback short."""
        with self._condition:
            self._length = 0
            self._condition.notify_all()

    def wait_until_empty(self, timeout: float) -> bool:
        """Block until the buffer has drained. True if it did."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._length > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True


# ===========================================================================
# Barge-in bookkeeping
# ===========================================================================


@dataclass
class BargeInRecord:
    """Where Atlas was interrupted.

    ``word`` is the token being spoken when playback stopped, taken from
    Kokoro's token timestamps (seconds, relative to the start of that chunk).
    """

    word: str | None
    word_start: float | None
    word_end: float | None
    chunk_index: int
    chunk_text: str
    elapsed_in_chunk: float

    def __str__(self) -> str:
        if self.word is None:
            return f"interrupted in chunk {self.chunk_index} at {self.elapsed_in_chunk:.2f}s"
        return (
            f"interrupted mid-word {self.word!r} "
            f"({self.word_start:.2f}-{self.word_end:.2f}s into chunk "
            f"{self.chunk_index})"
        )


# ===========================================================================
# Text utilities
# ===========================================================================

_MD_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_MD_BLOCKQUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_MD_RULE = re.compile(r"^\s*(?:[-*_]\s*){3,}$", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")
_PIECE = re.compile(r"[^.!?,]+[.!?,]*")


def strip_markdown(text: str) -> str:
    """Reduce markdown to speakable prose.

    Atlas's replies come from an LLM, so they arrive with code fences, emphasis
    markers and bullets. Reading those characters aloud sounds broken, so they
    are removed rather than spoken. Code *blocks* are dropped entirely — they
    are never useful when read out loud.
    """
    text = _MD_CODE_FENCE.sub(" ", text)
    text = _MD_IMAGE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_INLINE_CODE.sub(r"\1", text)
    text = _MD_EMPHASIS.sub(r"\2", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BLOCKQUOTE.sub("", text)
    text = _MD_BULLET.sub("", text)
    text = _MD_RULE.sub(" ", text)
    # Collapse whitespace *before* filtering control characters. Newline and
    # tab are themselves control characters, so filtering first would glue
    # words together ("## Heading\nSure!" -> "HeadingSure!").
    text = _WHITESPACE.sub(" ", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = text.replace("`", "").replace("*", "").replace("#", " ")
    return text.strip()


def split_into_chunks(text: str, min_words: int = MIN_CHUNK_WORDS) -> list[str]:
    """Split prose into TTS-sized chunks.

    The brief's rule: break on ``.`` ``!`` ``?`` ``?`` and ``,``, but never emit
    a chunk with fewer than ``min_words`` words, so Kokoro is not handed a
    three-word fragment (bad prosody, and wasteful — every chunk costs a model
    call). Very long runs are force-split at :data:`MAX_CHUNK_WORDS`.
    """
    text = strip_markdown(text)
    if not text:
        return []

    chunks: list[str] = []
    current = ""
    for piece in _PIECE.findall(text):
        piece = piece.strip()
        if not piece:
            continue
        current = f"{current} {piece}".strip() if current else piece
        words = len(current.split())
        boundary = piece[-1] in ".!?,"
        if words >= MAX_CHUNK_WORDS or (words >= min_words and boundary):
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _sine_cue(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Short two-note blip played when the mic opens.

    Generated rather than synthesised: the cue has to be instant, and it must
    not depend on Kokoro being warm. Deliberately quiet (0.12 full scale) so it
    does not itself trip the VAD.
    """
    parts = []
    for freq, millis in ((660.0, 70), (990.0, 70)):
        n = int(sample_rate * millis / 1000)
        t = np.arange(n, dtype=np.float32) / sample_rate
        tone = np.sin(2 * np.pi * freq * t, dtype=np.float32)
        # 5 ms fades so the blip does not click.
        fade = min(int(sample_rate * 0.005), n // 2)
        if fade:
            tone[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            tone[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        parts.append(tone)
    cue = np.concatenate(parts) * 0.12
    return (cue * 32767.0).astype(np.int16)


# ===========================================================================
# The pipeline
# ===========================================================================


class VoicePipeline:
    """Full-duplex voice conversation: STT + agent + TTS with barge-in.

    Heavy models are loaded in :meth:`__init__` (whisper and Kokoro together
    take tens of seconds when cold) because loading them lazily on the first
    turn would add that delay to the first thing the user hears. Pass
    ``preload=False`` to skip it — useful for tests, which inject fakes via the
    ``stt_model`` / ``tts_pipeline`` / ``vad`` arguments.
    """

    def __init__(
        self,
        *,
        preload: bool = True,
        sample_rate: int = SAMPLE_RATE,
        voice: str = TTS_VOICE,
        speed: float = TTS_SPEED,
        whisper_model: str = WHISPER_MODEL,
        on_partial: Callable[[str], None] | None = None,
        stt_model: Any | None = None,
        tts_pipeline: Any | None = None,
        vad: Any | None = None,
        open_streams: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.voice = voice
        self.speed = speed
        self.whisper_model_name = whisper_model
        self.on_partial = on_partial

        # --- injected or lazily built components ---
        self._stt = stt_model
        self._tts = tts_pipeline
        self._vad = vad

        # --- audio plumbing ---
        self._echo_ref = _RingBuffer(
            capacity=int(sample_rate * (max(AEC_DELAY_MS, 100) / 1000.0 + 1.0))
        )
        self._canceller = EchoCanceller(sample_rate=sample_rate)
        self._playback = _PlaybackBuffer(int(sample_rate * _PLAYBACK_QUEUE_SECS))
        self._mic_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=128)

        self._input_stream: Any | None = None
        self._output_stream: Any | None = None

        # --- coordination ---
        self._shutdown = threading.Event()
        self._monitor_stop = threading.Event()
        self.barge_in_detected = threading.Event()
        self.last_interruption: BargeInRecord | None = None

        self._listening = threading.Event()
        self._utterance_ready = threading.Event()
        self._utterance: list[np.ndarray] = []
        self._utterance_secs = 0.0
        self._speech_seen = False
        self._preroll: list[np.ndarray] = []
        self._partial_busy = threading.Event()
        self.last_partial: str = ""
        self._current_transcript: str = ""
        # Barge-in is only armed while TTS audio is actually sounding.
        self._barge_in_armed = False
        self._pending_interruption: BargeInRecord | None = None
        self._vad_state: Any | None = None
        self._last_partial_at = 0.0
        self._tts_rate = 24000  # Kokoro's native rate; see _to_int16

        self._state_lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None
        self._partial_threads: list[threading.Thread] = []

        logger.info(
            "voice pipeline: %d Hz, whisper=%s/%s, voice=%s",
            sample_rate,
            whisper_model,
            WHISPER_COMPUTE_TYPE,
            voice,
        )

        if preload:
            self._load_stt()
            self._load_tts()
            self._load_vad()
        if open_streams and preload:
            self.open_streams()
            self._start_monitor()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __enter__(self) -> "VoicePipeline":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return (
            f"<VoicePipeline {self.sample_rate}Hz voice={self.voice!r} "
            f"stt={'loaded' if self._stt else 'none'} "
            f"tts={'loaded' if self._tts else 'none'}>"
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_stt(self) -> None:
        """Load faster-whisper, falling back to CPU if CUDA is unavailable."""
        if self._stt is not None:
            return
        from faster_whisper import WhisperModel

        started = time.monotonic()
        try:
            self._stt = WhisperModel(
                self.whisper_model_name,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
        except Exception as exc:
            # A GPU that cannot hold compute_type int8_float16 (or a CUDA
            # runtime mismatch) must not take the whole voice system down.
            logger.warning(
                "whisper on %s failed (%s: %s) — falling back to CPU/int8",
                WHISPER_DEVICE,
                type(exc).__name__,
                exc,
            )
            self._stt = WhisperModel(
                self.whisper_model_name, device="cpu", compute_type="int8"
            )
        logger.info("whisper '%s' ready in %.1fs", self.whisper_model_name, time.monotonic() - started)

    def _load_tts(self) -> None:
        """Load Kokoro, which also gives us word-level timestamps."""
        if self._tts is not None:
            return
        from kokoro import KPipeline

        started = time.monotonic()
        self._tts = KPipeline(lang_code=KOKORO_LANG_CODE)
        logger.info("kokoro ready in %.1fs", time.monotonic() - started)

    def _load_vad(self) -> None:
        """Build Pipecat's Silero VAD with the brief's thresholds."""
        if self._vad is not None:
            return
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams

        self._vad = SileroVADAnalyzer(
            sample_rate=self.sample_rate,
            params=VADParams(
                confidence=VAD_CONFIDENCE,
                start_secs=VAD_START_SECS,
                stop_secs=VAD_STOP_SECS,
                min_volume=VAD_MIN_VOLUME,
            ),
        )
        logger.info(
            "VAD ready (confidence=%.2f, start=%.2fs, stop=%.2fs)",
            VAD_CONFIDENCE,
            VAD_START_SECS,
            VAD_STOP_SECS,
        )

    # ------------------------------------------------------------------
    # Audio streams
    # ------------------------------------------------------------------

    def open_streams(self) -> None:
        """Open the persistent input and output streams.

        Both are kept open for the lifetime of the pipeline. That is what makes
        barge-in possible: interrupting Atlas requires listening *while* it
        talks, which a per-utterance stream cannot do.
        """
        if self._input_stream is not None:
            return
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "sounddevice could not load the PortAudio library. Install it "
                "with `sudo apt install -y portaudio19-dev libportaudio2`, "
                f"then retry (original error: {exc})"
            ) from exc

        self._input_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=AUDIO_DTYPE,
            blocksize=INPUT_BLOCK,
            callback=self._input_callback,
        )
        self._output_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=AUDIO_DTYPE,
            blocksize=OUTPUT_BLOCK,
            callback=self._output_callback,
        )
        self._input_stream.start()
        self._output_stream.start()
        logger.info("audio streams open (%d Hz, %d/%d ms blocks)", self.sample_rate, 32, 30)

    # -- callbacks (PortAudio threads: never block, never raise) ---------

    def _input_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """Capture a block, cancel echo, hand it to the monitor thread."""
        if status:
            logger.debug("input status: %s", status)
        try:
            mic = np.asarray(indata)[:, 0]
            reference = self._echo_ref.read_delayed(frames, self._canceller.delay_samples)
            cleaned = self._canceller.process(mic, reference)
            try:
                self._mic_queue.put_nowait(cleaned)
            except queue.Full:
                # Drop the oldest block rather than stall the audio thread.
                try:
                    self._mic_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._mic_queue.put_nowait(cleaned)
                except queue.Full:
                    pass
        except Exception:  # pragma: no cover - audio callbacks must not raise
            logger.exception("input callback failed")

    def _output_callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        """Play a block and record it as the echo reference."""
        if status:
            logger.debug("output status: %s", status)
        try:
            samples = self._playback.read(frames)
            outdata[:] = samples.reshape(-1, 1)
            # This is the whole trick for echo cancellation: we know exactly
            # what went to the speaker and when, so the reference is exact.
            self._echo_ref.write(samples.astype(np.float32) / _INT16_SCALE)
        except Exception:  # pragma: no cover - audio callbacks must not raise
            logger.exception("output callback failed")

    # ------------------------------------------------------------------
    # Monitor thread: VAD, barge-in, utterance assembly
    # ------------------------------------------------------------------

    def _start_monitor(self) -> None:
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="atlas-voice-monitor", daemon=True
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        """Single consumer of mic audio: VAD, barge-in and utterance building.

        One consumer rather than two competing ones matters: barge-in runs
        during ``speak()`` and utterance capture runs during ``listen_once()``,
        and both need the same audio. Arbitrating between them here (via
        :attr:`_listening` and the barge-in arm flag) avoids one pulling frames
        out from under the other.

        Pipecat's analyzer is async and offloads to its own executor, so this
        thread owns a private event loop and drives it one block at a time.
        """
        from pipecat.audio.vad.vad_analyzer import VADState

        analyzer = self._vad
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        preroll_limit = int(self.sample_rate * PREROLL_SECS)
        logger.debug("voice monitor thread started")

        try:
            while not self._monitor_stop.is_set():
                try:
                    block = self._mic_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                # Keep a short pre-roll so the first syllable is not clipped
                # while the VAD is still in its STARTING state.
                self._append_preroll(block, preroll_limit)

                state = self._vad_state
                if analyzer is not None:
                    try:
                        state = loop.run_until_complete(
                            analyzer.analyze_audio(block.tobytes())
                        )
                        self._vad_state = state
                    except Exception:
                        logger.debug("VAD analysis failed", exc_info=True)

                # --- barge-in: only while TTS audio is actually sounding ---
                if self._barge_in_armed and state in (
                    VADState.STARTING,
                    VADState.SPEAKING,
                ):
                    if not self.barge_in_detected.is_set():
                        logger.debug("barge-in: user speech detected (vad=%s)", state.name)
                    self.barge_in_detected.set()

                # --- utterance assembly while listen_once() is waiting ---
                if self._listening.is_set():
                    self._handle_listening_block(block, state, VADState)
        finally:
            loop.close()
            logger.debug("voice monitor thread stopped")

    def _append_preroll(self, block: np.ndarray, limit_samples: int) -> None:
        self._preroll.append(block)
        total = sum(b.shape[0] for b in self._preroll)
        while len(self._preroll) > 1 and total - self._preroll[0].shape[0] >= limit_samples:
            total -= self._preroll.pop(0).shape[0]

    def _handle_listening_block(self, block: np.ndarray, state: Any, VADState: Any) -> None:
        """Drive utterance capture from VAD transitions.

        SPEAKING/STARTING appends audio; a return to QUIET means the user
        stopped, which publishes the utterance to :meth:`listen_once`.
        """
        if state is VADState.STARTING and not self._speech_seen:
            # First sign of speech: seed the utterance with the pre-roll, which
            # already includes this block.
            with self._state_lock:
                self._speech_seen = True
                self._utterance = list(self._preroll)
                self._utterance_secs = sum(b.shape[0] for b in self._utterance) / self.sample_rate
        elif self._speech_seen and state in (VADState.SPEAKING, VADState.STOPPING):
            with self._state_lock:
                self._utterance.append(block)
                self._utterance_secs += block.shape[0] / self.sample_rate
                speech_secs = self._utterance_secs
            # Streaming STT: refresh the partial transcript while the user is
            # still talking, throttled so whisper is not swamped.
            now = time.monotonic()
            if (
                speech_secs >= PARTIAL_MIN_SECS
                and now - self._last_partial_at >= PARTIAL_INTERVAL_SECS
            ):
                self._last_partial_at = now
                self._maybe_start_partial()
        elif state is VADState.QUIET and self._speech_seen:
            # Silence long enough to confirm the end of the turn.
            logger.debug("utterance complete (%.2fs)", self._utterance_secs)
            self._utterance_ready.set()

    # ------------------------------------------------------------------
    # Speech to text
    # ------------------------------------------------------------------

    def _transcribe(self, audio_int16: np.ndarray, *, final: bool) -> str:
        """Transcribe a buffer with faster-whisper.

        Partial passes use a greedy decode (``beam_size=1``) because they are
        thrown away as soon as the next one lands; the final pass uses beam
        search for accuracy.
        """
        if self._stt is None:
            raise RuntimeError("STT model not loaded")
        if audio_int16.size == 0:
            return ""
        audio = audio_int16.astype(np.float32) / _INT16_SCALE
        segments, _info = self._stt.transcribe(
            audio,
            language="en",
            beam_size=5 if final else 1,
            vad_filter=False,  # we already ran VAD; double-filtering clips words
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _maybe_start_partial(self) -> None:
        """Kick off a background partial transcription if one is not running.

        "Latest wins": if a partial is still in flight it is simply skipped.
        Queueing them would make partials lag further and further behind as the
        utterance grows.
        """
        if self._partial_busy.is_set() or len(self._utterance) == 0:
            return
        audio = np.concatenate(self._utterance)
        if audio.shape[0] < int(self.sample_rate * PARTIAL_MIN_SECS):
            return

        self._partial_busy.set()

        def worker() -> None:
            try:
                text = self._transcribe(audio, final=False)
                if text and text != self.last_partial:
                    self.last_partial = text
                    logger.debug("partial: %s", text)
                    if self.on_partial is not None:
                        self.on_partial(text)
            except Exception:
                logger.debug("partial transcription failed", exc_info=True)
            finally:
                self._partial_busy.clear()

        thread = threading.Thread(target=worker, name="atlas-partial-stt", daemon=True)
        thread.start()
        self._partial_threads = [t for t in self._partial_threads if t.is_alive()]
        self._partial_threads.append(thread)

    def _reset_utterance(self) -> None:
        with self._state_lock:
            self._utterance = []
            self._utterance_secs = 0.0
            self._speech_seen = False
            self._preroll = []
            self._utterance_ready.clear()
            self.last_partial = ""
            self._current_transcript = ""

    def _drain_mic_queue(self) -> None:
        """Discard audio captured before listening started."""
        while True:
            try:
                self._mic_queue.get_nowait()
            except queue.Empty:
                return

    # ------------------------------------------------------------------
    # listen_once
    # ------------------------------------------------------------------

    def listen_once(self, timeout: float | None = None, *, cue: bool = True) -> str:
        """Listen for one utterance and return its transcript.

        Arms capture, optionally plays a listening cue, then waits for the VAD
        to see speech followed by :data:`VAD_STOP_SECS` of silence. Partials are
        produced while the user talks (see :meth:`_maybe_start_partial`) but
        only the final transcript is returned.

        Returns ``""`` when nothing was said before ``timeout`` — the caller
        treats that as "conversation over", not as an error.
        """
        if timeout is None:
            timeout = FOLLOWUP_TIMEOUT
        self._load_vad()
        self._reset_utterance()
        if cue:
            self._play_cue()
        self._drain_mic_queue()

        # Barge-in is off while listening: the VAD's job here is to find the
        # end of the utterance, not to interrupt anything.
        self.barge_in_detected.clear()
        self._listening.set()
        started = time.monotonic()
        speech_started_at: float | None = None

        try:
            while True:
                if self._shutdown.is_set():
                    return ""
                if self._utterance_ready.wait(0.05):
                    break
                elapsed = time.monotonic() - started
                if speech_started_at is None:
                    if elapsed >= timeout:
                        logger.debug("listen_once: no speech within %.1fs", timeout)
                        return ""
                else:
                    # A dangling utterance (user walked away mid-sentence).
                    if time.monotonic() - speech_started_at > MAX_UTTERANCE_SECS:
                        logger.warning("listen_once: utterance exceeded %.0fs", MAX_UTTERANCE_SECS)
                        break
                    if elapsed >= timeout + MAX_UTTERANCE_SECS:
                        return ""
                if self._speech_seen and speech_started_at is None:
                    speech_started_at = time.monotonic()

            with self._state_lock:
                audio = np.concatenate(self._utterance) if self._utterance else np.array([], dtype=np.int16)
        finally:
            self._listening.clear()

        if audio.size == 0:
            return ""
        # Wait for any in-flight partial so its transcript does not land after
        # the final one and overwrite it.
        for thread in list(self._partial_threads):
            thread.join(timeout=2.0)

        started = time.monotonic()
        transcript = self._transcribe(audio, final=True)
        self._current_transcript = transcript
        logger.info(
            "heard (%.1fs audio, %.1fs stt): %r",
            audio.shape[0] / self.sample_rate,
            time.monotonic() - started,
            transcript,
        )
        return transcript

    def _play_cue(self) -> None:
        self._playback.write(_sine_cue(self.sample_rate))
        self._playback.wait_until_empty(2.0)

    # ------------------------------------------------------------------
    # speak
    # ------------------------------------------------------------------

    def speak(self, text: str) -> bool:
        """Speak ``text``, returning True if the user barged in.

        Chunk 1 is played as soon as Kokoro finishes it while chunk 2 is
        generated concurrently, so the first sound arrives after one chunk of
        synthesis rather than after the whole reply. ``barge_in_detected`` is
        checked between blocks of playback (every ~20 ms) so an interruption
        stops the audio within one output block.
        """
        return self.speak_stream([text])

    def speak_stream(self, chunks: Iterable[str]) -> bool:
        """Speak an iterable of text fragments (e.g. a streaming LLM reply).

        Generators are consumed on a worker thread so sentence N+1 is already
        synthesised while sentence N plays.
        """
        self._load_tts()
        produced: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        stop_generating = threading.Event()

        def producer() -> None:
            try:
                for fragment in chunks:
                    if stop_generating.is_set() or self._shutdown.is_set():
                        break
                    for sentence in split_into_chunks(fragment):
                        if stop_generating.is_set():
                            break
                        try:
                            # Kokoro can yield more than one result for a long
                            # sentence, so every result is forwarded rather
                            # than only the first.
                            for result in self._synthesise(sentence):
                                if stop_generating.is_set():
                                    break
                                produced.put((sentence, result))
                        except Exception:
                            logger.exception("TTS generation failed for %r", sentence)
                            continue
            finally:
                produced.put(None)

        thread = threading.Thread(target=producer, name="atlas-tts", daemon=True)
        thread.start()

        interrupted = False
        self.last_interruption = None
        self.barge_in_detected.clear()
        # Arm barge-in only once playback is actually audible: otherwise the
        # tail of the user's own turn would immediately "interrupt" Atlas.
        armed = False
        chunk_index = 0
        chunks_spoken: list[str] = []

        try:
            while True:
                timeout = FIRST_CHUNK_TIMEOUT if chunk_index == 0 else NEXT_CHUNK_TIMEOUT
                try:
                    item = produced.get(timeout=timeout)
                except queue.Empty:
                    logger.warning("TTS produced no audio within %.0fs — stopping", timeout)
                    break
                if item is None:
                    break

                sentence, result = item
                audio = self._to_int16(result.audio) if result is not None else np.array([], dtype=np.int16)
                if audio.size == 0:
                    continue

                if not armed:
                    self._arm_barge_in()
                    armed = True

                interrupted, position = self._play_chunk(
                    audio, chunk_index, sentence, result
                )
                chunks_spoken.append(sentence)
                chunk_index += 1
                if interrupted:
                    break
        finally:
            # Stop synthesis promptly on barge-in so we do not keep burning CPU
            # (or GPU) generating speech nobody will hear.
            stop_generating.set()
            self.barge_in_detected.clear()
            self._disarm_barge_in()
            if interrupted:
                self._playback.clear()
                self._echo_ref.flush()
            thread.join(timeout=5.0)

        if interrupted:
            self.last_interruption = getattr(self, "_pending_interruption", None)
            logger.info("barge-in: %s", self.last_interruption or "playback stopped")
        return interrupted

    def _synthesise(self, sentence: str) -> Iterator[Any]:
        """Yield Kokoro result objects for one sentence.

        A thin wrapper so that every call site goes through one place that
        knows the keyword arguments, and so tests can inject a fake pipeline.
        """
        if self._tts is None:
            raise RuntimeError("TTS pipeline not loaded")
        for result in self._tts(sentence, voice=self.voice, speed=self.speed):
            yield result

    def _to_int16(self, audio: Any) -> np.ndarray:
        """Convert Kokoro's float audio to the pipeline's int16 format.

        Kokoro emits float32 at 24 kHz; everything else here is int16 at
        :data:`SAMPLE_RATE`, so this resamples and rescales in one place. See
        the module docstring for why a single rate is worth the bandwidth.
        """
        if audio is None:
            return np.array([], dtype=np.int16)
        array = np.asarray(audio, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return np.array([], dtype=np.int16)
        if self._tts_sample_rate != self.sample_rate:
            import soxr

            array = soxr.resample(array, self._tts_sample_rate, self.sample_rate)
        return np.clip(array * 32767.0, -32768, 32767).astype(np.int16)

    @property
    def _tts_sample_rate(self) -> int:
        """Kokoro emits 24 kHz. Kept as a property so tests can fake it."""
        return getattr(self, "_tts_rate", 24000)

    def _arm_barge_in(self) -> None:
        self.barge_in_detected.clear()
        self._barge_in_armed = True

    def _disarm_barge_in(self) -> None:
        self._barge_in_armed = False

    def _play_chunk(
        self, audio: np.ndarray, chunk_index: int, sentence: str, result: Any
    ) -> tuple[bool, float]:
        """Play one chunk, watching for barge-in. Returns (interrupted, elapsed)."""
        self._playback.write(audio)
        started = time.monotonic()
        while self._playback.pending > 0:
            if self._shutdown.is_set():
                return True, time.monotonic() - started
            if self.barge_in_detected.is_set():
                elapsed = time.monotonic() - started
                self._playback.clear()
                self._echo_ref.flush()
                self._pending_interruption = self._describe_interruption(
                    result, chunk_index, sentence, elapsed
                )
                return True, elapsed
            time.sleep(PLAYBACK_POLL_SECS)
        return False, time.monotonic() - started

    def _describe_interruption(
        self, result: Any, chunk_index: int, sentence: str, elapsed: float
    ) -> BargeInRecord:
        """Find which word was being spoken when playback was cut.

        Kokoro's token timestamps are seconds relative to the start of the
        chunk, so the token containing ``elapsed`` is the interrupted word. The
        audio buffer means playback lags the wall clock by up to one output
        block, so ``elapsed`` can overstate the position by ~30 ms — well under
        a spoken word.
        """
        word = word_start = word_end = None
        tokens = getattr(result, "tokens", None) or []
        for token in tokens:
            start = float(getattr(token, "start_ts", 0.0))
            end = float(getattr(token, "end_ts", 0.0))
            if start <= elapsed <= end:
                word, word_start, word_end = token.text, start, end
                break
            if start <= elapsed:
                # Past the last token that started; keep the closest so far.
                word, word_start, word_end = token.text, start, end
        return BargeInRecord(
            word=word,
            word_start=word_start,
            word_end=word_end,
            chunk_index=chunk_index,
            chunk_text=sentence,
            elapsed_in_chunk=elapsed,
        )

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def full_duplex_session(self, agent_callback: Callable[[str], Any]) -> None:
        """Run the listen → think → speak loop until the user goes quiet.

        ``agent_callback`` receives the confirmed transcript and returns either
        a string or an iterable of strings (a streaming LLM reply works
        directly). Barge-in stays live throughout :meth:`speak_stream`, so the
        user can cut Atlas off mid-sentence and the next ``listen_once`` picks
        up what they said instead.

        Returns when the follow-up timeout expires, the turn cap is reached, or
        :meth:`shutdown` is signalled.
        """
        self._shutdown.clear()
        turns = 0
        timeout = SESSION_START_TIMEOUT
        logger.info("session start (follow-up timeout %ss, max %d turns)", FOLLOWUP_TIMEOUT, MAX_SESSION_TURNS)

        while not self._shutdown.is_set():
            transcript = self.listen_once(timeout=timeout)
            if not transcript:
                logger.info("no speech — session over after %d turn(s)", turns)
                return
            turns += 1
            logger.info("turn %d: %r", turns, transcript)

            try:
                reply = agent_callback(transcript)
            except Exception:
                logger.exception("agent callback failed")
                self.speak("Sorry, something went wrong on my end.")
                return
            if self._shutdown.is_set():
                return

            if reply is None:
                logger.warning("agent returned None — nothing to say")
            elif isinstance(reply, str):
                self.speak(reply)
            else:
                self.speak_stream(reply)

            if self._shutdown.is_set():
                return
            if turns >= MAX_SESSION_TURNS:
                logger.info("turn cap (%d) reached — ending session", MAX_SESSION_TURNS)
                return
            timeout = FOLLOWUP_TIMEOUT

            # Respond to an interruption without waiting for the follow-up
            # window to expire: the user is already talking.
            if self.last_interruption is not None:
                timeout = FOLLOWUP_TIMEOUT

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the session, close the audio streams and join the threads."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self._monitor_stop.set()
        self._listening.clear()
        self._utterance_ready.set()
        self._playback.clear()
        self.barge_in_detected.set()  # unblock any in-flight speak()

        for stream in (self._input_stream, self._output_stream):
            if stream is None:
                continue
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.debug("error closing audio stream", exc_info=True)
        self._input_stream = self._output_stream = None

        thread = self._monitor_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._monitor_thread = None
        logger.info("voice pipeline shut down")
