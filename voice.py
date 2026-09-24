# voice.py
import re
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime
from faster_whisper import WhisperModel
from kokoro import KPipeline
from config import (
    WHISPER_MODEL, SAMPLE_RATE, CHANNELS,
    SILENCE_THRESH, SILENCE_SECS, TTS_VOICE, TTS_SPEED
)


def _strip_markdown(text: str) -> str:
    """Remove markdown so TTS reads cleanly."""
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"`+", "", text)
    return text.strip()


class VoicePipeline:
    def __init__(self):
        print("[Voice] Loading Whisper STT model...")
        self.whisper = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
        )

        print("[Voice] Loading Kokoro TTS model...")
        self.tts = KPipeline(lang_code="a")  # 'a' = American English

        print("[Voice] Ready.")

    # ── TTS ──────────────────────────────────────────────────────────────

    def speak(self, text: str):
        clean = _strip_markdown(text)
        print(f"[Atlas]: {clean}")

        generator = self.tts(clean, voice=TTS_VOICE, speed=TTS_SPEED)
        for _, _, audio in generator:
            sd.play(audio, samplerate=24000)
            sd.wait()

    # ── STT ──────────────────────────────────────────────────────────────

    def listen_once(self, timeout: int = 12) -> str:
        """Record until silence, then return transcribed text."""
        print("[Voice] Listening...")

        frames      = []
        silent_frames = 0
        required    = int(SILENCE_SECS * SAMPLE_RATE / 1024)
        has_speech  = False

        def callback(indata, frame_count, time_info, status):
            nonlocal silent_frames, has_speech
            frames.append(indata.copy())
            rms = float(np.sqrt(np.mean(indata ** 2)))
            if rms > SILENCE_THRESH:
                has_speech    = True
                silent_frames = 0
            elif has_speech:
                silent_frames += 1

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=1024,
            callback=callback,
        ):
            start = datetime.now()
            while True:
                sd.sleep(100)
                if has_speech and silent_frames >= required:
                    break
                if (datetime.now() - start).seconds >= timeout:
                    break

        if not has_speech or not frames:
            return ""

        audio = np.concatenate(frames).flatten()

        # Guard — ignore recordings shorter than 0.8 seconds of actual speech
        if len(audio) / SAMPLE_RATE < 0.8:
            return ""

        segments, _ = self.whisper.transcribe(audio, language="en", beam_size=5, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
        text = " ".join(s.text for s in segments).strip()
        print(f"[You]: {text}")
        return text

    # ── Wake Word ────────────────────────────────────────────────────────

    def wait_for_wake_word(self) -> bool:
        """Listen in rolling chunks and use Whisper to detect the wake phrase."""
        wake_phrases = ["hey atlas", "atlas", "hay atlas"]  # variations/mishears
        chunk_samples = int(SAMPLE_RATE * 2.0)  # 2-second listening windows

        print("[Voice] Waiting for 'Hey Atlas'...")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
        ) as stream:
            while True:
                audio, _ = stream.read(chunk_samples)
                audio_flat = audio.flatten()

                # Energy gate — skip transcription on silence
                rms = float(np.sqrt(np.mean(audio_flat ** 2)))
                if rms < SILENCE_THRESH:
                    continue

                # Transcribe the chunk quickly
                segments, _ = self.whisper.transcribe(
                    audio_flat,
                    language="en",
                    beam_size=1,       # speed over accuracy — we only need keyword
                    vad_filter=True,
                )
                text = " ".join(s.text for s in segments).lower().strip()

                if text and any(phrase in text for phrase in wake_phrases):
                    print(f"[Voice] Wake phrase detected in: '{text}'")
                    return True
