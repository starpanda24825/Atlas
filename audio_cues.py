# audio_cues.py
import numpy as np
import sounddevice as sd


def _tone(freq: float, duration: float, volume: float = 0.25, sr: int = 44100):
    t    = np.linspace(0, duration, int(sr * duration), False)
    wave = np.sin(2 * np.pi * freq * t) * volume
    fade = int(sr * 0.012)
    wave[:fade]  *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)
    sd.play(wave.astype(np.float32), sr)
    sd.wait()


def cue_listening():
    """Two quick ascending tones — Atlas is ready to hear you."""
    _tone(880,  0.07)
    _tone(1100, 0.07)


def cue_thinking():
    """Single soft mid tone — Atlas is processing."""
    _tone(660, 0.09, volume=0.12)


def cue_done():
    """Descending pair — Atlas has finished speaking."""
    _tone(880, 0.07)
    _tone(660, 0.07)


def cue_error():
    """Low double tone — something went wrong."""
    _tone(220, 0.14)
    _tone(180, 0.14)


def cue_timeout():
    """Very soft single low tone — conversation session ended."""
    _tone(440, 0.09, volume=0.08)
