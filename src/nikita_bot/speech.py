import subprocess
import tempfile
import threading
import wave
from pathlib import Path
import re

import imageio_ffmpeg
from piper import PiperVoice


def clean_text_for_speech(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*>\s*", "", text)
    text = re.sub(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", text)
    text = re.sub(r"[*_`~]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class PiperSpeaker:
    def __init__(self, voice_path: str) -> None:
        self.voice_path = Path(voice_path)
        self._voice: PiperVoice | None = None
        self._lock = threading.Lock()

    def synthesize_ogg(self, text: str) -> bytes:
        text = clean_text_for_speech(text)
        if not text:
            raise ValueError("Нельзя озвучить пустой текст")

        with self._lock:
            if self._voice is None:
                if not self.voice_path.is_file():
                    raise FileNotFoundError(
                        f"Не найден голос Piper: {self.voice_path}. "
                        "Скачай его командой из README.md."
                    )
                self._voice = PiperVoice.load(str(self.voice_path))

            with tempfile.TemporaryDirectory() as temp_dir:
                wav_path = Path(temp_dir) / "reply.wav"
                ogg_path = Path(temp_dir) / "reply.ogg"
                with wave.open(str(wav_path), "wb") as wav_file:
                    self._voice.synthesize_wav(text, wav_file)

                subprocess.run(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        str(wav_path),
                        "-c:a",
                        "libopus",
                        "-b:a",
                        "32k",
                        "-vbr",
                        "on",
                        "-application",
                        "voip",
                        str(ogg_path),
                    ],
                    check=True,
                    capture_output=True,
                )
                return ogg_path.read_bytes()