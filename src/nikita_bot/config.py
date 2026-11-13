import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    telegram_mode: str
    telegram_bot_token: str | None
    api_id: str | None
    api_hash: str | None
    session_path: str
    ollama_url: str
    ollama_model: str
    piper_voice_path: str
    sd_cpp_path: str
    sd_turbo_model_path: str
    allow_bot_senders: bool
    group_reply_probability: float
    proactive_group_enabled: bool
    proactive_group_chance: float
    proactive_group_cooldown_minutes: int
    proactive_group_min_messages: int
    max_history_messages: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        mode = os.getenv("TELEGRAM_MODE", "user").strip().lower()

        if mode not in {"bot", "user"}:
            raise ValueError("TELEGRAM_MODE должен быть bot или user")

        return cls(
            telegram_mode=mode,
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None,
            api_id=os.getenv("API_ID", "").strip() or None,
            api_hash=os.getenv("API_HASH", "").strip() or None,
            session_path=os.getenv("TELEGRAM_SESSION_PATH", ".sessions/nikita_user"),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").strip().rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3.5:4b").strip(),
            piper_voice_path=os.getenv(
                "PIPER_VOICE_PATH", ".models/piper/ru_RU-ruslan-medium.onnx"
            ),
            sd_cpp_path=os.getenv("SD_CPP_PATH", ".tools/stable-diffusion/sd-cli.exe"),
            sd_turbo_model_path=os.getenv(
                "SD_TURBO_MODEL_PATH", ".models/sd-turbo/sd_turbo-f16-q8_0.gguf"
            ),
            allow_bot_senders=os.getenv(
                "ALLOW_BOT_SENDERS", "true" if mode == "user" else "false"
            ).lower()
            in {"1", "true", "yes", "on"},
            group_reply_probability=min(
                1.0, max(0.0, float(os.getenv("GROUP_REPLY_PROBABILITY", "1.0")))
            ),
            proactive_group_enabled=os.getenv("PROACTIVE_GROUP_ENABLED", "true").lower()
            in {"1", "true", "yes", "on"},
            proactive_group_chance=min(
                1.0, max(0.0, float(os.getenv("PROACTIVE_GROUP_CHANCE", "0.02")))
            ),
            proactive_group_cooldown_minutes=max(
                1, int(os.getenv("PROACTIVE_GROUP_COOLDOWN_MINUTES", "60"))
            ),
            proactive_group_min_messages=max(
                1, int(os.getenv("PROACTIVE_GROUP_MIN_MESSAGES", "5"))
            ),
            max_history_messages=max(2, int(os.getenv("MAX_HISTORY_MESSAGES", "64"))),
        )