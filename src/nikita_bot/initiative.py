import random
import re
import time
from dataclasses import dataclass


@dataclass
class _GroupActivity:
    human_messages: int = 0
    last_proactive_at: float = 0.0


class GroupInitiative:
    def __init__(
        self,
        min_messages: int = 5,
        chance: float = 0.02,
        cooldown_seconds: float = 3600,
    ) -> None:
        self.min_messages = max(1, min_messages)
        self.chance = min(1.0, max(0.0, chance))
        self.cooldown_seconds = max(1.0, cooldown_seconds)
        self._activity: dict[int, _GroupActivity] = {}

    def should_initiate(
        self,
        chat_id: int,
        *,
        sender_is_bot: bool = False,
        now: float | None = None,
        random_value: float | None = None,
    ) -> bool:
        if sender_is_bot:
            return False

        activity = self._activity.setdefault(chat_id, _GroupActivity())
        activity.human_messages += 1
        current_time = time.monotonic() if now is None else now
        if activity.human_messages < self.min_messages:
            return False
        if current_time - activity.last_proactive_at < self.cooldown_seconds:
            return False

        if (random.random() if random_value is None else random_value) >= self.chance:
            return False

        activity.last_proactive_at = current_time
        activity.human_messages = 0
        return True


_PROACTIVE_STOP_WORDS = {
    "это", "что", "как", "для", "или", "его", "она", "они", "тебе", "тебя",
    "мне", "меня", "сами", "сам", "тут", "там", "уже", "просто", "тогда",
    "когда", "почему", "давай", "может", "вот", "такой", "такая", "того",
    "which", "what", "that", "this", "with", "from", "your", "about",
}
_PROACTIVE_WORD_PATTERN = re.compile(r"[a-zа-яё]{3,}", re.IGNORECASE)


def proactive_reply_is_relevant(reply: str, context: str, max_words: int = 32) -> bool:
    reply_words = reply.split()
    if not reply_words or len(reply_words) > max_words:
        return False
    if re.search(r"не ври|объясни|что тут смешного|как дела|кто тут главный", reply, re.I):
        return False

    reply_terms = {
        word.casefold()
        for word in _PROACTIVE_WORD_PATTERN.findall(reply)
        if word.casefold() not in _PROACTIVE_STOP_WORDS
    }
    context_terms = {
        word.casefold()
        for word in _PROACTIVE_WORD_PATTERN.findall(context)
        if word.casefold() not in _PROACTIVE_STOP_WORDS
    }
    return bool(reply_terms & context_terms)