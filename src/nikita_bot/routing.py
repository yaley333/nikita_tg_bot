import re
from collections import defaultdict


_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_ADDRESS_BOUNDARY_PATTERN = re.compile(r"[\s,!?;:—-]")


class KnownBotNames:
    def __init__(self) -> None:
        self._names: defaultdict[int, set[str]] = defaultdict(set)

    def observe(self, chat_id: int, sender: object) -> None:
        if not bool(getattr(sender, "bot", getattr(sender, "is_bot", False))):
            return

        for attribute in ("first_name", "username"):
            name = getattr(sender, attribute, None)
            if isinstance(name, str) and name.strip():
                normalized = self._normalize(name)
                self._names[chat_id].add(normalized)
                if attribute == "first_name":
                    self._names[chat_id].add(normalized.split()[0])

    def addresses_other_bot(self, chat_id: int, text: str) -> bool:
        normalized_text = self._normalize(text)
        for name in self._names.get(chat_id, set()):
            if not normalized_text.startswith(name):
                continue
            if len(normalized_text) == len(name) or _ADDRESS_BOUNDARY_PATTERN.match(
                normalized_text[len(name) :]
            ):
                return True
        return False

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(_WORD_PATTERN.findall(value.casefold().replace("ё", "е")))
