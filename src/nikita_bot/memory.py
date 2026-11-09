from collections import defaultdict


class ConversationMemory:
    def __init__(self, max_messages: int = 16) -> None:
        self.max_messages = max_messages
        self._conversations: dict[int, list[dict[str, str]]] = defaultdict(list)

    def history(self, chat_id: int) -> list[dict[str, str]]:
        return list(self._conversations[chat_id])

    def add(self, chat_id: int, role: str, content: str) -> None:
        messages = self._conversations[chat_id]
        messages.append({"role": role, "content": content})
        del messages[:-self.max_messages]