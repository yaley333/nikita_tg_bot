import asyncio
from collections import defaultdict


class LatestMessageCoordinator:
    def __init__(self) -> None:
        self._latest_message_ids: dict[int, int] = {}
        self._locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    def observe(self, chat_id: int, message_id: int) -> None:
        self._latest_message_ids[chat_id] = max(
            message_id,
            self._latest_message_ids.get(chat_id, message_id),
        )

    def is_latest(self, chat_id: int, message_id: int) -> bool:
        return self._latest_message_ids.get(chat_id) == message_id

    def should_process(
        self,
        chat_id: int,
        message_id: int,
        *,
        preserve: bool = False,
    ) -> bool:
        return preserve or self.is_latest(chat_id, message_id)

    def lock_for(self, chat_id: int) -> asyncio.Lock:
        return self._locks[chat_id]
