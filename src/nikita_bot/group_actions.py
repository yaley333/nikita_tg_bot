import logging
import random
import re
from io import BytesIO
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger(__name__)

STICKER_PACKS = {
    "beer_cat": "catsunicmass",
    "smoking": "Nakroshkua_by_fStikBot",
}

_REACTIONS = (
    (re.compile(r"спасибо|благодар|люблю|обожаю|мил(о|ый|ая)|❤|❤️", re.IGNORECASE), "❤"),
    (re.compile(r"круто|класс|супер|ура|получилось|успех|огонь|wow|🔥", re.IGNORECASE), "⚡"),
    (re.compile(r"шутк|смешно|лол|хаха|ахаха|прикол|абсурд|банан|😂", re.IGNORECASE), "🍌"),
)
_SMOKING_WORDS = re.compile(
    r"кур(ю|ишь|ит|ить|ящ|ение)|сигарет|сигар|вейп|smok|cigarette|vape|🚬",
    re.IGNORECASE,
)
_BEER_WORDS = re.compile(r"пив|пиво|пивк|beer|🍺|🍻", re.IGNORECASE)
_STICKER_REQUEST = re.compile(r"стикер|стикеры|наклейк", re.IGNORECASE)


@dataclass(frozen=True)
class StickerItem:
    pack: str
    emoji: str
    bot_file_id: str | None = None
    telethon_document: object | None = None


class StickerLibrary:
    def __init__(self) -> None:
        self._items: dict[str, list[StickerItem]] = {key: [] for key in STICKER_PACKS}

    async def load_bot_api(self, bot: object) -> None:
        for pack_key, short_name in STICKER_PACKS.items():
            try:
                sticker_set = await bot.get_sticker_set(short_name)
                self._items[pack_key] = [
                    StickerItem(pack_key, sticker.emoji or "", bot_file_id=sticker.file_id)
                    for sticker in sticker_set.stickers
                ]
                logger.info("Загружен набор стикеров %s: %s", short_name, len(self._items[pack_key]))
            except Exception:
                logger.exception("Не удалось загрузить набор стикеров %s", short_name)

    async def load_telethon(self, client: object) -> None:
        from telethon import functions, types

        for pack_key, short_name in STICKER_PACKS.items():
            try:
                sticker_set = await client(
                    functions.messages.GetStickerSetRequest(
                        stickerset=types.InputStickerSetShortName(short_name),
                        hash=0,
                    )
                )
                pack_emojis = {
                    document_id: pack.emoticon
                    for pack in sticker_set.packs
                    for document_id in pack.documents
                }
                items: list[StickerItem] = []
                for document in sticker_set.documents:
                    sticker_attribute = next(
                        (
                            attribute
                            for attribute in document.attributes
                            if isinstance(attribute, types.DocumentAttributeSticker)
                        ),
                        None,
                    )
                    emoji = getattr(sticker_attribute, "alt", "") or pack_emojis.get(
                        document.id, ""
                    )
                    items.append(
                        StickerItem(pack_key, emoji, telethon_document=document)
                    )
                self._items[pack_key] = items
                logger.info("Загружен набор стикеров %s: %s", short_name, len(items))
            except Exception:
                logger.exception("Не удалось загрузить набор стикеров %s", short_name)

    def choose(self, pack_key: str) -> StickerItem | None:
        items = self._items.get(pack_key, [])
        if not items:
            return None

        preferred_emoji = {
            "beer_cat": ("🍺", "🍻"),
            "smoking": ("🚬", "💨"),
        }.get(pack_key, ())
        matching = [
            item for item in items if any(emoji in item.emoji for emoji in preferred_emoji)
        ]
        return random.choice(matching or items)


def sticker_text_hint(emoji: str | None) -> str:
    return f"Собеседник прислал стикер {emoji}." if emoji else "Собеседник прислал стикер."


def normalize_sticker_image(data: bytes) -> bytes:
    with Image.open(BytesIO(data)) as image:
        image.seek(0)
        normalized = image.convert("RGB")
        output = BytesIO()
        normalized.save(output, format="JPEG", quality=90)
        return output.getvalue()


async def download_bot_sticker_preview(bot: object, sticker: object) -> tuple[bytes | None, str]:
    emoji = getattr(sticker, "emoji", None)
    thumbnail = getattr(sticker, "thumbnail", None)
    file_id = getattr(thumbnail, "file_id", None) or getattr(sticker, "file_id", None)
    if not file_id:
        return None, sticker_text_hint(emoji)

    output = BytesIO()
    await bot.download(file_id, destination=output)
    data = output.getvalue()
    if not data:
        return None, sticker_text_hint(emoji)
    try:
        data = normalize_sticker_image(data)
    except Exception:
        logger.debug("Не удалось декодировать sticker preview", exc_info=True)
        return None, sticker_text_hint(emoji)
    return data, sticker_text_hint(emoji)


async def download_telethon_sticker_preview(
    event: object,
) -> tuple[bytes | None, str]:
    from telethon import types

    sticker = event.message.sticker
    if not sticker:
        return None, ""
    sticker_attribute = next(
        (
            attribute
            for attribute in sticker.attributes
            if isinstance(attribute, types.DocumentAttributeSticker)
        ),
        None,
    )
    emoji = getattr(sticker_attribute, "alt", None)
    output = BytesIO()
    try:
        result = await event.message.download_media(file=output, thumb=-1)
        if not result or not output.getvalue():
            if not (sticker.mime_type or "").startswith("image/"):
                return None, sticker_text_hint(emoji)
            output = BytesIO()
            result = await event.message.download_media(file=output)
        data = output.getvalue()
        if not result or not data:
            return None, sticker_text_hint(emoji)
        return normalize_sticker_image(data), sticker_text_hint(emoji)
    except Exception:
        logger.debug("Не удалось скачать sticker preview", exc_info=True)
        return None, sticker_text_hint(emoji)


def choose_reaction(text: str) -> str | None:
    for pattern, emoji in _REACTIONS:
        if pattern.search(text):
            return emoji
    return None


def choose_sticker_pack(text: str) -> str | None:
    if _SMOKING_WORDS.search(text):
        return "smoking"
    if _BEER_WORDS.search(text):
        return "beer_cat"
    if _STICKER_REQUEST.search(text):
        return random.choice(tuple(STICKER_PACKS))
    return None


async def react_in_bot_api(bot: object, chat_id: int, message_id: int, text: str) -> None:
    emoji = choose_reaction(text)
    if not emoji:
        return
    try:
        from aiogram.types import ReactionTypeEmoji

        await bot.set_message_reaction(
            chat_id,
            message_id,
            reaction=[ReactionTypeEmoji(emoji=emoji)],
        )
    except Exception:
        logger.debug("Telegram не разрешил поставить реакцию", exc_info=True)


async def react_in_telethon(client: object, event: object, text: str) -> None:
    emoji = choose_reaction(text)
    if not emoji:
        return
    try:
        from telethon import functions, types

        await client(
            functions.messages.SendReactionRequest(
                peer=event.input_chat,
                msg_id=event.message.id,
                reaction=[types.ReactionEmoji(emoticon=emoji)],
            )
        )
    except Exception:
        logger.debug("Telegram не разрешил поставить реакцию", exc_info=True)