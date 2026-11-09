import asyncio
import getpass
import io
import logging
import random
import re
from pathlib import Path

import qrcode
from aiogram import Bot, Dispatcher
from aiogram.enums import ChatAction
from aiogram.types import BufferedInputFile, Message
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError

from .config import Settings
from .group_actions import (
    StickerLibrary,
    choose_sticker_pack,
    download_bot_sticker_preview,
    download_telethon_sticker_preview,
    react_in_bot_api,
    react_in_telethon,
)
from .initiative import GroupInitiative, proactive_reply_is_relevant
from .imagegen import StableDiffusionCpp
from .memory import ConversationMemory
from .model import OllamaClient
from .speech import PiperSpeaker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
VOICE_REQUEST_PATTERN = re.compile(
    r"\b(?:ответь|скажи|пришли|отправь|скинь|запиши|озвучь|прочитай)\b"
    r".{0,40}\b(?:голосом|голосовое(?:\s+сообщение)?|голосовым(?:\s+сообщением)?|гс)\b"
    r"|\b(?:голосом|голосовое(?:\s+сообщение)?|голосовым(?:\s+сообщением)?|гс)\b"
    r".{0,20}\b(?:ответь|скажи|пришли|отправь|скинь|запиши|озвучь)\b",
    re.IGNORECASE,
)


async def main() -> None:
    settings = Settings.from_env()
    model = OllamaClient(settings.ollama_url, settings.ollama_model)
    speaker = PiperSpeaker(settings.piper_voice_path)
    image_generator = StableDiffusionCpp(settings.sd_cpp_path, settings.sd_turbo_model_path)
    sticker_library = StickerLibrary()
    initiative = GroupInitiative(
        min_messages=settings.proactive_group_min_messages,
        chance=settings.proactive_group_chance,
        cooldown_seconds=settings.proactive_group_cooldown_minutes * 60,
    )
    memory = ConversationMemory(settings.max_history_messages)

    try:
        if settings.telegram_mode == "bot":
            await run_bot_api(
                settings,
                model,
                speaker,
                image_generator,
                sticker_library,
                initiative,
                memory,
            )
        else:
            await run_user_session(
                settings,
                model,
                speaker,
                image_generator,
                sticker_library,
                initiative,
                memory,
            )
    finally:
        await model.close()


async def run_bot_api(
    settings: Settings,
    model: OllamaClient,
    speaker: PiperSpeaker,
    image_generator: StableDiffusionCpp,
    sticker_library: StickerLibrary,
    initiative: GroupInitiative,
    memory: ConversationMemory,
) -> None:
    token = settings.telegram_bot_token or getpass.getpass(
        "Токен бота от @BotFather (ввод скрыт): "
    ).strip()
    if not token:
        raise ValueError("Токен бота не может быть пустым")

    bot = Bot(token)
    dispatcher = Dispatcher()
    bot_user = await bot.get_me()
    await sticker_library.load_bot_api(bot)
    logger.info("Nikita запущен через Bot API (id=%s)", bot_user.id)

    @dispatcher.message()
    async def handle_bot_message(message: Message) -> None:
        if not should_respond_to_bot_api(message, bot_user.id, bot_user.username):
            return

        text = (message.text or message.caption or "").strip()
        is_group = message.chat.type in {"group", "supergroup"}
        sender = message.from_user
        sticker_image = None
        if message.sticker:
            sticker_image, sticker_hint = await download_bot_sticker_preview(
                bot,
                message.sticker,
            )
            text = text or sticker_hint
        if is_group and sender and sender.is_bot and not settings.allow_bot_senders:
            return
        sender_is_bot = bool(sender and sender.is_bot)
        if (
            is_group
            and settings.proactive_group_enabled
            and not sticker_image
            and initiative.should_initiate(message.chat.id, sender_is_bot=sender_is_bot)
        ):
            try:
                proactive = await create_proactive_group_output(
                    message.chat.id,
                    sender,
                    text,
                    model,
                    memory,
                    speaker,
                    image_generator,
                )
                if proactive:
                    await send_bot_proactive(bot, message.chat.id, proactive)
                    await apply_bot_group_actions(bot, sticker_library, message, text)
                    return
            except Exception:
                logger.exception("Не удалось отправить инициативное сообщение")
        if is_group and not should_respond_to_group_message(
            sender_is_bot,
            settings.allow_bot_senders,
            settings.group_reply_probability,
            random.random(),
        ):
            memory.add(
                message.chat.id,
                "user",
                format_sender_context(sender, text or "[изображение]"),
            )
            await apply_bot_group_actions(bot, sticker_library, message, text)
            return

        image_requested, image_prompt = parse_image_request(text)
        if image_requested:
            if not image_prompt:
                await message.answer("Опиши, что именно нарисовать.")
                return
            try:
                await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
                english_prompt = await model.generate(
                    [{"role": "user", "content": image_prompt}],
                    image_prompt=True,
                )
                await model.unload()
                await bot.send_chat_action(message.chat.id, ChatAction.UPLOAD_PHOTO)
                image_bytes = await asyncio.to_thread(image_generator.generate, english_prompt)
                memory.add(
                    message.chat.id,
                    "user",
                    format_sender_context(sender, text) if is_group else text,
                )
                memory.add(message.chat.id, "assistant", "Создал изображение.")
                await bot.send_photo(
                    message.chat.id,
                    BufferedInputFile(image_bytes, filename="nikita.png"),
                    caption="Готово.",
                )
                if is_group:
                    await apply_bot_group_actions(bot, sticker_library, message, text)
            except Exception:
                logger.exception("Не удалось сгенерировать изображение")
                await message.answer("Не получилось создать изображение. Подробности смотри в PowerShell.")
            return

        voice_requested, text = parse_voice_command(text)
        voice_requested = voice_requested or wants_voice_reply(text)
        if voice_requested and not text:
            await message.answer("Напиши текст после команды: /voice что-нибудь расскажи")
            return
        image = None
        if message.photo:
            text = text or "Проанализируй это изображение."
            image_stream = io.BytesIO()
            await bot.download(message.photo[-1].file_id, destination=image_stream)
            image = image_stream.getvalue()
        elif sticker_image:
            image = sticker_image
        elif not text:
            return

        await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        answer = await generate_answer(
            message.chat.id,
            format_sender_context(sender, text) if is_group else text,
            image,
            model,
            memory,
            voice_reply=voice_requested,
        )
        if voice_requested:
            try:
                await bot.send_chat_action(message.chat.id, ChatAction.RECORD_VOICE)
                audio = await asyncio.to_thread(speaker.synthesize_ogg, answer)
                await bot.send_voice(
                    message.chat.id,
                    BufferedInputFile(audio, filename="nikita.ogg"),
                    caption=answer,
                )
            except Exception:
                logger.exception("Не удалось создать голосовой ответ")
                await message.answer("Не получилось создать голосовой ответ. Проверь Piper.")
        else:
            await message.answer(answer)
        if is_group:
            await apply_bot_group_actions(bot, sticker_library, message, text)

    try:
        await dispatcher.start_polling(bot)
    finally:
        await bot.session.close()


async def run_user_session(
    settings: Settings,
    model: OllamaClient,
    speaker: PiperSpeaker,
    image_generator: StableDiffusionCpp,
    sticker_library: StickerLibrary,
    initiative: GroupInitiative,
    memory: ConversationMemory,
) -> None:
    api_id_text = settings.api_id or input("API_ID: ").strip()
    api_hash = settings.api_hash or getpass.getpass("API_HASH (ввод скрыт): ").strip()
    if not api_id_text.isdecimal() or not api_hash:
        raise ValueError("Для режима user нужны числовой API_ID и API_HASH")

    Path(settings.session_path).parent.mkdir(parents=True, exist_ok=True)
    client = TelegramClient(settings.session_path, int(api_id_text), api_hash)
    try:
        await client.connect()
        if await client.is_user_authorized():
            logger.info("Используется сохранённая авторизация Telethon")
        else:
            try:
                while True:
                    qr_login = await client.qr_login()
                    print(
                        "\nОткрой Telegram на телефоне: Настройки -> Устройства -> "
                        "Подключить устройство, затем отсканируй этот QR-код.\n"
                    )
                    qr_code = qrcode.QRCode(border=2)
                    qr_code.add_data(qr_login.url)
                    qr_code.make(fit=True)
                    qr_code.print_ascii(invert=True)
                    try:
                        await qr_login.wait()
                        break
                    except asyncio.TimeoutError:
                        logger.info("QR-код истёк; создаю новый")
            except SessionPasswordNeededError:
                password = getpass.getpass(
                    "Пароль двухэтапной проверки Telegram (ввод скрыт): "
                )
                await client.sign_in(password=password)

        me = await client.get_me()
        logger.info(
            "Nikita запущен через пользовательскую сессию (id=%s, username=@%s)",
            me.id,
            me.username or "нет",
        )
        await sticker_library.load_telethon(client)

        @client.on(events.NewMessage(incoming=True))
        async def handle_user_message(event: events.NewMessage.Event) -> None:
            sender = await event.get_sender()
            sender_is_bot = bool(getattr(sender, "bot", False))
            chat_id = event.chat_id
            if chat_id is None:
                return

            raw_text = (event.raw_text or "").strip()
            if sender_is_bot and not settings.allow_bot_senders:
                return
            sticker_image = None
            sticker_hint = ""
            if event.message.sticker:
                sticker_image, sticker_hint = await download_telethon_sticker_preview(event)
                raw_text = raw_text or sticker_hint

            if (
                event.is_group
                and settings.proactive_group_enabled
                and not sticker_image
                and initiative.should_initiate(chat_id, sender_is_bot=sender_is_bot)
            ):
                try:
                    proactive = await create_proactive_group_output(
                        chat_id,
                        sender,
                        raw_text,
                        model,
                        memory,
                        speaker,
                        image_generator,
                    )
                    if proactive:
                        await send_telethon_proactive(client, chat_id, proactive)
                        await apply_telethon_group_actions(
                            client,
                            sticker_library,
                            event,
                            raw_text,
                        )
                        return
                except Exception:
                    logger.exception("Не удалось отправить инициативное сообщение")

            if event.is_group:
                if not should_respond_to_group_message(
                    sender_is_bot,
                    settings.allow_bot_senders,
                    settings.group_reply_probability,
                    random.random(),
                ):
                    memory.add(
                        chat_id,
                        "user",
                        format_sender_context(sender, raw_text or "[изображение]"),
                    )
                    await apply_telethon_group_actions(
                        client,
                        sticker_library,
                        event,
                        raw_text,
                    )
                    return
            elif not await should_respond(event):
                return

            text = raw_text
            image_requested, image_prompt = parse_image_request(text)
            if image_requested:
                if not image_prompt:
                    await event.reply("Опиши, что именно нарисовать.", parse_mode=None)
                    return
                chat_id = event.chat_id
                if chat_id is None:
                    return
                try:
                    async with client.action(chat_id, "typing"):
                        english_prompt = await model.generate(
                            [{"role": "user", "content": image_prompt}],
                            image_prompt=True,
                        )
                    await model.unload()
                    async with client.action(chat_id, "upload_photo"):
                        image_bytes = await asyncio.to_thread(
                            image_generator.generate,
                            english_prompt,
                        )
                    memory.add(
                        chat_id,
                        "user",
                        format_sender_context(sender, text) if event.is_group else text,
                    )
                    memory.add(chat_id, "assistant", "Создал изображение.")
                    await client.send_file(
                        chat_id,
                        image_bytes,
                        force_document=False,
                        caption="Готово.",
                        reply_to=event.message.id,
                    )
                    if event.is_group:
                        await apply_telethon_group_actions(
                            client,
                            sticker_library,
                            event,
                            text,
                        )
                except Exception:
                    logger.exception("Не удалось сгенерировать изображение")
                    await event.reply(
                        "Не получилось создать изображение. Подробности смотри в PowerShell.",
                        parse_mode=None,
                    )
                return

            voice_requested, text = parse_voice_command(text)
            voice_requested = voice_requested or wants_voice_reply(text)
            if voice_requested and not text:
                await event.reply("Напиши текст после команды: /voice что-нибудь расскажи")
                return

            image = None
            if event.message.photo:
                text = text or "Проанализируй это изображение."
                image_buffer = io.BytesIO()
                await event.message.download_media(file=image_buffer)
                image = image_buffer.getvalue()
            elif sticker_image:
                image = sticker_image
            elif not text:
                return

            async with client.action(chat_id, "typing"):
                answer = await generate_answer(
                    chat_id,
                    format_sender_context(sender, text)
                    if event.is_group or sender_is_bot
                    else text,
                    image,
                    model,
                    memory,
                    voice_reply=voice_requested,
                )
            if voice_requested:
                try:
                    audio = await asyncio.to_thread(speaker.synthesize_ogg, answer)
                    await client.send_file(
                        chat_id,
                        audio,
                        voice_note=True,
                        caption=answer,
                        reply_to=event.message.id,
                    )
                except Exception:
                    logger.exception("Не удалось создать голосовой ответ")
                    await event.reply(
                        "Не получилось создать голосовой ответ. Проверь Piper.",
                        parse_mode=None,
                    )
            else:
                await event.reply(answer, parse_mode=None)
            if event.is_group:
                await apply_telethon_group_actions(
                    client,
                    sticker_library,
                    event,
                    raw_text,
                )

        await client.run_until_disconnected()
    finally:
        await client.disconnect()


def should_respond_to_bot_api(message: Message, bot_id: int, bot_username: str | None) -> bool:
    if message.chat.type == "private":
        return True
    if message.chat.type not in {"group", "supergroup"}:
        return False
    return True


def should_respond_to_group_message(
    sender_is_bot: bool,
    allow_bot_senders: bool,
    reply_probability: float,
    random_value: float,
) -> bool:
    if sender_is_bot and not allow_bot_senders:
        return False
    return random_value < reply_probability


async def apply_bot_group_actions(
    bot: Bot,
    sticker_library: StickerLibrary,
    message: Message,
    text: str,
) -> None:
    await react_in_bot_api(bot, message.chat.id, message.message_id, text)
    sticker_pack = choose_sticker_pack(text)
    sticker = sticker_library.choose(sticker_pack) if sticker_pack else None
    if not sticker or not sticker.bot_file_id:
        return
    try:
        await bot.send_sticker(
            message.chat.id,
            sticker.bot_file_id,
            reply_to_message_id=message.message_id,
        )
    except Exception:
        logger.exception("Не удалось отправить стикер из набора %s", sticker_pack)


async def apply_telethon_group_actions(
    client: TelegramClient,
    sticker_library: StickerLibrary,
    event: events.NewMessage.Event,
    text: str,
) -> None:
    await react_in_telethon(client, event, text)
    sticker_pack = choose_sticker_pack(text)
    sticker = sticker_library.choose(sticker_pack) if sticker_pack else None
    if not sticker or not sticker.telethon_document:
        return
    try:
        await client.send_file(
            event.chat_id,
            sticker.telethon_document,
            force_document=False,
            reply_to=event.message.id,
        )
    except Exception:
        logger.exception("Не удалось отправить стикер из набора %s", sticker_pack)


async def create_proactive_group_output(
    chat_id: int,
    sender: object,
    text: str,
    model: OllamaClient,
    memory: ConversationMemory,
    speaker: PiperSpeaker,
    image_generator: StableDiffusionCpp,
) -> tuple[str, bytes | str, str] | None:
    user_context = format_sender_context(sender, text or "[сообщение в группе]")
    history = [*memory.history(chat_id), {"role": "user", "content": user_context}]
    media_choice = random.random()

    if media_choice < 0.05:
        prompt = await model.generate(history, image_prompt=True, proactive=True)
        if prompt.strip() == "[SKIP]":
            return None
        await model.unload()
        image = await asyncio.to_thread(image_generator.generate, prompt)
        memory.add(chat_id, "user", user_context)
        memory.add(chat_id, "assistant", "[самостоятельно отправил картинку]")
        return "image", image, ""

    voice = media_choice < 0.20
    answer = await model.generate(history, voice_reply=voice, proactive=True)
    recent_context = " ".join(message["content"] for message in history[-8:])
    if answer.strip() == "[SKIP]" or not proactive_reply_is_relevant(answer, recent_context):
        return None
    memory.add(chat_id, "user", user_context)
    memory.add(chat_id, "assistant", answer)
    if voice:
        audio = await asyncio.to_thread(speaker.synthesize_ogg, answer)
        return "voice", audio, answer
    return "text", answer, answer


async def send_bot_proactive(
    bot: Bot,
    chat_id: int,
    proactive: tuple[str, bytes | str, str],
) -> None:
    kind, content, caption = proactive
    if kind == "image":
        await bot.send_photo(chat_id, BufferedInputFile(content, filename="nikita.png"))
    elif kind == "voice":
        await bot.send_voice(
            chat_id,
            BufferedInputFile(content, filename="nikita.ogg"),
            caption=caption,
        )
    else:
        await bot.send_message(chat_id, content)


async def send_telethon_proactive(
    client: TelegramClient,
    chat_id: int,
    proactive: tuple[str, bytes | str, str],
) -> None:
    kind, content, caption = proactive
    if kind == "image":
        await client.send_file(chat_id, content)
    elif kind == "voice":
        await client.send_file(chat_id, content, voice_note=True, caption=caption)
    else:
        await client.send_message(chat_id, content)


async def generate_answer(
    chat_id: int,
    text: str,
    image: bytes | None,
    model: OllamaClient,
    memory: ConversationMemory,
    voice_reply: bool = False,
) -> str:
    memory.add(chat_id, "user", text)
    try:
        answer = await model.generate(memory.history(chat_id), image, voice_reply)
    except Exception:
        logger.exception("Не удалось обработать сообщение")
        answer = "Не удалось получить ответ от модели. Подробности ошибки смотри в окне PowerShell."
    memory.add(chat_id, "assistant", answer)
    return answer


def parse_voice_command(text: str) -> tuple[bool, str]:
    parts = text.split(maxsplit=1)
    if not parts or parts[0].split("@", 1)[0].casefold() != "/voice":
        return False, text
    return True, parts[1].strip() if len(parts) > 1 else ""


def format_sender_context(sender: object, text: str) -> str:
    first_name = getattr(sender, "first_name", None)
    last_name = getattr(sender, "last_name", None)
    username = getattr(sender, "username", None)
    is_bot = bool(getattr(sender, "bot", getattr(sender, "is_bot", False)))
    display_name = " ".join(part for part in (first_name, last_name) if part)
    if username and username.casefold() not in display_name.casefold():
        display_name = f"{display_name} (@{username})" if display_name else f"@{username}"
    if not display_name:
        display_name = "Собеседник"
    role = "бот" if is_bot else "участник"
    return f"[{display_name}, {role}]: {text}"


def wants_voice_reply(text: str) -> bool:
    return parse_voice_command(text)[0] or bool(VOICE_REQUEST_PATTERN.search(text))


IMAGE_REQUEST_PATTERN = re.compile(
    r"^\s*(?:нарисуй|сгенерируй|создай|сделай)\s+"
    r"(?:(?:мне|пожалуйста)\s+)*"
    r"(?:(?:картинку|изображение|иллюстрацию|арт|фото)\b\s*)?"
    r"(.*)$",
    re.IGNORECASE | re.DOTALL,
)


def parse_image_request(text: str) -> tuple[bool, str]:
    match = IMAGE_REQUEST_PATTERN.match(text)
    if not match:
        return False, text
    prompt = match.group(1).strip().lstrip(":,;.!? ")
    return True, prompt


async def should_respond(event: events.NewMessage.Event) -> bool:
    if event.is_private:
        return True
    if not event.is_group:
        return False
    image_requested, _ = parse_image_request(event.raw_text or "")
    if image_requested:
        return True
    if wants_voice_reply(event.raw_text or ""):
        return True
    if event.message.mentioned:
        return True
    if event.is_reply:
        replied = await event.get_reply_message()
        return bool(replied and replied.out)
    return False


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (ValueError, KeyboardInterrupt) as exc:
        if isinstance(exc, ValueError):
            raise SystemExit(str(exc)) from exc