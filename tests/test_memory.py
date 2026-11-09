import os
from io import BytesIO
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image
from telethon.errors import ChatWriteForbiddenError
from telethon import utils as telethon_utils

from nikita_bot.config import Settings
from nikita_bot.group_actions import (
    StickerItem,
    StickerLibrary,
    choose_reaction,
    choose_sticker_pack,
    normalize_sticker_image,
    sticker_text_hint,
)
from nikita_bot.initiative import GroupInitiative, proactive_reply_is_relevant
from nikita_bot.main import (
    format_sender_context,
    parse_image_request,
    parse_voice_command,
    run_with_telethon_action,
    should_respond_to_group_message,
    telethon_image_file,
    telethon_voice_file,
    wants_voice_reply,
)
from nikita_bot.memory import ConversationMemory
from nikita_bot.model import (
    IMAGE_PROMPT_INSTRUCTION,
    PROACTIVE_INSTRUCTION,
    VOICE_REPLY_INSTRUCTION,
    build_messages,
    get_num_predict,
)
from nikita_bot.persona import SYSTEM_PROMPT
from nikita_bot.speech import clean_text_for_speech


class ConversationMemoryTests(unittest.TestCase):
    def test_bot_mode_does_not_need_telethon_credentials(self) -> None:
        with patch("nikita_bot.config.load_dotenv"), patch.dict(
            os.environ, {"TELEGRAM_MODE": "bot"}, clear=True
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.telegram_mode, "bot")
        self.assertIsNone(settings.telegram_bot_token)
        self.assertIsNone(settings.api_id)
        self.assertIsNone(settings.api_hash)
        self.assertEqual(settings.group_reply_probability, 1.0)
        self.assertEqual(settings.session_path, ".sessions/nikita_user")
        self.assertEqual(
            settings.piper_voice_path,
            ".models/piper/ru_RU-ruslan-medium.onnx",
        )
        self.assertEqual(
            settings.sd_turbo_model_path,
            ".models/sd-turbo/sd_turbo-f16-q8_0.gguf",
        )

    def test_user_mode_can_respond_to_bots_without_storing_credentials(self) -> None:
        with patch("nikita_bot.config.load_dotenv"), patch.dict(
            os.environ,
            {"TELEGRAM_MODE": "user", "ALLOW_BOT_SENDERS": "true"},
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.telegram_mode, "user")
        self.assertTrue(settings.allow_bot_senders)

    def test_bot_token_is_loaded_from_environment(self) -> None:
        with patch("nikita_bot.config.load_dotenv"), patch.dict(
            os.environ,
            {"TELEGRAM_MODE": "bot", "TELEGRAM_BOT_TOKEN": "example-token"},
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.telegram_bot_token, "example-token")

    def test_blank_bot_token_is_treated_as_unset(self) -> None:
        with patch("nikita_bot.config.load_dotenv"), patch.dict(
            os.environ,
            {"TELEGRAM_MODE": "bot", "TELEGRAM_BOT_TOKEN": "  "},
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertIsNone(settings.telegram_bot_token)

    def test_user_api_credentials_are_loaded_from_environment(self) -> None:
        with patch("nikita_bot.config.load_dotenv"), patch.dict(
            os.environ,
            {"TELEGRAM_MODE": "user", "API_ID": "12345", "API_HASH": "example-hash"},
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.api_id, "12345")
        self.assertEqual(settings.api_hash, "example-hash")

    def test_blank_api_credentials_are_treated_as_unset(self) -> None:
        with patch("nikita_bot.config.load_dotenv"), patch.dict(
            os.environ,
            {"TELEGRAM_MODE": "user", "API_ID": "", "API_HASH": "  "},
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertIsNone(settings.api_id)
        self.assertIsNone(settings.api_hash)

    def test_group_context_includes_sender_name_and_bot_role(self) -> None:
        sender = SimpleNamespace(first_name="Алиса", username="alice_bot", bot=True)
        self.assertEqual(
            format_sender_context(sender, "Привет"),
            "[Алиса (@alice_bot), бот]: Привет",
        )

    def test_group_bot_messages_are_always_answered_when_enabled(self) -> None:
        self.assertTrue(should_respond_to_group_message(True, True, 0.7, 0.69))
        self.assertFalse(should_respond_to_group_message(True, True, 0.7, 0.71))
        self.assertFalse(should_respond_to_group_message(True, False, 0.7, 0.1))
        self.assertTrue(should_respond_to_group_message(False, False, 0.7, 0.1))
        self.assertTrue(should_respond_to_group_message(False, False, 1.0, 0.999))

    def test_group_reply_probability_can_be_set_explicitly(self) -> None:
        with patch("nikita_bot.config.load_dotenv"), patch.dict(
            os.environ,
            {"TELEGRAM_MODE": "user", "GROUP_REPLY_PROBABILITY": "0.4"},
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.group_reply_probability, 0.4)

    def test_group_initiative_waits_for_activity_and_respects_cooldown(self) -> None:
        initiative = GroupInitiative(min_messages=2, chance=0.5, cooldown_seconds=60)
        self.assertFalse(initiative.should_initiate(10, now=100, random_value=0.0))
        self.assertTrue(initiative.should_initiate(10, now=101, random_value=0.0))
        self.assertFalse(initiative.should_initiate(10, now=102, random_value=0.0))
        self.assertFalse(
            initiative.should_initiate(10, sender_is_bot=True, now=200, random_value=0.0)
        )
        self.assertTrue(initiative.should_initiate(10, now=200, random_value=0.0))

    def test_proactive_reply_must_be_short_and_contextual(self) -> None:
        context = "[Алиса]: кот уронил стакан со стола"
        self.assertTrue(proactive_reply_is_relevant("Кот снова проверяет гравитацию.", context))
        self.assertFalse(proactive_reply_is_relevant("Ну давай объясни, что тут смешного.", context))
        self.assertFalse(proactive_reply_is_relevant("очень длинный ответ " * 20, context))

    def test_proactive_model_prompt_is_short_and_contextual(self) -> None:
        messages = build_messages(
            [{"role": "user", "content": "Собеседник обсуждает аниме"}],
            proactive=True,
        )
        self.assertIn(PROACTIVE_INSTRUCTION, messages[0]["content"])

    def test_group_actions_match_reactions_and_sticker_packs(self) -> None:
        self.assertEqual(choose_reaction("Спасибо, это круто!"), "❤")
        self.assertEqual(choose_reaction("Ахаха, очень смешно"), "🍌")
        self.assertEqual(choose_sticker_pack("кот с пивом"), "beer_cat")
        self.assertEqual(choose_sticker_pack("покажи курящий стикер"), "smoking")

    def test_sticker_library_prefers_beer_cat_emoji(self) -> None:
        library = StickerLibrary()
        library._items["beer_cat"] = [
            StickerItem("beer_cat", "🐱", bot_file_id="cat"),
            StickerItem("beer_cat", "🍺", bot_file_id="cat_beer"),
        ]

        self.assertEqual(library.choose("beer_cat").bot_file_id, "cat_beer")

    def test_sticker_preview_is_converted_to_jpeg(self) -> None:
        source = BytesIO()
        Image.new("RGB", (12, 12), color="red").save(source, format="PNG")

        normalized = normalize_sticker_image(source.getvalue())
        with Image.open(BytesIO(normalized)) as preview:
            self.assertEqual(preview.format, "JPEG")
            self.assertEqual(preview.size, (12, 12))
        self.assertEqual(sticker_text_hint("🚬"), "Собеседник прислал стикер 🚬.")

    def test_voice_command_extracts_text(self) -> None:
        self.assertEqual(parse_voice_command("/voice@NikitaBot привет"), (True, "привет"))
        self.assertEqual(parse_voice_command("просто текст"), (False, "просто текст"))
        self.assertEqual(parse_voice_command(""), (False, ""))

    def test_voice_request_is_detected_in_normal_text(self) -> None:
        self.assertTrue(wants_voice_reply("Ответь голосом: расскажи шутку"))
        self.assertTrue(wants_voice_reply("Пришли голосовое сообщение"))
        self.assertTrue(wants_voice_reply("гс запиши, пожалуйста"))
        self.assertFalse(wants_voice_reply("Я слушаю голосовые сообщения"))

    def test_image_request_extracts_prompt(self) -> None:
        self.assertEqual(parse_image_request("Нарисуй кота в космосе"), (True, "кота в космосе"))
        self.assertEqual(
            parse_image_request("Некит, нарисуй изображение кота"),
            (True, "кота"),
        )
        self.assertEqual(
            parse_image_request("Никита: нарисуй кота"),
            (True, "кота"),
        )
        self.assertEqual(
            parse_image_request("Сгенерируй изображение: зимний лес"),
            (True, "зимний лес"),
        )
        self.assertEqual(parse_image_request("Расскажи про котов"), (False, "Расскажи про котов"))

    def test_telethon_image_file_is_recognized_as_a_photo(self) -> None:
        image_file = telethon_image_file(b"\x89PNG\r\n\x1a\n")

        self.assertEqual(image_file.name, "nikita.png")
        self.assertTrue(telethon_utils.is_image(image_file))

    def test_telethon_voice_file_is_recognized_as_ogg_voice(self) -> None:
        audio_file = telethon_voice_file(b"OggS" + b"\x00" * 64)
        attributes, mime_type = telethon_utils.get_attributes(
            audio_file,
            voice_note=True,
        )
        audio_attribute = next(
            attribute
            for attribute in attributes
            if type(attribute).__name__ == "DocumentAttributeAudio"
        )

        self.assertEqual(audio_file.name, "nikita.ogg")
        self.assertEqual(mime_type, "audio/ogg")
        self.assertTrue(audio_attribute.voice)

    def test_history_is_limited_to_recent_messages(self) -> None:
        memory = ConversationMemory(max_messages=2)
        memory.add(1, "user", "первое")
        memory.add(1, "assistant", "второе")
        memory.add(1, "user", "третье")

        self.assertEqual(
            memory.history(1),
            [
                {"role": "assistant", "content": "второе"},
                {"role": "user", "content": "третье"},
            ],
        )

    def test_persona_prompt_is_present(self) -> None:
        self.assertTrue(SYSTEM_PROMPT.strip())

    def test_voice_request_instructions_answer_directly(self) -> None:
        self.assertIn("сразу формулируй ответ по теме запроса", SYSTEM_PROMPT)
        self.assertIn("Не путай просьбу ответить голосом с просьбой прослушать", SYSTEM_PROMPT)

    def test_voice_mode_adds_direct_instruction_to_model(self) -> None:
        messages = build_messages([{"role": "user", "content": "расскажи о Японии"}], True)
        self.assertIn(VOICE_REPLY_INSTRUCTION, messages[0]["content"])
        self.assertEqual(messages[1]["content"], "расскажи о Японии")

    def test_persona_encourages_casual_context_aware_replies(self) -> None:
        self.assertIn("не ставь точку в конце короткого сообщения", SYSTEM_PROMPT)
        self.assertIn("не выдумывай, кто что просил", SYSTEM_PROMPT)
        self.assertIn("коротко признай", SYSTEM_PROMPT)

    def test_small_talk_uses_short_reply_budget(self) -> None:
        self.assertEqual(get_num_predict([{"role": "user", "content": "Как дела?"}]), 64)
        self.assertEqual(
            get_num_predict([{"role": "user", "content": "Расскажи факты про Японию"}]),
            512,
        )
        self.assertIn("одна естественная фраза", SYSTEM_PROMPT)

    def test_image_prompt_mode_translates_to_english_prompt(self) -> None:
        messages = build_messages([{"role": "user", "content": "кот в космосе"}], image_prompt=True)
        self.assertEqual(messages[0]["content"], IMAGE_PROMPT_INSTRUCTION)

    def test_speech_text_removes_markdown(self) -> None:
        self.assertEqual(
            clean_text_for_speech("1. **Факт** — *текст* и [ссылка](https://example.com)"),
            "Факт — текст и ссылка",
        )


class TelethonActionTests(unittest.IsolatedAsyncioTestCase):
    async def test_action_exit_error_does_not_discard_completed_operation(self) -> None:
        action_context = AsyncMock()
        action_context.__aexit__.side_effect = ChatWriteForbiddenError(None)
        client = Mock()
        client.action.return_value = action_context
        operation = AsyncMock(return_value="Ответ")

        result = await run_with_telethon_action(client, 123, "typing", operation)

        self.assertEqual(result, "Ответ")
        operation.assert_awaited_once()

    async def test_action_entry_error_retries_operation_without_action(self) -> None:
        action_context = AsyncMock()
        action_context.__aenter__.side_effect = ChatWriteForbiddenError(None)
        client = Mock()
        client.action.return_value = action_context
        operation = AsyncMock(return_value=b"image")

        result = await run_with_telethon_action(client, 123, "photo", operation)

        self.assertEqual(result, b"image")
        operation.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()