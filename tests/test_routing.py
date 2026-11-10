import unittest
from types import SimpleNamespace

from nikita_bot.routing import KnownBotNames


class KnownBotNamesTests(unittest.TestCase):
    def test_detects_other_bot_first_name_and_username_as_direct_address(self) -> None:
        known_bots = KnownBotNames()
        known_bots.observe(
            10,
            SimpleNamespace(first_name="Нэмочка", username="nemo_helper", bot=True),
        )

        self.assertTrue(known_bots.addresses_other_bot(10, "Нэмочка, ты жива?"))
        self.assertTrue(known_bots.addresses_other_bot(10, "@nemo_helper ты жива"))

    def test_does_not_match_name_inside_another_word_or_in_another_chat(self) -> None:
        known_bots = KnownBotNames()
        known_bots.observe(
            10,
            SimpleNamespace(first_name="Кира", username="kira_bot", bot=True),
        )

        self.assertFalse(known_bots.addresses_other_bot(10, "Киран, ты жива?"))
        self.assertFalse(known_bots.addresses_other_bot(11, "Кира, ты жива?"))

    def test_ignores_non_bot_names(self) -> None:
        known_bots = KnownBotNames()
        known_bots.observe(
            10,
            SimpleNamespace(first_name="Кира", username="kira", bot=False),
        )

        self.assertFalse(known_bots.addresses_other_bot(10, "Кира, ты жива?"))
