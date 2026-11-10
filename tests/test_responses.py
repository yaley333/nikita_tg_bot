import unittest

from nikita_bot.responses import LatestMessageCoordinator


class LatestMessageCoordinatorTests(unittest.TestCase):
    def test_latest_message_is_monotonic_and_scoped_to_chat(self) -> None:
        coordinator = LatestMessageCoordinator()
        coordinator.observe(10, 4)
        coordinator.observe(10, 6)
        coordinator.observe(10, 5)
        coordinator.observe(20, 2)

        self.assertFalse(coordinator.is_latest(10, 4))
        self.assertFalse(coordinator.is_latest(10, 5))
        self.assertTrue(coordinator.is_latest(10, 6))
        self.assertTrue(coordinator.is_latest(20, 2))

    def test_preserved_task_is_not_skipped_when_a_newer_message_arrives(self) -> None:
        coordinator = LatestMessageCoordinator()
        coordinator.observe(10, 4)
        coordinator.observe(10, 5)

        self.assertFalse(coordinator.should_process(10, 4))
        self.assertTrue(coordinator.should_process(10, 4, preserve=True))
        self.assertTrue(coordinator.should_process(10, 5))
