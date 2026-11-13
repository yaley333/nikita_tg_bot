import json
import unittest
from unittest.mock import patch

import httpx

from nikita_bot.model import OllamaClient


class OllamaClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_sends_qwen_chat_request(self) -> None:
        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={"message": {"content": "Привет"}},
            )

        client = OllamaClient("http://ollama.test", "qwen3.5:4b")
        http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch.object(client, "_http", http):
            try:
                answer = await client.generate(
                    [{"role": "user", "content": "Привет"}]
                )
            finally:
                await client.close()

        payload = json.loads(requests[0].content)
        self.assertEqual(answer, "Привет")
        self.assertEqual(str(requests[0].url), "http://ollama.test/api/chat")
        self.assertEqual(payload["model"], "qwen3.5:4b")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["temperature"], 0.5)
        self.assertIn("Ты Ася", payload["messages"][0]["content"])

    async def test_empty_response_retries_with_a_larger_token_limit(self) -> None:
        requests: list[dict[str, object]] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                return httpx.Response(
                    200,
                    json={
                        "message": {"content": ""},
                        "done_reason": "length",
                        "eval_count": 96,
                    },
                )
            return httpx.Response(200, json={"message": {"content": "Ответ"}})

        client = OllamaClient("http://ollama.test", "qwen3.5:4b")
        http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch.object(client, "_http", http):
            try:
                answer = await client.generate(
                    [{"role": "user", "content": "Расскажи о Японии"}]
                )
            finally:
                await client.close()

        self.assertEqual(answer, "Ответ")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["options"]["num_predict"], 64)
        self.assertEqual(requests[1]["options"]["num_predict"], 128)
        self.assertIn("не оставляй ответ пустым", requests[1]["messages"][0]["content"])

    async def test_empty_proactive_response_retries_without_forcing_a_reply(self) -> None:
        requests: list[dict[str, object]] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                return httpx.Response(
                    200,
                    json={"message": {"content": ""}, "done_reason": "stop"},
                )
            return httpx.Response(200, json={"message": {"content": "[SKIP]"}})

        client = OllamaClient("http://ollama.test", "qwen3.5:4b")
        http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch.object(client, "_http", http):
            try:
                answer = await client.generate(
                    [{"role": "user", "content": "Контекст группы"}],
                    proactive=True,
                )
            finally:
                await client.close()

        self.assertEqual(answer, "[SKIP]")
        self.assertIn("[SKIP]", requests[1]["messages"][0]["content"])
        self.assertNotIn("не оставляй ответ пустым", requests[1]["messages"][0]["content"])

    async def test_empty_response_after_retry_includes_completion_details(self) -> None:
        def respond(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "message": {"content": ""},
                    "done_reason": "length",
                    "eval_count": 192,
                },
            )

        client = OllamaClient("http://ollama.test", "qwen3.5:4b")
        http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch.object(client, "_http", http):
            try:
                with self.assertRaisesRegex(
                    ValueError,
                    "done_reason='length'.*eval_count=192.*message_fields=\\['content'\\]",
                ):
                    await client.generate(
                        [{"role": "user", "content": "Расскажи о Японии"}]
                    )
            finally:
                await client.close()
