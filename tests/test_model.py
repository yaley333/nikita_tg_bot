import json
import unittest
from unittest.mock import patch

import httpx

from nikita_bot.model import OllamaClient


class OllamaClientTests(unittest.IsolatedAsyncioTestCase):
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
                        "eval_count": 64,
                    },
                )
            return httpx.Response(200, json={"message": {"content": "Привет"}})

        client = OllamaClient("http://ollama.test", "qwen3.5:4b")
        http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch.object(client, "_http", http):
            try:
                answer = await client.generate(
                    [{"role": "user", "content": "Привет"}]
                )
            finally:
                await client.close()

        self.assertEqual(answer, "Привет")
        self.assertEqual(len(requests), 2)
        first_options = requests[0]["options"]
        retry_options = requests[1]["options"]
        self.assertEqual(first_options["num_predict"], 64)
        self.assertEqual(retry_options["num_predict"], 128)

    async def test_empty_response_after_retry_includes_completion_details(self) -> None:
        def respond(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "message": {"content": ""},
                    "done_reason": "length",
                    "eval_count": 128,
                },
            )

        client = OllamaClient("http://ollama.test", "qwen3.5:4b")
        http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch.object(client, "_http", http):
            try:
                with self.assertRaisesRegex(
                    ValueError,
                    "done_reason='length'.*eval_count=128",
                ):
                    await client.generate([{"role": "user", "content": "Привет"}])
            finally:
                await client.close()
