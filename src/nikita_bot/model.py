import base64
import re

import httpx

from .persona import SYSTEM_PROMPT

VOICE_REPLY_INSTRUCTION = """Сейчас включён режим голосового ответа: приложение уже умеет озвучить твой ответ и отправить его. Выполни исходную просьбу пользователя напрямую. Не утверждай, что не умеешь отправлять голосовые, не упоминай микрофон, прослушивание аудио, текстовый интерфейс или технические ограничения. Не добавляй вступительные оправдания и заключительные предложения о том, что пользователь может сделать сам. Если просят факты, дай 2-3 простых, широко известных факта. Не выдумывай даты, названия и числа; если сомневаешься, не включай это утверждение. Пиши естественный текст без Markdown-разметки."""
IMAGE_PROMPT_INSTRUCTION = """Преобразуй описание пользователя в короткий, конкретный prompt на английском для генератора изображений Stable Diffusion. Сохрани предметы, стиль, цвета, композицию и настроение из запроса. Не добавляй новые детали. Верни только prompt, без пояснений, кавычек, Markdown и отрицательного prompt."""
PROACTIVE_INSTRUCTION = """Ты сам решил ненавязчиво добавить одну короткую реплику в активный групповой разговор. Продолжи последнюю тему или шутку, не отвечай так, будто собеседник только что задал тебе прямой вопрос.

Пример:
Контекст: «Алиса: кот уронил стакан», «Боб: кот считает, что стакан сам виноват».
Уместно: «Судя по всему, кот уже назначил стакан виновным. Адвокат у стакана есть?»
Неуместно: «Ну давай-ка ты сам объясни, что тут смешного» или «Кто тут главный?».

Не обвиняй участников во лжи или скрытых чувствах, не требуй объяснений, не начинай новую тему и не используй общие фразы. Если не находишь конкретного естественного продолжения, верни ровно [SKIP]."""
PROACTIVE_IMAGE_INSTRUCTION = """По недавнему контексту группы придумай уместный короткий prompt на английском для забавной иллюстрации, которой Nikita ненавязчиво оживит разговор. Не изображай реальных участников. Верни только prompt без объяснений."""
SMALL_TALK_PATTERN = re.compile(
    r"^(?:привет|здравствуй|здравствуйте|доброе утро|добрый день|добрый вечер|"
    r"как дела|как ты|как сам|как сама|что нового|что делаешь)[?!.,\s]*$",
    re.IGNORECASE,
)


def build_messages(
    history: list[dict[str, str]],
    voice_reply: bool = False,
    image_prompt: bool = False,
    proactive: bool = False,
) -> list[dict[str, object]]:
    system_prompt = SYSTEM_PROMPT
    if image_prompt and proactive:
        system_prompt = PROACTIVE_IMAGE_INSTRUCTION
    elif image_prompt:
        system_prompt = IMAGE_PROMPT_INSTRUCTION
    elif proactive:
        system_prompt = f"{system_prompt}\n\n{PROACTIVE_INSTRUCTION}"
    if voice_reply and not image_prompt:
        system_prompt = f"{system_prompt}\n\n{VOICE_REPLY_INSTRUCTION}"
    return [{"role": "system", "content": system_prompt}, *history]


def get_num_predict(
    history: list[dict[str, str]],
    image_prompt: bool = False,
    proactive: bool = False,
) -> int:
    if image_prompt:
        return 96
    if proactive:
        return 96
    last_user_message = next(
        (message["content"].strip() for message in reversed(history) if message["role"] == "user"),
        "",
    )
    if SMALL_TALK_PATTERN.fullmatch(last_user_message):
        return 64
    return 512


class OllamaClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model
        self._http = httpx.AsyncClient(timeout=180)

    async def close(self) -> None:
        await self._http.aclose()

    async def generate(
        self,
        history: list[dict[str, str]],
        image: bytes | None = None,
        voice_reply: bool = False,
        image_prompt: bool = False,
        proactive: bool = False,
    ) -> str:
        messages = build_messages(history, voice_reply, image_prompt, proactive)
        if image and history:
            messages[-1] = {
                **messages[-1],
                "images": [base64.b64encode(image).decode("ascii")],
            }

        response = await self._http.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0.2 if image_prompt else 0.3 if voice_reply or proactive else 0.8,
                    "num_predict": get_num_predict(history, image_prompt, proactive),
                },
            },
        )
        response.raise_for_status()
        answer = response.json().get("message", {}).get("content", "").strip()
        if not answer:
            raise ValueError("Модель вернула пустой ответ")
        return answer

    async def unload(self) -> None:
        response = await self._http.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": "", "stream": False, "keep_alive": 0},
        )
        response.raise_for_status()