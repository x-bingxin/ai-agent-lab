import asyncio

import httpx
from pydantic import BaseModel
from openai import OpenAI


class LLMConfig(BaseModel):
    api_key: str
    base_url: str
    model: str
    timeout: int = 60
    max_retries: int = 3


class OpenAILLMClient:
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        import os

        config = LLMConfig(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),    
            model = os.getenv("LLM_MODEL")
        )

        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

class EmbeddingModelEntry:
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        import os

        config = LLMConfig(
            api_key=os.getenv("EMBEDDING_MODEL_API_KEY"),
            base_url=os.getenv("EMBEDDING_MODEL_BASE_URL"),    
            model=os.getenv("EMBEDDING_MODEL_NAME")
        )

        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
    

class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

        # 连接池复用
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=self.config.timeout,
        )

    async def chat(self, messages: list[dict]) -> dict:
        """单次调用，拿到完整相应"""
        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.post(
                    "/chat/completions",
                    json={
                        "model": self.config.model,
                        "messages": messages,
                    },
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                print(f"LLM API error (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # 指数退避

    async def stream_chat(self, messages: list[dict]):
        """流式输出，返回token迭代器"""
        async with self._client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": self.config.model,
                "messages": messages,
                "stream": True,
            }
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    import json
                    chunk=json.loads(line[6:])
                    delta=chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]

    async def close(self):
        await self._client.aclose()


async def main():
    config = LLMConfig(
        api_key="sk-88451ca8791545c7b7a1181516601a27",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="MiniMax-M2.5"
    )
    client = LLMClient(config)
    async for token in client.stream_chat([
        {"role": "user", "content": "用一句话总结java和python的区"}
    ]):
        print(token, end="", flush=True)
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())