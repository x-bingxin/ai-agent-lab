import asyncio
import inspect

import httpx
from pydantic import BaseModel
from openai import OpenAI
import json


class LLMConfig(BaseModel):
    api_key: str
    base_url: str
    model: str
    timeout: int = 60
    max_retries: int = 3


class OpenAILLMEntry:
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

        self._openai_client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

        self.context_messages = []

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

    async def stream_chat_with_tools(self, messages: list[dict], tools: list[dict], tool_handlers: dict[str, callable], max_iterations: int = 5):
        """支持 tool calling 的流式对话，内部自动处理工具调用，对外仍然流式输出文本 token"""

        try:
            from context_manager import ContextManager
            contextManager = ContextManager(self)
            self.context_messages = await contextManager.manage_context(self.context_messages, 100)
        except Exception as e:
            print(f"Manage context 错误：{str(e)}")

        self.context_messages.extend(messages)

        for _ in range(max_iterations):
            content_parts = []
            collected_tool_calls = []
            finish_reason = None

            payload = {
                'model': self.config.model,
                'messages': self.context_messages,
                'stream': True,
            }

            if tools:
                payload['tools'] = tools
                payload['tool_choice'] = 'auto'

            print("当前 Context：")
            print(*self.context_messages, sep='\n')
            print("\n"*3)

            async with self._client.stream(
                'Post',
                'chat/completions',
                json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    # print(f"====> {line}")
                    if line.startswith('data: ') and line != 'data: [DONE]':
                        chunk = json.loads(line[6:])
                        choice = chunk['choices'][0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason") or finish_reason

                        if "content" in delta and delta["content"]:
                            content_parts.append(delta["content"])
                            yield delta["content"]

                        if "tool_calls" in delta and delta["tool_calls"]:
                            for tc_delta in delta["tool_calls"]:
                                index = tc_delta.get("index", 0)
                                while len(collected_tool_calls) <= index:
                                    collected_tool_calls.append({
                                        "id": None,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })
                                tc = collected_tool_calls[index]
                                if tc_delta.get("id"):
                                    tc["id"] = tc_delta["id"]
                                if tc_delta.get("type"):
                                    tc["type"] = tc_delta["type"]
                                fn_delta = tc_delta.get("function", {})
                                if fn_delta.get("name"):
                                    tc["function"]["name"] = fn_delta["name"]
                                if fn_delta.get("arguments"):
                                    tc["function"]["arguments"] += fn_delta["arguments"]

            assistant_msg = {
                "role": "assistant",
                "content": "".join(content_parts) if content_parts else None,
            }

            # 检查是否需要调用工具
            if finish_reason == "tool_calls" or any(tc.get("id") for tc in collected_tool_calls):
                assistant_msg["tool_calls"] = collected_tool_calls
                self.context_messages.append(assistant_msg)

                for tc in collected_tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"]["arguments"])
                    handler = tool_handlers.get(tool_name)

                    if not handler:
                        result = {"error": f"Tool {tool_name} not found"}
                    # elif asyncio.iscoroutinefunction(handler):
                    elif inspect.iscoroutinefunction(handler):
                        result = await handler(**tool_args)
                    else:
                        result = handler(**tool_args)

                    self.context_messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                continue
            else:
                self.context_messages.append(assistant_msg)
                break

    async def close(self):
        await self._client.aclose()


def get_llm_client():
    from llm_client import LLMClient, LLMConfig
    from dotenv import load_dotenv
    import os

    load_dotenv()

    llm_client = LLMClient(LLMConfig(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL"),
    ))
    return llm_client



async def main():

    client = get_llm_client()

    # async for token in client.stream_chat([
    #     {"role": "user", "content": "用一句话总结java和python的区"}
    # ]):
    #     print(token, end="", flush=True)
    
    from tools import TOOL_DEFINITIONS, TOOL_HANDLERS

    # await client.stream_chat_with_tools([
    #     {"role": "user", "content": "北京天气怎么样？"},
    # ], tools=TOOL_DEFINITIONS, tool_handlers=TOOL_HANDLERS)


    async for token in client.stream_chat_with_tools([
        {"role": "user", "content": "北京天气怎么样？"},
    ], tools=TOOL_DEFINITIONS, tool_handlers=TOOL_HANDLERS):
        print(token, end="", flush=True)

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())