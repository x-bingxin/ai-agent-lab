from llm_client import LLMClient, LLMConfig
import os

class ContextManager:
    """当对话太长时，自动压缩历史"""
    def __init__(self, client: LLMClient):
        self._client = client

    async def summarize_history(self, messages: list[dict]) -> str:
        """调用模型总结对话历史，返回一个简短的摘要"""

        prompt = """请总结以下对话历史，压缩为一段不超过200字的摘要，保留关键信息：
            对话：{conversation}

            摘要：
        """
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages if msg['role'] != 'system'])

        response = await self._client.chat([
            {"role": "user", "content": prompt.format(conversation=conversation_text)}
        ])

        return response["choices"][0]["message"]["content"]
    

    async def manage_context(self, messages: list[dict], max_tokens: int) -> list[dict]:
        """如果消息历史超过token限制，先总结历史再返回"""
        from token_utils import TokenManager
        token_utils = TokenManager()

        total_tokens = token_utils.count_messages(messages)

        print(f"当前 messages total tokens: {total_tokens}")

        if total_tokens <= max_tokens * 0.8:
            return messages
        
        # 分离 system prompt 和对话历史，保留 system prompt
        system = [m for m in messages if m["role"] == "system"]
        conversation = [m for m in messages if m["role"] != "system"]

        # 保留最近 3 轮对话
        recent = conversation[-3:]
        old = conversation[:-3]

        if old:
            print(f"开始总结旧的 messages : {total_tokens}")
            summary = await self.summarize_history(old)
            summarized = [{"role": "system", "content": f"历史对话总结：{summary}"}]
            return system + summarized + recent
        
        return system + recent