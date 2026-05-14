import asyncio

import tiktoken
from typing import List, Dict

class TokenManager:
    """管理token预算，避免超限和浪费"""

    def __init__(self, model: str = "gpt-4o"):
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        # 不同模型的上下文窗口
        self.context_limits = {
            "MiniMax-M2.5": 4096,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-3.5-turbo": 16385,
        }

    def count(self, text: str) -> int:
        """计算文本的token数量"""
        encodedStr = self.encoder.encode(text)
        return len(encodedStr)
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """估算多轮对话的token消耗"""
        total = 0
        for msg in messages:
            # 每条消息的role和content都要算
            total += self.count(msg.get("role", ""))
            total += self.count(msg.get("content", ""))
            # 还要加上格式化的token，比如每条消息前后可能有固定的token
            total += 4  # 这是一个经验值，具体根据模型可能需要调整
        total += 2  # 整个消息列表的开头和结尾可能有特殊token
        return total
    
    def fit_to_window(self, messages: List[Dict], max_tokens: int, reserve_for_output: int = 1024) -> List[Dict]:
        """截断消息历史以适应窗口"""
        available = max_tokens - reserve_for_output
        system_msg = None
        conversation = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg
            else:
                conversation.append(msg)
        
        # 主要一下是 python 里的三元表达式，先判断 system_msg 是否存在，如果存在就把它放在结果列表的开头，否则结果列表就是空的
        result = [system_msg] if system_msg else []
        used = self.count(system_msg['content']) if system_msg else 0

        # 从最新消息开始保留
        for msg in reversed(conversation):
            msg_tokens = self.count(msg['content']) + self.count(msg['role']) + 4  # 加上格式化token
            if used + msg_tokens > available:
                break
            result.insert(1, msg)  # 插入到 system_msg 后面
            used += msg_tokens

        return result


async def main():
    tm = TokenManager()
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm good, thanks! How can I assist you today?"},
        {"role": "user", "content": "Can you tell me a joke?"},
    ]
    import sys
    if len(sys.argv) > 1:
        text = sys.argv[1] if len(sys.argv) > 1 else open(sys.argv[1]).read()
        print(f"sys.argv[1]: {text}")
        print (f"Token count for sys.argv[1] {tm.count(text)}")
    print("Total tokens:", tm.count_messages(msgs))
    fitted = tm.fit_to_window(msgs, max_tokens=20, reserve_for_output=5)
    print("Fitted messages:", fitted)


if __name__ == "__main__":
    asyncio.run(main())