from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

class Intent(str, Enum):
    ORDER_QUERY = "order_query"
    COMPLAINT = "complaint"
    REFUND = "refund"
    CHAT = "chat"

class Message(BaseModel):
    # Literal 表示 role 只能是 "system", "user", "assistant", "tool" 中的一个
    role: Literal["system", "user", "assistant", "tool"]
    content: str

    # Optional 表示 tool_call_id 可以是 str 或 None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class AgentState(BaseModel):
    """这是 Agent 在图中流转的核心状态对象"""
    messages: List[Message] = Field(default_factory=list)
    intent: Optional[Intent] = None
    user_id: str
    session_id: str
    next_step: Optional[str] = None

# 测试：故意传入错误类型，看 pydantic 报错信息
try:
    state = AgentState(user_id=123, session_id="abc")
except Exception as e:
    print(f"Pydantic validation error: {e}")