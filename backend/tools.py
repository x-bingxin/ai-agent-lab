from pydantic import BaseModel, Field
from typing import Literal
from llm_client import LLMClient, LLMConfig
import json
import asyncio


# ======== 工具定义 =========
# 每个 Tool = 函数 + Pydantic参数模型 + docstring

class WeatherParams(BaseModel):
    city: str = Field(description="城市名，如‘北京’")
    unit: Literal["celsius", "fahrenheit"] = "celsius"

async def get_weather(city: str, unit: str = "celsius") -> dict:
    """查询指定城市的天气信息"""
    mock_data = {
        "北京": {"celsius": 18, "fahrenheit": 64, "condition": "晴"},
        "上海": {"celsius": 22, "fahrenheit": 72, "condition": "多云"},
        "深圳": {"celsius": 28, "fahrenheit": 82, "condition": "阵雨"},
    }

    weather = mock_data.get(city, {"celsius": 20, "fahrenheit": 68, "condition": "未知"})
    return {
        "city": city,
        "temperature": weather[unit],
        "unit": unit,
        "condition": weather["condition"]
    }


class CalculatorParams(BaseModel):
    expression: str = Field(description="数学表达式，如‘2+3*4’")

async def calculate(expression: str) -> dict:
    """执行数学计算"""
    try:
        # 安全的eval， 仅允许数字和运算符
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return {"error": "表达式含有非法字符"}
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


class OrderQueryParams(BaseModel):
    order_id: str = Field(description="订单号")

async def query_order(order_id: str) -> dict:
    """查询订单状态"""
    # mock 数据
    return {
        "order_id": order_id,
        "status": "已发货",
        "tracking_number": "SF1234567890",
        "estimated_delivery": "2025-01-20"
    }



# ============= 工具注册表 ==============
# Agent 核心，告诉 LLM 哪些工具可以调用

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "根据订单号查询订单状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号"}
                },
                "required": ["order_id"]
            }
        }
    }
]

# 工具名到函数的映射
TOOL_HANDLERS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "query_order": query_order
}


# ============= 手动 ReAct 循环（理解 Agent 本质）==============

async def react_loop(user_input: str, llm_client: LLMClient, max_iterations: int = 5):
    """手动实现 ReAct 循环，就是 LangChain AgentExecutor 内部逻辑"""
    messages = [
        {"role": "system", "content": "你是一个有用的助手，当需要实时信息是，使用工具来获取。"},
        {"role": "user", "content": user_input}
    ]

    for iteration in range(max_iterations):
        print(f"\n======== Iteration {iteration + 1} ========")

        # 1. 让 LLM 思考（可能决定使用工具，也可能直接回答
        response = await llm_client._client.post(
            "/chat/completions",
            json={
                "model": llm_client.config.model,
                "messages": messages,
                "tools": TOOL_DEFINITIONS,
                "tool_choice": "auto" # LLM 自己决定是否调用工具
            }
        )

        result = response.json()
        choice = result['choices'][0]

        # 2. 检查 LLM 是否决定调用工具
        if choice["finish_reason"] == "tool_calls":
            assistant_msg = choice["message"]
            messages.append(assistant_msg) # 把工具调用加入历史消息

            for tool_call in assistant_msg["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])

                print(f"-> 调用工具： {tool_name}({tool_args})")

                # 3. 执行工具
                handler = TOOL_HANDLERS[tool_name]
                tool_result = await handler(**tool_args)
                
                print(f"← 工具返回: {tool_result}")
                
                # 4. 把工具结果加入消息历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })

            # 5. 继续循环，让LLM基于工具结果再思考
            continue

        elif choice["finish_reason"] == "stop":
            # LLM认为可以回答了
            result = choice["message"]["content"]
            print(f"最终回答：{result}")
            return result
        
    return "抱歉，我无法在限定步骤内完成这个任务。"



async def main():
    from dotenv import load_dotenv
    load_dotenv()
    import os

    config = LLMConfig(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),    
        model = os.getenv("LLM_MODEL")
    )
    user_input = "北京今天天气怎么样？华氏温度转成摄氏度是多少？"
    client = LLMClient(config)

    await react_loop(user_input, client, 5)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())