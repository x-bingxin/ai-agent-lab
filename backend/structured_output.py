from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI
from llm_client import OpenAILLMClient, LLMConfig
import json
import re

# ======= 简历解析相关的 Pydantic 模型 =======
class WorkExperience(BaseModel):
    company: str = Field(description="公司名称")
    position: str = Field(description="职位名称")
    start_date: str = Field(description="开始日期")
    end_date: Optional[str] = Field(description="结束日期，若在职则为 None")
    description: Optional[str] = Field(description="工作描述")

class Resume(BaseModel):
    name: str = Field(description="姓名")
    email: Optional[str] = Field(description="邮箱")
    phone: Optional[str] = Field(description="电话号码")
    education: Optional[List[str]] = Field(description="教育背景列表")
    skills: Optional[List[str]] = Field(description="技能列表")
    work_experience: Optional[List[WorkExperience]] = Field(description="工作经历列表")

    class Config:
        extra = "ignore"



# ======= 使用 OpenAI structured output =======

async def parse_resume(text: str, client: OpenAI, model: str) -> Resume:
    """使用 OpenAI 的 structured output 功能解析简历文本，返回 Resume 对象"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": """你是一个简历解析助手。从用户提供的简历文本中提取结构化信息，返回一个严格符合以下格式的 JSON 对象：
                - name: 字符串，姓名。
                - email: 字符串或 null，邮箱。
                - phone: 字符串或 null，电话号码。
                - education: 字符串列表或 null，教育背景列表（例如 ["清华大学计算机科学与技术专业"]）。
                - skills: 字符串列表或 null，技能列表（例如 ["Python", "Java", "C++"]）。
                - work_experience: WorkExperience 对象列表或 null，每个对象包含 company (字符串，公司名称), position (字符串，职位名称), start_date (字符串，开始日期), end_date (字符串或 null，结束日期，若在职则为 null), description (字符串或 null，工作描述)。
                找不到的信息留空，不要编造。确保输出是有效的 JSON。"""},
            {"role": "user", "content": f"请解析以下简历文本，并返回一个 JSON 格式的 Resume 对象：\n\n{text}"}
        ],
        temperature=0,
        response_format={"type": "json_object"} # 对于国产模型更推荐这种方式，然后手动来处理json parse
        # response_format=Resume
    )

    # 正确的格式化输出方案： 获取原始文本 -> 手动 json.loads -> pydantic validate -> retry/repair
    content = response.choices[0].message.content
    print(f"LLM 输出：{content}")

    try:
        json_str = extract_json(content)
        data = json.loads(json_str)
        resume = Resume.model_validate(data)
        return resume
    except Exception as e:
        print(f"Format response error: {e}")
        # repaired = await repair_json(content, client, model)
        # return Resume.model_validate(repaired)


def extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON found")

    return match.group()



# ============ 意图分类（Agent 路由的基础）=============

from enum import Enum

class Intent(str, Enum):
    ORDER_STATUS = "order_status"
    ORDER_CANCEL = "order_cancel"
    REFUND_REQUEST = "refund_request"
    TECHNICAL_SUPPORT = "technical_support"
    GENERAL_CHAT = "general_chat"

class IntentClassification(BaseModel):
    intent: Intent = Field(description="用户意图分类")
    confidence: float = Field(ge=0, le=1, description="置信度0-1")
    reason: str = Field(description="分类理由")

async def classify_intent(user_message: str, client: OpenAI, model: str) -> IntentClassification:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": """你是一个客服意图分类器。将用户消息分到以下类别：
                - order_status: 查询订单状态
                - order_cancel: 取消订单
                - refund_request: 退款申请
                - technical_support: 技术问题
                - general_chat: 闲聊
                输出分类，置信度和理由。最终数据返回为json格式"""},
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "json_object"} # 对于国产模型更推荐这种方式，然后手动来处理json parse
        # response_format=IntentClassification
    )
    content = response.choices[0].message.content
    print(f"意图分析结果： {content}")

    try:
        json_str = extract_json(content)
        data = json.loads(json_str)
        intent = IntentClassification.model_validate(data)
        return intent
    except Exception as e:
        print(f"Format response error: {e}")


async def main():
    from dotenv import load_dotenv
    load_dotenv()
    import os

    config = LLMConfig(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),    
        model = os.getenv("LLM_MODEL")
    )
    client = OpenAILLMClient(config)

    resume_text = """
        姓名：张三
        邮箱：zhangsan@example.com
        电话：13800138000
        教育背景：清华大学计算机科学与技术专业
        技能：Python, Java, C++
        工作经历：
        - 公司：ABC公司，职位：软件工程师，开始日期：2020-01-01，结束日期：2023-12-31，描述：负责开发软件。
    """
    # test parse_resume
    # resume = await parse_resume(resume_text, client.get_client(), config.model)
    # print(resume)

    # test user intent
    test_cases = [
        "我的订单什么时候到？",
        "我要退款，商品坏了",
        "Python和Java哪个好？",
        "你们的app打不开了"
    ]
    for i, case in enumerate(test_cases, 0):
        result = await classify_intent(case, client.get_client(), config.model)
        print(f"第 {i+1} 个意图是： {result} \n\n\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())