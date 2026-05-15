from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import RAGPipeline
from embeddings import EmbeddingService
from llm_client import EmbeddingModelEntry
from vector_store import DocumentStore
from tools import TOOL_DEFINITIONS, TOOL_HANDLERS


from llm_client import LLMClient, LLMConfig
import os
from dotenv import load_dotenv

load_dotenv()

# 全局客户端
llm_client = LLMClient(LLMConfig(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    model=os.getenv("LLM_MODEL"),
))

app = FastAPI(title="AI Agent Lab API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_session"

model_entry = EmbeddingModelEntry()
embedding_service = EmbeddingService(model_entry.client, model_entry.config.model)
doc_store = DocumentStore()
ragPip = RAGPipeline(embedding_service, doc_store, llm_client)

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE 流式接口，调用 LLMClient 的 stream_chat 方法"""
    async def generate():
        try:
            if doc_store.is_empty():
                async for token in llm_client.stream_chat_with_tools([
                    {"role": "user", "content": request.message}
                ], TOOL_DEFINITIONS, TOOL_HANDLERS):
                    yield f"data: {token}\n\n"
            else:
                async for token in ragPip.stream_query(request.message):
                    # SSE 格式要求每条消息以 "data: " 开头，结尾以两个换行符结束
                    yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


class KnowledgeRequest(BaseModel):
    content: str
    # session_id: str = "default_session"

@app.post("/knowledge/process")
async def knowledge_process(request: KnowledgeRequest):
    """调用 vector store 将知识写入"""

    content = request.content
    if content is not None:
        await ragPip.ingest(request.content)
        await embedding_service.embed(request.content)
    else:
        doc_store.cleanup()

    return {"data": "SUCCESS"}
    

# @app.post("/chat")
# async def chat(request: ChatRequest):
#     """非流式接口"""
#     return {"reply": f"Echo: {request.message}"}

# @app.post("/chat/stream")
# async def chat_stream(request: ChatRequest):
#     """SSE 流式接口，模拟把输入的 message 切成 10 份依次返回"""
#     message = request.message or ""
#     parts = []
#     total = len(message)
#     if total == 0:
#         parts = [""] * 10
#     else:
#         base = total // 10
#         rem = total % 10
#         start = 0
#         for i in range(10):
#             end = start + base + (1 if i < rem else 0)
#             parts.append(message[start:end])
#             start = end

#     async def generate():
#         for i, part in enumerate(parts):
#             # 相当于每隔0.5秒返回一部分数据，模拟流式输出
#             yield f"data: {part}\n\n"
#             import asyncio
#             await asyncio.sleep(0.5)
#         yield "data: [DONE]\n\n"

#     return StreamingResponse(
#         generate(),
#         media_type="text/event-stream",
#         headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
#     )

# 启动: uv run uvicorn main:app --reload
# 查看端口占用情况： lsof -i :8000