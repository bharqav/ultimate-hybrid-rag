import json

from config.logging import setup_logging
from config.settings import get_settings
from core.deps import FastAPI, JSONResponse, Request, Response, StreamingResponse, generate_latest
from retrieval.retriever import UltimateRetriever

log = setup_logging()
settings = get_settings()


async def stream_ollama(prompt: str, model_name: str = None):
    import aiohttp

    model_name = model_name or settings.ollama_model_default
    async with aiohttp.ClientSession() as session:
        payload = {"model": model_name, "prompt": prompt, "stream": True}
        async with session.post(settings.ollama_url, json=payload) as resp:
            async for line in resp.content:
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except Exception:
                        continue


app = FastAPI() if FastAPI else None
retriever = None

if app:

    @app.on_event("startup")
    async def startup():
        global retriever
        retriever = UltimateRetriever()

    @app.post("/query")
    async def query_endpoint(request: Request):
        data = await request.json()
        query = data.get("query", "")
        if not query:
            return JSONResponse({"error": "No query"}, 400)
        results = retriever.search(query)

        async def event_stream():
            meta = {"num_chunks": len(results), "top_scores": [r.fused_score for r in results]}
            yield f"data: {json.dumps(meta)}\n\n"
            context_parts = []
            for r in results:
                src = f"[Source: {r.chunk.source_document}, Page {r.chunk.page_number}]\n{r.chunk.text}"
                context_parts.append(src)
            context = "\n\n".join(context_parts)
            prompt = f"Answer using only this context:\n{context}\n\nQuestion: {query}\nAnswer:"
            async for token in stream_ollama(prompt):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/metrics")
    def metrics():
        if generate_latest:
            return Response(generate_latest(), media_type="text/plain")
        return {"message": "prometheus_client not installed"}
