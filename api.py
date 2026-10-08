"""
Smart Education Project — REST API Service
============================================
Lightweight asynchronous REST API powered by Starlette & Uvicorn.
Provides programmatic HTTP endpoints for:
- Health check & hardware telemetry (`GET /health`)
- 5-Model student knowledge state & recommendation prediction (`POST /predict/student`)
- Multimodal attention classification from video upload or features (`POST /predict/attention`)
- End-to-end multimodal pipeline (`POST /predict/multimodal`)

Launch command:
  python api.py
  OR:
  uvicorn api:app --host 0.0.0.0 --port 8000
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import torch
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Lazy engine holders
_kt_engine = None
_att_engine = None


def get_kt_engine():
    global _kt_engine
    if _kt_engine is None:
        from main import MultiModelInferenceEngine
        _kt_engine = MultiModelInferenceEngine()
    return _kt_engine


def get_att_engine():
    global _att_engine
    if _att_engine is None:
        from attention.attention_inference import AttentionInferenceEngine
        _att_engine = AttentionInferenceEngine()
    return _att_engine


_ai_generator = None
_ai_tutor = None


def get_ai_generator():
    global _ai_generator
    if _ai_generator is None:
        from ollama_ai.response_generator import ResponseGenerator
        _ai_generator = ResponseGenerator()
    return _ai_generator


def get_ai_tutor():
    global _ai_tutor
    if _ai_tutor is None:
        from ollama_ai.tutor import AITutor
        _ai_tutor = AITutor()
    return _ai_tutor


async def health_endpoint(request):
    """Health check, GPU status, and Generative AI availability."""
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    ai_gen = get_ai_generator()
    ai_online = ai_gen.is_ai_available()
    from ollama_ai.ollama_config import OLLAMA_MODEL, OLLAMA_ENABLED, OLLAMA_BASE_URL
    return JSONResponse({
        "status": "healthy",
        "service": "Smart Education AI Backend",
        "cuda_available": torch.cuda.is_available(),
        "device": device_name,
        "models": [
            "Model 1: LSTM + Attention KT",
            "Model 2: Transformer + KT",
            "Model 3: BERT-style Transformer + NCF",
            "Model 4: Autoencoder + Recommender",
            "Model 5: CNN + LSTM",
            "Model 6: Multimodal Video + Audio Attention"
        ],
        "generative_ai": {
            "enabled": OLLAMA_ENABLED,
            "ollama_online": ai_online,
            "model": OLLAMA_MODEL,
            "base_url": OLLAMA_BASE_URL
        }
    })



async def predict_student_endpoint(request):
    """5-Model prediction for a student ID."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    student_id = data.get("student_id")
    split = data.get("split", "test")

    if not student_id:
        return JSONResponse({"error": "Missing 'student_id' field"}, status_code=400)

    try:
        engine = get_kt_engine()
        res = engine.predict_student_by_id(student_id, split=split)
        if not res:
            return JSONResponse({"error": f"Student '{student_id}' not found"}, status_code=404)
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def predict_attention_endpoint(request):
    """Predict attention from uploaded video file or JSON features."""
    content_type = request.headers.get("content-type", "")

    try:
        att_engine = get_att_engine()

        if "multipart/form-data" in content_type:
            form = await request.form()
            file = form.get("video")
            if not file:
                return JSONResponse({"error": "Missing 'video' file in multipart form"}, status_code=400)

            suffix = Path(file.filename).suffix if hasattr(file, "filename") else ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                content = await file.read()
                tf.write(content)
                temp_path = tf.name

            try:
                res = att_engine.predict_video(temp_path)
                return JSONResponse(res)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        elif "application/json" in content_type:
            data = await request.json()
            v_feat = data.get("video_features")
            a_feat = data.get("audio_features")
            if not v_feat and not a_feat:
                return JSONResponse({"error": "Must provide 'video_features' or 'audio_features'"}, status_code=400)

            import numpy as np
            vf = np.array(v_feat, dtype=np.float32) if v_feat else np.zeros(10, dtype=np.float32)
            af = np.array(a_feat, dtype=np.float32) if a_feat else np.zeros(10, dtype=np.float32)

            res = att_engine.predict_features(
                vf, af,
                video_available=(v_feat is not None),
                audio_available=(a_feat is not None)
            )
            return JSONResponse(res)

        else:
            return JSONResponse({"error": "Expected multipart/form-data or application/json"}, status_code=415)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def predict_multimodal_endpoint(request):
    """End-to-end multimodal pipeline: video upload + optional student ID."""
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        return JSONResponse({"error": "Requires multipart/form-data containing 'video' and optional 'student_id'"}, status_code=415)

    try:
        form = await request.form()
        file = form.get("video")
        student_id = form.get("student_id")

        if not file:
            return JSONResponse({"error": "Missing 'video' file field"}, status_code=400)

        suffix = Path(file.filename).suffix if hasattr(file, "filename") else ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            content = await file.read()
            tf.write(content)
            temp_path = tf.name

        try:
            from attention.attention_pipeline import run_video_attention_pipeline
            kt_eng = get_kt_engine() if student_id else None
            att_eng = get_att_engine()

            pipeline_res = run_video_attention_pipeline(
                video_path=temp_path,
                student_id=student_id,
                attention_engine=att_eng,
                kt_engine=kt_eng
            )
            return JSONResponse(pipeline_res)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def ai_status_endpoint(request):
    """Returns Ollama Generative AI configuration and connection status."""
    ai_gen = get_ai_generator()
    from ollama_ai.ollama_config import get_ollama_config
    cfg = get_ollama_config()
    cfg["ollama_online"] = ai_gen.is_ai_available()
    cfg["installed_models"] = ai_gen.client.list_models() if cfg["ollama_online"] else []
    return JSONResponse(cfg)


async def ai_explain_learning_gap_endpoint(request):
    """Generates natural language explanation of diagnosed learning gaps."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    student_id = data.get("student_id")
    split = data.get("split", "test")
    model = data.get("model")

    from ollama_ai.learner_context import build_learner_context
    if student_id:
        kt_eng = get_kt_engine()
        ml_res = kt_eng.predict_student_by_id(student_id, split=split)
        if not ml_res:
            return JSONResponse({"error": f"Student '{student_id}' not found"}, status_code=404)
        ctx = build_learner_context(inference_result=ml_res, student_id=student_id)
    elif "context" in data:
        ctx = build_learner_context(inference_result=data["context"])
    else:
        return JSONResponse({"error": "Must provide 'student_id' or 'context'"}, status_code=400)

    ai_gen = get_ai_generator()
    res = ai_gen.explain_learning_gaps(ctx, model=model)
    return JSONResponse(res)


async def ai_explain_recommendation_endpoint(request):
    """Explains why specific EdNet resources were selected."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    student_id = data.get("student_id")
    split = data.get("split", "test")
    model = data.get("model")

    from ollama_ai.learner_context import build_learner_context
    if student_id:
        kt_eng = get_kt_engine()
        ml_res = kt_eng.predict_student_by_id(student_id, split=split)
        if not ml_res:
            return JSONResponse({"error": f"Student '{student_id}' not found"}, status_code=404)
        ctx = build_learner_context(inference_result=ml_res, student_id=student_id)
    elif "context" in data:
        ctx = build_learner_context(inference_result=data["context"])
    else:
        return JSONResponse({"error": "Must provide 'student_id' or 'context'"}, status_code=400)

    ai_gen = get_ai_generator()
    res = ai_gen.explain_recommendations(ctx, model=model)
    return JSONResponse(res)


async def ai_study_plan_endpoint(request):
    """Generates a personalized study schedule based on diagnosed gaps."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    student_id = data.get("student_id")
    split = data.get("split", "test")
    target_hours = float(data.get("target_hours", 2.0))
    model = data.get("model")

    from ollama_ai.learner_context import build_learner_context
    if student_id:
        kt_eng = get_kt_engine()
        ml_res = kt_eng.predict_student_by_id(student_id, split=split)
        if not ml_res:
            return JSONResponse({"error": f"Student '{student_id}' not found"}, status_code=404)
        ctx = build_learner_context(inference_result=ml_res, student_id=student_id)
    elif "context" in data:
        ctx = build_learner_context(inference_result=data["context"])
    else:
        return JSONResponse({"error": "Must provide 'student_id' or 'context'"}, status_code=400)

    ai_gen = get_ai_generator()
    res = ai_gen.generate_study_plan(ctx, target_hours=target_hours, model=model)
    return JSONResponse(res)


async def ai_tutor_endpoint(request):
    """Interactive educational AI Tutor answering questions using learner context."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    query = data.get("query")
    if not query:
        return JSONResponse({"error": "Missing 'query' field in JSON request"}, status_code=400)

    student_id = data.get("student_id")
    split = data.get("split", "test")
    chat_history = data.get("chat_history", [])
    model = data.get("model")

    from ollama_ai.learner_context import build_learner_context
    if student_id:
        kt_eng = get_kt_engine()
        ml_res = kt_eng.predict_student_by_id(student_id, split=split)
        ctx = build_learner_context(inference_result=ml_res, student_id=student_id) if ml_res else build_learner_context(student_id=student_id)
    elif "context" in data:
        ctx = build_learner_context(inference_result=data["context"])
    else:
        ctx = build_learner_context()

    ai_tutor = get_ai_tutor()
    res = ai_tutor.ask(ctx, query=query, chat_history=chat_history, model=model)
    return JSONResponse(res)


routes = [
    Route("/health", health_endpoint, methods=["GET"]),
    Route("/predict/student", predict_student_endpoint, methods=["POST"]),
    Route("/predict/attention", predict_attention_endpoint, methods=["POST"]),
    Route("/predict/multimodal", predict_multimodal_endpoint, methods=["POST"]),
    Route("/ai/status", ai_status_endpoint, methods=["GET"]),
    Route("/ai/explain-learning-gap", ai_explain_learning_gap_endpoint, methods=["POST"]),
    Route("/ai/explain-recommendation", ai_explain_recommendation_endpoint, methods=["POST"]),
    Route("/ai/study-plan", ai_study_plan_endpoint, methods=["POST"]),
    Route("/ai/tutor", ai_tutor_endpoint, methods=["POST"]),
]


middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
]

app = Starlette(routes=routes, middleware=middleware)


if __name__ == "__main__":
    import uvicorn
    print("Starting Smart Education REST API on http://0.0.0.0:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
