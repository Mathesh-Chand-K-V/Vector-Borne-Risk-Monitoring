import asyncio
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException
)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from app.config import load_generator_config
from app.generator import DatasetGenerator
from app.utils.logging import logger


app = FastAPI(
    title="Satellite Image + Historical Weather Dataset Generator"
)

active_generator: Optional[DatasetGenerator] = None
generator_thread: Optional[threading.Thread] = None
connected_websockets: List[WebSocket] = []

# The event loop is captured from the FastAPI request context.
# Background threads use this reference to schedule WebSocket updates.
main_event_loop: Optional[asyncio.AbstractEventLoop] = None

static_dir = Path(__file__).parent / "static"

app.mount(
    "/static",
    StaticFiles(directory=str(static_dir)),
    name="static"
)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_file = static_dir / "index.html"
    return HTMLResponse(
        content=index_file.read_text(encoding="utf-8")
    )


async def broadcast_ws(payload: Dict[str, Any]):
    disconnected = []

    for websocket in list(connected_websockets):
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            disconnected.append(websocket)

    for websocket in disconnected:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)


def sync_progress_callback(data: Dict[str, Any]):
    """
    Schedule WebSocket broadcasting safely from the generator thread.
    """
    loop = main_event_loop

    if loop is None or not loop.is_running():
        return

    try:
        asyncio.run_coroutine_threadsafe(
            broadcast_ws({
                "type": "progress",
                "data": data
            }),
            loop
        )
    except RuntimeError as exc:
        logger.warning(
            f"Could not schedule progress broadcast: {exc}"
        )


@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)

    try:
        if active_generator:
            generator = active_generator

            data = {
                "target": generator.target_count,
                "completed": generator.completed_count,
                "rejected": generator.rejected_count,
                "failed": generator.failed_count,
                "remaining": max(
                    0,
                    generator.target_count - generator.completed_count
                ),
                "is_paused": generator._is_paused,
                "is_cancelled": generator._is_cancelled
            }

            await websocket.send_text(
                json.dumps({
                    "type": "progress",
                    "data": data
                })
            )

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)

    finally:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)


class StartRequest(BaseModel):
    location: Optional[str] = None
    radius_km: Optional[float] = None
    target_samples: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    max_cloud_coverage: Optional[float] = None
    max_date_difference_days: Optional[int] = None
    image_size: Optional[int] = None
    workers: Optional[int] = None
    random_seed: Optional[int] = None
    sampling_strategy: Optional[str] = None
    output_dir: Optional[str] = None


@app.post("/api/start")
async def start_generation(request: StartRequest):
    global active_generator
    global generator_thread
    global main_event_loop

    if generator_thread and generator_thread.is_alive():
        raise HTTPException(
            status_code=400,
            detail="A generation task is already running."
        )

    # Capture the FastAPI event loop before starting the worker thread.
    main_event_loop = asyncio.get_running_loop()

    try:
        config = load_generator_config(
            overrides=request.model_dump(exclude_none=True)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid generator configuration: {exc}"
        )

    generator_instance = DatasetGenerator(
        config=config,
        progress_callback=sync_progress_callback
    )

    active_generator = generator_instance

    def run_wrapper():
        logger.info("Background generation thread started.")

        try:
            result = generator_instance.run()

            logger.info(
                "Background generation completed with status: "
                f"{result.get('status')}"
            )

            status = result.get("status", "completed")

            sync_progress_callback({
                "target": generator_instance.target_count,
                "completed": generator_instance.completed_count,
                "rejected": generator_instance.rejected_count,
                "failed": generator_instance.failed_count,
                "remaining": max(
                    0,
                    generator_instance.target_count
                    - generator_instance.completed_count
                ),
                "progress_percent": (
                    round(
                        generator_instance.completed_count
                        / generator_instance.target_count * 100,
                        1
                    )
                    if generator_instance.target_count else 0
                ),
                "is_paused": False,
                "is_cancelled": status == "cancelled",
                "current_stage": "Finished & Validated",
                "status": status
            })

        except Exception as exc:
            logger.exception(
                f"Background generation failed: {exc}"
            )

            sync_progress_callback({
                "target": generator_instance.target_count,
                "completed": generator_instance.completed_count,
                "rejected": generator_instance.rejected_count,
                "failed": generator_instance.failed_count,
                "remaining": max(
                    0,
                    generator_instance.target_count
                    - generator_instance.completed_count
                ),
                "is_paused": False,
                "is_cancelled": False,
                "current_stage": "Failed",
                "status": "error",
                "error": str(exc)
            })

    generator_thread = threading.Thread(
        target=run_wrapper,
        daemon=True
    )

    generator_thread.start()

    return {
        "status": "started",
        "config": config.model_dump(mode="json")
    }


@app.post("/api/pause")
async def pause_generation():
    if not active_generator or not (
        generator_thread and generator_thread.is_alive()
    ):
        raise HTTPException(
            status_code=400,
            detail="No active generation task."
        )

    active_generator.pause()
    return {"status": "paused"}


@app.post("/api/resume")
async def resume_generation():
    if not active_generator or not (
        generator_thread and generator_thread.is_alive()
    ):
        raise HTTPException(
            status_code=400,
            detail="No active generation task."
        )

    active_generator.resume()
    return {"status": "resumed"}


@app.post("/api/cancel")
async def cancel_generation():
    if not active_generator or not (
        generator_thread and generator_thread.is_alive()
    ):
        raise HTTPException(
            status_code=400,
            detail="No active generation task."
        )

    active_generator.cancel()
    return {"status": "cancelled"}


@app.get("/api/image/{sample_id}")
async def get_sample_image(sample_id: str):
    safe_sample_id = Path(sample_id).name
    config = load_generator_config()

    image_path = (
        Path(config.output_dir)
        / "images"
        / f"{safe_sample_id}.jpg"
    )

    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Image not found."
        )

    return FileResponse(
        str(image_path),
        media_type="image/jpeg"
    )


def start_server(
    host: str = "127.0.0.1",
    port: int = 8000
):
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )