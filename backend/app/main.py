import os
from urllib.parse import urlparse
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .api import router
from .storage_api import router as storage_router

app = FastAPI(title="BobTheBeatbuilder", version="0.1.0")
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver", "bob"]
)


@app.middleware("http")
async def local_requests(request: Request, call_next):
    from .api import jobs

    origin = request.headers.get("origin")
    if origin and urlparse(origin).netloc != request.headers.get("host"):
        return JSONResponse({"detail": "Cross-origin requests are disabled"}, status_code=403)
    uploading = (
        request.url.path in ("/api/audio", "/api/import", "/api/models/import")
        and request.method == "POST"
    )
    if uploading:
        with jobs.lock:
            jobs.uploads += 1
    try:
        return await call_next(request)
    finally:
        if uploading:
            with jobs.lock:
                jobs.uploads -= 1


@app.exception_handler(ValueError)
async def invalid_input(request, error):
    return JSONResponse({"detail": str(error)}, status_code=400)


@app.exception_handler(PermissionError)
async def inaccessible_path(request, error):
    return JSONResponse({"detail": "Folder is not accessible"}, status_code=403)


app.include_router(router)
app.include_router(storage_router)
frontend = Path(
    os.getenv("BOB_FRONTEND", Path(__file__).resolve().parents[2] / "frontend" / "dist")
)
if frontend.is_dir():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
