import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app import __version__
from app.api.deps import DbSession
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.services.storage import get_storage

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.validate_runtime()
    try:
        get_storage().ensure_bucket()
        logger.info("Object storage bucket %r is ready", settings.s3_bucket)
    except Exception as exc:
        # The API still boots (auth and metadata work) so operators can inspect
        # logs instead of staring at a crash loop.
        logger.warning("Object storage is not reachable yet: %s", exc)
    yield


app = FastAPI(
    title=settings.project_name,
    version=__version__,
    description=(
        "Asynchronous document ingestion and grounded RAG Q&A with scoped "
        "conversations, citations, and answer feedback."
    ),
    lifespan=lifespan,
    openapi_url=f"{settings.api_v1_prefix}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Document-Id"],
)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/", tags=["meta"], summary="Service metadata")
def root() -> dict[str, str]:
    return {
        "name": settings.project_name,
        "version": __version__,
        "environment": settings.environment,
        "docs": "/docs",
        "api": settings.api_v1_prefix,
    }


@app.get("/healthz", tags=["meta"], summary="Liveness and database check")
def healthz(db: DbSession) -> JSONResponse:
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        logger.exception("Database health check failed")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "database": "unavailable"},
        )
    return JSONResponse(status_code=200, content={"status": "ok", "database": "ok"})
