from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.db.base import Base
from app.main import app

EXPECTED_PATHS = {
    "/",
    "/healthz",
    "/api/v1/auth/register",
    "/api/v1/auth/token",
    "/api/v1/auth/me",
    "/api/v1/documents",
    "/api/v1/documents/{document_id}",
    "/api/v1/documents/{document_id}/reindex",
    "/api/v1/documents/{document_id}/download",
    "/api/v1/chat/sessions",
    "/api/v1/chat/sessions/{session_id}",
    "/api/v1/chat/sessions/{session_id}/ask",
}


def test_openapi_includes_phase_one_routes() -> None:
    spec = app.openapi()
    assert EXPECTED_PATHS <= set(spec["paths"])


def test_document_endpoints_require_bearer_auth() -> None:
    spec = app.openapi()
    upload = spec["paths"]["/api/v1/documents"]["post"]
    assert upload["security"] == [{"OAuth2PasswordBearer": []}]


def test_all_models_are_registered() -> None:
    expected = {
        "users",
        "documents",
        "chunks",
        "chat_sessions",
        "messages",
        "feedback",
    }
    assert expected <= set(Base.metadata.tables)


def test_model_metadata_compiles_for_postgres() -> None:
    dialect = postgresql.dialect()
    for table in Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=dialect))
        assert table.name in ddl
    chunks_ddl = str(CreateTable(Base.metadata.tables["chunks"]).compile(dialect=dialect))
    assert "GENERATED ALWAYS AS" in chunks_ddl
    assert "VECTOR(1536)" in chunks_ddl
