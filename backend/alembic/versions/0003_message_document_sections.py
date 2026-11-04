"""Add the document-section scope used for each message.

Revision ID: 0003_message_document_sections
Revises: 0002_message_tool_calls
Create Date: 2026-11-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0003_message_document_sections"
down_revision: str | None = "0002_message_tool_calls"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column(
            "document_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            server_default=sa.text("'{}'::uuid[]"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("messages", "document_ids")
