from __future__ import annotations

import json
from datetime import datetime, timezone

from redis import Redis
from redis.exceptions import RedisError
from sqlalchemy import text

from app.core.config import settings
from app.db.session import engine


VALID_EVENTS = {"like", "save", "not_interested"}
EVENT_WEIGHTS = {
    "like": 0.8,
    "save": 1.2,
    "not_interested": -0.9,
}

RECENT_EVENT_LIMIT = 30
RECENT_EVENT_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days


class RealtimeFeedbackStore:
    def __init__(self) -> None:
        self.redis = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        self._table_ready = False

    def _redis_key(self, user_id: str) -> str:
        return f"inspirerank:realtime:{user_id}:events"

    def ensure_table(self) -> None:
        if self._table_ready:
            return

        statement = text(
            """
            CREATE TABLE IF NOT EXISTS realtime_interactions (
                id BIGSERIAL PRIMARY KEY,
                user_id VARCHAR(128) NOT NULL,
                item_id VARCHAR(64) NOT NULL,
                event_type VARCHAR(32) NOT NULL,
                source VARCHAR(32) NOT NULL DEFAULT 'web',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )

        index_statement = text(
            """
            CREATE INDEX IF NOT EXISTS
            ix_realtime_interactions_user_created
            ON realtime_interactions (user_id, created_at DESC)
            """
        )

        with engine.begin() as connection:
            connection.execute(statement)
            connection.execute(index_statement)

        self._table_ready = True

    def persist_event(
        self,
        user_id: str,
        item_id: str,
        event_type: str,
    ) -> None:
        self.ensure_table()

        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO realtime_interactions
                        (user_id, item_id, event_type, source)
                    VALUES
                        (:user_id, :item_id, :event_type, 'web')
                    """
                ),
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "event_type": event_type,
                },
            )

    def push_recent_event(
        self,
        user_id: str,
        item_id: str,
        event_type: str,
    ) -> bool:
        payload = {
            "user_id": user_id,
            "item_id": item_id,
            "event_type": event_type,
            "weight": EVENT_WEIGHTS[event_type],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        key = self._redis_key(user_id)

        try:
            pipe = self.redis.pipeline()
            pipe.lpush(key, json.dumps(payload))
            pipe.ltrim(key, 0, RECENT_EVENT_LIMIT - 1)
            pipe.expire(key, RECENT_EVENT_TTL_SECONDS)
            pipe.execute()
            return True
        except RedisError:
            return False

    def recent_events(
        self,
        user_id: str,
        limit: int = RECENT_EVENT_LIMIT,
    ) -> list[dict]:
        try:
            rows = self.redis.lrange(
                self._redis_key(user_id),
                0,
                max(0, limit - 1),
            )
        except RedisError:
            return []

        events = []

        for row in rows:
            try:
                event = json.loads(row)
            except json.JSONDecodeError:
                continue

            if (
                event.get("event_type") in VALID_EVENTS
                and event.get("item_id")
            ):
                events.append(event)

        return events

    def clear_recent_events(self, user_id: str) -> bool:
        try:
            self.redis.delete(self._redis_key(user_id))
            return True
        except RedisError:
            return False


_store: RealtimeFeedbackStore | None = None


def get_feedback_store() -> RealtimeFeedbackStore:
    global _store

    if _store is None:
        _store = RealtimeFeedbackStore()

    return _store
