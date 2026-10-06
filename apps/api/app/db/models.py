from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class Item(Base):
    __tablename__ = "items"

    item_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str | None] = mapped_column(Text)
    main_category: Mapped[str | None] = mapped_column(String(255), index=True)
    description: Mapped[str | None] = mapped_column(Text)
    features: Mapped[str | None] = mapped_column(Text)
    image_url: Mapped[str | None] = mapped_column(Text)
    price: Mapped[str | None] = mapped_column(String(64))
    average_rating: Mapped[float | None] = mapped_column(Float)
    rating_number: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class Interaction(Base):
    __tablename__ = "interactions"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "item_id",
            "event_timestamp_ms",
            name="uq_interaction_user_item_time",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"),
        index=True,
    )
    item_id: Mapped[str] = mapped_column(
        ForeignKey("items.item_id", ondelete="CASCADE"),
        index=True,
    )
    rating: Mapped[float | None] = mapped_column(Float)
    event_timestamp_ms: Mapped[int] = mapped_column(BigInteger, index=True)
    verified_purchase: Mapped[bool | None]
    review_title: Mapped[str | None] = mapped_column(Text)
    review_text: Mapped[str | None] = mapped_column(Text)
