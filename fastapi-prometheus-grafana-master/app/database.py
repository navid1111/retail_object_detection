"""Database models and setup."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import DATABASE_URL, DB_WAIT_SECONDS
from time import sleep


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class Prediction(Base):
    """Database model for predictions."""
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    image_path: Mapped[str] = mapped_column(String(512))
    annotated_image_path: Mapped[str] = mapped_column(String(512))
    model_name: Mapped[str] = mapped_column(String(255))
    inference_ms: Mapped[float] = mapped_column(Float)
    ground_truth_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    detections: Mapped[list["Detection"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan"
    )


class Detection(Base):
    """Database model for detections."""
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    class_id: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(255))
    confidence: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)

    prediction: Mapped[Prediction] = relationship(back_populates="detections")


# Database engine and session factory
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _relax_legacy_prediction_columns() -> None:
    """Keep old demo databases compatible with the retail prediction schema."""
    legacy_defaults = {
        "human_count": "INT NOT NULL DEFAULT 0",
        "car_count": "INT NOT NULL DEFAULT 0",
    }

    with engine.begin() as conn:
        existing_columns = {
            row[0]
            for row in conn.execute(
                text(
                    """
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = 'predictions'
                    """
                )
            )
        }

        for column_name, column_definition in legacy_defaults.items():
            if column_name in existing_columns:
                conn.execute(
                    text(
                        f"ALTER TABLE predictions "
                        f"MODIFY COLUMN {column_name} {column_definition}"
                    )
                )


def init_db() -> None:
    """Initialize the database and ensure it's ready."""
    attempts = max(1, DB_WAIT_SECONDS // 2)
    last_error: Exception | None = None

    for _ in range(attempts):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            Base.metadata.create_all(bind=engine)
            _relax_legacy_prediction_columns()
            return
        except Exception as exc:
            last_error = exc
            sleep(2)

    raise RuntimeError(f"MySQL is not ready after waiting {DB_WAIT_SECONDS}s: {last_error}")
