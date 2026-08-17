"""Relational store (SQLite + SQLAlchemy 2.0, WAL).

Product variables (owner's spec): embodiment skeleton assets, motion FAMILY,
DURATION, and data SOURCE — the source axis is what turns user uploads into a
growing training corpus for future model versions.

Generated-motion writes go through the serialized job runner. Small metadata
writes for validated uploads and skeleton registration use short WAL
transactions in the API process.
"""
from __future__ import annotations

import datetime as _dt
import json

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer,
                        String, Text, create_engine, event)
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            relationship, sessionmaker)

from ..paths import db_path


class Base(DeclarativeBase):
    pass


def _now() -> _dt.datetime:
    return _dt.datetime.utcnow()


class Skeleton(Base):
    __tablename__ = "skeletons"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    kind: Mapped[str] = mapped_column(String(16))          # bundled|user_urdf|user_mjcf|human
    joints: Mapped[int] = mapped_column(Integer, default=0)
    height_cm: Mapped[float] = mapped_column(Float, default=0.0)
    xml_path: Mapped[str] = mapped_column(Text, default="")
    sem_labels: Mapped[dict] = mapped_column(JSON, default=dict)
    limit_report: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_now)


class Motion(Base):
    __tablename__ = "motions"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(256))
    family: Mapped[str] = mapped_column(String(32), index=True)
    duration_s: Mapped[float] = mapped_column(Float)
    fps: Mapped[float] = mapped_column(Float, default=30.0)
    n_frames: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(32), index=True)
    # library|text_prompt|video|ai_mixed|edit|user_upload
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    skeleton_id: Mapped[int | None] = mapped_column(
        ForeignKey("skeletons.id"), nullable=True)
    parent_motion_id: Mapped[int | None] = mapped_column(
        ForeignKey("motions.id"), nullable=True)
    tokens: Mapped[list | None] = mapped_column(JSON, nullable=True)
    trace_path: Mapped[str] = mapped_column(Text, default="")
    gate_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    gate_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    qc: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_now)


class Asset(Base):
    __tablename__ = "assets"
    id: Mapped[int] = mapped_column(primary_key=True)
    motion_id: Mapped[int | None] = mapped_column(
        ForeignKey("motions.id"), nullable=True)
    kind: Mapped[str] = mapped_column(String(16))   # trace|mp4|smpl|urdf|upload
    path: Mapped[str] = mapped_column(Text)
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_now)


class AtlasEdge(Base):
    __tablename__ = "atlas_edges"
    id: Mapped[int] = mapped_column(primary_key=True)
    src_motion_id: Mapped[int] = mapped_column(ForeignKey("motions.id"),
                                               index=True)
    src_slot: Mapped[int] = mapped_column(Integer)
    dst_window: Mapped[int] = mapped_column(Integer)
    dst_clip: Mapped[str] = mapped_column(String(256))
    dst_family: Mapped[str] = mapped_column(String(32))
    score: Mapped[float] = mapped_column(Float)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="queued",
                                        index=True)
    request: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    motion_id: Mapped[int | None] = mapped_column(ForeignKey("motions.id"),
                                                  nullable=True)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_now)
    finished_at: Mapped[_dt.datetime | None] = mapped_column(DateTime,
                                                             nullable=True)


_engine = None
_SessionLocal = None


def engine():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(f"sqlite:///{db_path()}",
                                connect_args={"check_same_thread": False})

        @event.listens_for(_engine, "connect")
        def _wal(dbapi, _rec):
            cur = dbapi.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()
        Base.metadata.create_all(_engine)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def session() -> Session:
    engine()
    return _SessionLocal()
