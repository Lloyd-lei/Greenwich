"""BodyDataStudio lifecycle and catalog bridge.

The source corpus is read-only. BodyDataStudio owns indexing, derived files,
labels, and export history in its cache root; AlphaMotion consumes that index
as a motion catalog without rewriting the original archives.
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import CONFIG


def _json(value: str | None) -> dict:
    try:
        result = json.loads(value or "{}")
        return result if isinstance(result, dict) else {}
    except (TypeError, ValueError):
        return {}


def _short_source(value: str) -> str:
    value = str(value or "").strip()
    if "/" in value:
        value = value.rsplit("/", 1)[-1].strip()
    return value.replace("AMASS ", "").strip()


def _open_catalog(path: Path) -> sqlite3.Connection | None:
    if not path.is_file():
        return None
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
        db.row_factory = sqlite3.Row
        return db
    except sqlite3.Error:
        return None


def _catalog_path(requested: Path | None = None) -> Path:
    """Prefer the separated cache, with the legacy index as read-only bridge.

    The first Linux full scan runs in the background. Until it has indexed the
    AMASS corpus, the former index still provides useful lineage metadata and
    is never opened writable by AlphaMotion.
    """
    primary = requested or CONFIG.data_studio_db
    legacy = Path(CONFIG.data_studio_root) / "_preview_cache" / \
        "bodydata_studio.sqlite3"
    for path in (primary, legacy):
        db = _open_catalog(path)
        if db is None:
            continue
        try:
            count = int(db.execute(
                "SELECT COUNT(*) FROM assets WHERE kind='smplh_motion'"
            ).fetchone()[0])
        except sqlite3.Error:
            count = 0
        finally:
            db.close()
        if count >= 8000 or path == legacy:
            return path
    return primary


def _label_index(cache_root: Path) -> dict[str, list[str]]:
    """Collect labels attached by completed processing runs.

    Labels remain sidecars. The catalog exposes their semantic type without
    copying them into source NPZ files.
    """
    labels: dict[str, set[str]] = defaultdict(set)
    process_root = cache_root / "process_runs"
    if process_root.is_dir():
        for path in process_root.glob("*/run.json"):
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            config = (run.get("config") or {}).get("labeling") or {}
            label_type = "foot_contact" if config.get("enabled") else ""
            for item in ((run.get("labeling") or {}).get("items") or []):
                asset_id = str(item.get("asset_id") or "")
                if asset_id and label_type:
                    labels[asset_id].add(label_type)
    return {key: sorted(value) for key, value in labels.items()}


def _catalog_records_from(path: Path) -> list[dict[str, Any]]:
    """Read one BodyDataStudio database into individual clip records."""
    db = _open_catalog(path)
    if db is None:
        return []
    try:
        rows = db.execute(
            """SELECT id,source,title,folder,kind,format,locator_type,
                      container,inner_path,animated,status,metadata_json
                 FROM assets
                WHERE kind='smplh_motion' AND animated=1 AND status='ready'"""
        ).fetchall()
        relation: dict[str, dict[str, Any]] = {}
        for table, kind, extra in (
            ("augmentation_outputs", "time", "duration_scale"),
            ("augmentation_bone_outputs", "bone_length", "variant_index"),
        ):
            try:
                query = (
                    f"SELECT original_asset_id,augmented_asset_id,{extra},"
                    f"validation_json FROM {table} "
                    "WHERE augmented_asset_id IS NOT NULL"
                )
                for row in db.execute(query):
                    relation[row["augmented_asset_id"]] = {
                        "origin_id": row["original_asset_id"],
                        "augmentation": kind,
                        "augmentation_value": row[extra],
                        "validation": _json(row["validation_json"]),
                    }
            except sqlite3.Error:
                continue
    finally:
        db.close()

    labels = _label_index(path.parent)
    records = []
    rows_by_id = {str(row["id"]): row for row in rows}
    for row in rows:
        metadata = _json(row["metadata_json"])
        rel = relation.get(row["id"])
        role = "augmented" if rel or metadata.get("augmentation_type") else "original"
        # BodyDataStudio creates decodable draft records while an augmentation
        # is being reviewed.  They are not part of the shared library until
        # the user saves/publishes them, so Motion Studio must not leak them.
        if role == "augmented" and metadata.get("augmentation_draft") \
                and not metadata.get("augmentation_saved"):
            continue
        origin_id = str((rel or {}).get("origin_id")
                        or metadata.get("parent_asset_id") or row["id"])
        augmentation = str((rel or {}).get("augmentation")
                           or metadata.get("augmentation_type") or "")
        augmentation_value = (rel or {}).get("augmentation_value")
        if augmentation_value is None:
            augmentation_value = (metadata.get("duration_scale")
                                  if augmentation == "time"
                                  else metadata.get("variant_index"))
        parent = rows_by_id.get(origin_id) if role == "augmented" else None
        source = str(parent["source"] if parent is not None else row["source"])
        records.append({
            "asset_id": row["id"],
            "origin_id": origin_id,
            "role": role,
            "source": source,
            "source_key": _short_source(source),
            "title": row["title"],
            "folder": row["folder"],
            "format": row["format"],
            "locator_type": row["locator_type"],
            "container": row["container"],
            "inner_path": row["inner_path"],
            "augmentation": augmentation,
            "augmentation_value": augmentation_value,
            "augmentation_metadata": metadata,
            "validation": (rel or {}).get("validation") or {},
            "labels": labels.get(row["id"], []),
            "variants": [],
        })
    children: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record["role"] == "augmented":
            children[record["origin_id"]].append(record["asset_id"])
    for record in records:
        record["variants"] = sorted(children.get(record["asset_id"], []))
    return records


def _smpl_count(path: Path) -> int:
    db = _open_catalog(path)
    if db is None:
        return 0
    try:
        return int(db.execute(
            "SELECT COUNT(*) FROM assets WHERE kind='smplh_motion'"
        ).fetchone()[0])
    except sqlite3.Error:
        return 0
    finally:
        db.close()


def catalog_records(db_path: Path | None = None) -> list[dict[str, Any]]:
    """Return individual SMPL-family records with lineage and labels.

    While the Linux index is being rebuilt, the legacy catalog remains the
    authoritative bridge. Once the new scan is complete, both catalogs are
    merged: duplicate originals collapse to one clip, historical published
    augmentations remain available, and new publications are appended.
    """
    if db_path is not None:
        return _catalog_records_from(Path(db_path))

    primary = Path(CONFIG.data_studio_db)
    legacy = Path(CONFIG.data_studio_root) / "_preview_cache" / \
        "bodydata_studio.sqlite3"
    if _smpl_count(primary) < 8000 or not legacy.is_file():
        return _catalog_records_from(_catalog_path())

    legacy_records = _catalog_records_from(legacy)
    primary_records = _catalog_records_from(primary)
    merged = [dict(record) for record in legacy_records]
    original_key: dict[tuple[str, str], dict[str, Any]] = {
        (record["source_key"].casefold(), record["title"].casefold()): record
        for record in merged if record["role"] == "original"
    }
    aliases: dict[str, str] = {}
    for record in primary_records:
        if record["role"] != "original":
            continue
        key = (record["source_key"].casefold(), record["title"].casefold())
        existing = original_key.get(key)
        if existing is not None:
            aliases[record["asset_id"]] = existing["asset_id"]
            existing["labels"] = sorted(set(existing["labels"])
                                        | set(record["labels"]))
            continue
        merged.append(dict(record))
        original_key[key] = merged[-1]
    known_ids = {record["asset_id"] for record in merged}
    for record in primary_records:
        if record["role"] != "augmented" or record["asset_id"] in known_ids:
            continue
        item = dict(record)
        item["origin_id"] = aliases.get(item["origin_id"], item["origin_id"])
        merged.append(item)
        known_ids.add(item["asset_id"])

    children: dict[str, list[str]] = defaultdict(list)
    for record in merged:
        record["variants"] = []
        if record["role"] == "augmented":
            children[record["origin_id"]].append(record["asset_id"])
    for record in merged:
        record["variants"] = sorted(children.get(record["asset_id"], []))
    return merged


def catalog_by_id(db_path: Path | None = None) -> dict[str, dict[str, Any]]:
    return {record["asset_id"]: record for record in catalog_records(db_path)}


def catalog_lookup(db_path: Path | None = None) -> dict[tuple[str, str], dict]:
    """Index records by source and title for existing encoded library shards."""
    result = {}
    for record in catalog_records(db_path):
        result[(record["source_key"].casefold(), record["title"].casefold())] = record
    return result


def catalog_summary(db_path: Path | None = None) -> dict[str, Any]:
    path = _catalog_path(db_path)
    records = catalog_records(db_path)
    roles = Counter(record["role"] for record in records)
    augmentations = Counter(record["augmentation"] for record in records
                            if record["augmentation"])
    labels = Counter(label for record in records for label in record["labels"])
    sources = Counter(record["source_key"] for record in records)
    return {
        "ready": bool(records),
        "database": str(path),
        "revision": path.stat().st_mtime_ns if path.is_file() else 0,
        "total": len(records),
        "roles": [{"id": key, "label": key.title(), "count": value}
                  for key, value in sorted(roles.items())],
        "augmentations": [
            {"id": key, "label": key.replace("_", " ").title(), "count": value}
            for key, value in sorted(augmentations.items())],
        "labels": [{"id": key, "label": key.replace("_", " ").title(),
                    "count": value} for key, value in sorted(labels.items())],
        "sources": [{"id": key, "label": key, "count": value}
                    for key, value in sources.most_common()],
    }


class DataStudioManager:
    """Start/stop the existing BodyDataStudio worker with AlphaMotion."""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.log_handle = None

    @staticmethod
    def _reachable() -> bool:
        try:
            with socket.create_connection(
                    ("127.0.0.1", CONFIG.data_studio_port), timeout=.25):
                return True
        except OSError:
            return False

    def start(self) -> dict[str, Any]:
        repo = Path(CONFIG.data_studio_repo)
        server = repo / "bodydata_server.py"
        cache = Path(CONFIG.data_studio_cache)
        if self._reachable():
            return {"ready": True, "managed": False, "port": CONFIG.data_studio_port}
        if not server.is_file():
            return {"ready": False, "error": f"BodyDataStudio missing: {server}"}
        cache.mkdir(parents=True, exist_ok=True)
        export_root = cache / "published"
        export_root.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "BODY_DATA_ROOT": CONFIG.data_studio_root,
            "BODY_DATA_CACHE": str(cache),
            "BODY_DATA_EXPORT_ROOT": str(export_root),
            "BODY_DATA_INTEGRATED": "1",
            "PYTHONUNBUFFERED": "1",
        })
        python = env.get("ALPHAMOTION_DATA_STUDIO_PYTHON", sys.executable)
        self.log_handle = (cache / "service.log").open("ab", buffering=0)
        self.process = subprocess.Popen(
            [python, str(server), "--host", "127.0.0.1", "--port",
             str(CONFIG.data_studio_port), "--full-index"],
            cwd=repo, env=env, stdout=self.log_handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        # A full corpus scan makes the first process start slower than the
        # lightweight health probe. Give the HTTP server enough time to bind
        # before reporting a false-negative in AlphaMotion's status panel.
        for _ in range(200):
            if self._reachable():
                return {"ready": True, "managed": True,
                        "port": CONFIG.data_studio_port}
            if self.process.poll() is not None:
                break
            time.sleep(.1)
        return {"ready": False, "managed": True,
                "error": "BodyDataStudio did not become ready; see service.log"}

    def status(self) -> dict[str, Any]:
        running = bool(self.process is not None and self.process.poll() is None)
        return {"ready": self._reachable(), "managed": running,
                "port": CONFIG.data_studio_port}

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
