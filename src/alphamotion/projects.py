"""Project-local media workspaces for the online AlphaMotion editor.

The shared Data Studio catalog is a read-only source library.  Importing an
item creates a durable reference in a project; user uploads are stored inside
that project's media directory.  Motion Studio is scoped exclusively to this
manifest.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from .paths import data_dir


PROJECT_VERSION = 2
PROJECT_SUFFIX = ".alphamotion-project.json"
_ID = re.compile(r"^[a-f0-9]{32}$")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _clean_name(value: Any) -> str:
    name = " ".join(str(value or "").strip().split())
    if not name:
        raise ValueError("project name is required")
    return name[:120]


def _unique(values: list[Any], *, integer: bool = False) -> list[Any]:
    result, seen = [], set()
    for value in values:
        if integer:
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value < 0:
                continue
        else:
            value = str(value or "").strip()
            if not value:
                continue
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


class ProjectStore:
    """Atomic project documents plus a private media directory per project."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or (data_dir() / "projects"))
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, project_id: str) -> Path:
        if not _ID.fullmatch(str(project_id)):
            raise KeyError(project_id)
        return self.root / f"{project_id}{PROJECT_SUFFIX}"

    def media_dir(self, project_id: str, kind: str = "") -> Path:
        self._path(project_id)  # validates the identifier
        root = self.root / project_id / "media"
        path = root / kind if kind else root
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _read(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise KeyError(path.name.split(".", 1)[0]) from exc
        if not isinstance(value, dict):
            raise ValueError(f"invalid project document: {path.name}")
        return value

    def _write(self, project: dict[str, Any]) -> None:
        path = self._path(project["id"])
        temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(project, ensure_ascii=False, indent=2,
                                   sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)

    @staticmethod
    def _summary(project: dict[str, Any]) -> dict[str, Any]:
        assets = project.get("assets") or {}
        return {
            "id": project["id"], "name": project["name"],
            "created_at": project["created_at"],
            "updated_at": project["updated_at"],
            "motion_count": len(assets.get("motions") or []),
            "body_count": len(assets.get("bodies") or []),
            "autosave": bool(project.get("autosave", True)),
        }

    def list(self) -> list[dict[str, Any]]:
        projects = []
        with self._lock:
            for path in self.root.glob(f"*{PROJECT_SUFFIX}"):
                try:
                    projects.append(self._summary(self._read(path)))
                except (OSError, ValueError, KeyError):
                    continue
        return sorted(projects, key=lambda value: value["updated_at"],
                      reverse=True)

    def create(self, name: str, *, fps: float = 30.0) -> dict[str, Any]:
        now, project_id = _now(), uuid.uuid4().hex
        project = {
            "version": PROJECT_VERSION, "id": project_id,
            "name": _clean_name(name), "created_at": now,
            "updated_at": now, "autosave": True,
            "assets": {"motions": [], "bodies": [], "bins": []},
            "timeline": {"segments": [], "se3": [], "selected": -1,
                         "frame": 0, "fps": max(1.0, min(120.0, float(fps))),
                         "title": "", "target_body": "", "render_mp4": True},
            "ui": {},
        }
        with self._lock:
            self._write(project)
            self.media_dir(project_id, "motions")
            self.media_dir(project_id, "bodies")
        return copy.deepcopy(project)

    def get(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._read(self._path(project_id)))

    def save(self, project_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            project = self._read(self._path(project_id))
            if "name" in changes:
                project["name"] = _clean_name(changes["name"])
            if "autosave" in changes:
                project["autosave"] = bool(changes["autosave"])
            for key in ("timeline", "ui"):
                if isinstance(changes.get(key), dict):
                    project[key] = changes[key]
            project["updated_at"] = _now()
            self._write(project)
            return copy.deepcopy(project)

    def add_media(self, project_id: str, *, motions: list[dict] | None = None,
                  bodies: list[dict] | None = None,
                  bin_name: str = "") -> dict[str, Any]:
        with self._lock:
            project = self._read(self._path(project_id))
            assets = project.setdefault("assets", {})
            for kind, incoming, identity in (
                    ("motions", motions or [], "asset_id"),
                    ("bodies", bodies or [], "name")):
                current = list(assets.get(kind) or [])
                positions = {str(item.get(identity)): i for i, item in enumerate(current)}
                for item in incoming:
                    key = str(item.get(identity) or "").strip()
                    if not key:
                        continue
                    clean = copy.deepcopy(item)
                    clean.setdefault("imported_at", _now())
                    if key in positions:
                        current[positions[key]].update(clean)
                    else:
                        positions[key] = len(current)
                        current.append(clean)
                assets[kind] = current
            if bin_name:
                bins = list(assets.get("bins") or [])
                bins.insert(0, {"id": uuid.uuid4().hex[:12],
                                "name": _clean_name(bin_name),
                                "asset_ids": _unique([
                                    item.get("asset_id") for item in motions or []])})
                assets["bins"] = bins[:100]
            project["updated_at"] = _now()
            self._write(project)
            return copy.deepcopy(project)

    def remove_media(self, project_id: str, *, kind: str,
                     asset_id: str = "", library_id: int | None = None,
                     name: str = "") -> tuple[dict[str, Any], int]:
        """Remove project references without deleting shared/source assets."""
        normalized = {"motion": "motions", "motions": "motions",
                      "robot": "bodies", "body": "bodies",
                      "bodies": "bodies"}.get(str(kind).casefold())
        if normalized is None:
            raise ValueError("kind must be motion or robot")
        asset_id, name = str(asset_id or "").strip(), str(name or "").strip()
        if not asset_id and library_id is None and not name:
            raise ValueError("an asset selector is required")
        with self._lock:
            project = self._read(self._path(project_id))
            assets = project.setdefault("assets", {})
            current = list(assets.get(normalized) or [])

            def matches(item: dict[str, Any]) -> bool:
                return bool(
                    (asset_id and str(item.get("asset_id") or "") == asset_id) or
                    (library_id is not None and
                     item.get("library_id") is not None and
                     int(item["library_id"]) == int(library_id)) or
                    (name and str(item.get("name") or "") == name))

            removed_items = [item for item in current if matches(item)]
            assets[normalized] = [item for item in current if not matches(item)]
            if normalized == "motions" and removed_items:
                removed_ids = {str(item.get("asset_id") or "")
                               for item in removed_items}
                for value in assets.get("bins") or []:
                    value["asset_ids"] = [candidate for candidate in
                                          value.get("asset_ids") or []
                                          if str(candidate) not in removed_ids]
            if removed_items:
                project["updated_at"] = _now()
                self._write(project)
            return copy.deepcopy(project), len(removed_items)

    def motion_scope(self, project_id: str) -> tuple[set[int], set[str]]:
        motions = (self.get(project_id).get("assets") or {}).get("motions") or []
        indices = {int(item["library_id"]) for item in motions
                   if item.get("library_id") is not None}
        asset_ids = {str(item["asset_id"]) for item in motions
                     if item.get("asset_id")}
        return indices, asset_ids

    def body_scope(self, project_id: str) -> set[str]:
        bodies = (self.get(project_id).get("assets") or {}).get("bodies") or []
        return {str(item["name"]) for item in bodies if item.get("name")}
