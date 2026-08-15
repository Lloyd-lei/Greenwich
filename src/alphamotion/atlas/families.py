"""Motion family classification by clip/prompt name (vendored FAM_RE)."""
from __future__ import annotations

import re

FAM_RE = [
    ("jacks", re.compile(r"jumping_?jack|jumping jack", re.I)),
    ("run",   re.compile(r"(?<![a-z])run|jog|sprint", re.I)),
    ("walk",  re.compile(r"walk|stroll|march", re.I)),
    ("dance", re.compile(r"danc|salsa|cha_?cha|tango|ballet|samba|zeibekiko|"
                         r"maleviziotikos", re.I)),
    ("kick",  re.compile(r"kick", re.I)),
    ("punch", re.compile(r"punch|hook|jab|backfist|uppercut|boxing", re.I)),
    ("jump",  re.compile(r"jump|hop|leap", re.I)),
    ("stand", re.compile(r"stand|still|idle|posing|t_?pose", re.I)),
    ("sit",   re.compile(r"\bsit|chair|crouch|squat", re.I)),
    ("throw", re.compile(r"throw|toss|catch", re.I)),
]
FAMILIES = [f for f, _ in FAM_RE] + ["other"]


def family_of(name: str) -> str:
    for fam, rx in FAM_RE:
        if rx.search(name):
            return fam
    return "other"


def family_id(name: str) -> int:
    return FAMILIES.index(family_of(name))
