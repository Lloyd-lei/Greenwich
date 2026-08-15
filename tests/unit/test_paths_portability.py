"""Release blocker: no research-machine literals anywhere in the package."""
from pathlib import Path

SRC = Path(__file__).parent.parent.parent / "src"
FORBIDDEN = ("/var/tmp", "be water", "arenalabs")


def test_no_forbidden_paths():
    bad = []
    for p in SRC.rglob("*.py"):
        text = p.read_text(errors="ignore")
        for f in FORBIDDEN:
            if f in text:
                bad.append(f"{p.name}: {f}")
    assert not bad, bad
