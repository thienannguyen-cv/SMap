#!/usr/bin/env python
"""Rebuild the ephemeral audit sandbox from canonical source.

Run this BEFORE EVERY audit (``python audit/setup_sandbox.py``). It wipes
``audit/sandbox/`` and recreates it as a fresh, self-contained audit workspace
synced from the canonical packages, with the audit flag
(``smap.utils.DEBUG_FLAG``) forced ON inside the copy ONLY.

This removes the silent-drift risk: the audit never runs against a stale copy,
because the copy is regenerated from the single source of truth each time.

Canonical (never wiped):
  smap/  tools/          -- source of truth (mirror of GitHub; DEBUG_FLAG=False)
  audit/kit/probes/      -- audit instruments (t_*.py)
  audit/kit/data/        -- reference HDVO dumps (SMap_Z_10_*.npy)

Ephemeral (rebuilt on every run):
  audit/sandbox/
    smap/ tools/                 <- fresh copy of canonical code
    smap/utils.py                <- DEBUG_FLAG forced True  (== AUDIT MODE)
    t_*.py                       <- probes from audit/kit/probes/
    _reference_dumps/*.npy       <- reference dumps from audit/kit/data/
    tests/vtest_data/output/     <- where testbot (HDVO) writes fresh dumps

Usage:
    python audit/setup_sandbox.py
Then run any probe from inside the sandbox, e.g.:
    cd audit/sandbox && <python> t_a2.py
"""
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent   # project root (harness/audit/ -> ../..)
AUDIT = ROOT / "harness" / "audit"
SANDBOX = AUDIT / "sandbox"
KIT = AUDIT / "kit"
CODE_PKGS = ("smap", "tools")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def log(msg):
    print(f"[setup_sandbox] {msg}")


def main():
    # Guard: never run from inside the dir we are about to wipe.
    if Path.cwd().resolve() == SANDBOX or SANDBOX in Path.cwd().resolve().parents:
        sys.exit(f"ERROR: run setup_audit.py from outside {SANDBOX.name}/ (cwd is inside it)")

    # 1. Wipe the audit directory so every audit starts from a clean slate.
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX, ignore_errors=True)
    if SANDBOX.exists():
        # Windows may hold a handle on the top dir (e.g. another shell's cwd),
        # so rmdir of the top fails even though contents are removable. Clear any
        # remnants and reuse the existing (now empty) directory instead.
        for child in SANDBOX.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    os.unlink(child)
                except OSError:
                    pass
        log(f"reused {SANDBOX.name}/ (top dir locked; contents cleared)")
    else:
        SANDBOX.mkdir()
        log(f"wiped {SANDBOX.name}/")

    # 2. Copy the canonical packages (the single source of truth).
    for pkg in CODE_PKGS:
        src = ROOT / pkg
        if not src.is_dir():
            sys.exit(f"ERROR: canonical package '{pkg}/' not found at {src}")
        shutil.copytree(src, SANDBOX / pkg, ignore=IGNORE)
        log(f"copied canonical {pkg}/")

    # 3. Flip the audit flag ON in the copy ONLY (reuse the existing DEBUG flag).
    utils_py = SANDBOX / "smap" / "utils.py"
    text = utils_py.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r"^DEBUG_FLAG\s*=\s*\w+", "DEBUG_FLAG = True", text, count=1, flags=re.M
    )
    if n != 1:
        sys.exit("ERROR: could not locate 'DEBUG_FLAG = ...' in copied utils.py")
    utils_py.write_text(new_text, encoding="utf-8")
    log("set DEBUG_FLAG = True (AUDIT MODE) in sandbox copy")

    # 4. Bring in the audit instruments (probes).
    probes = sorted((KIT / "probes").glob("*.py"))
    for p in probes:
        shutil.copy2(p, SANDBOX / p.name)
    log(f"copied {len(probes)} probe script(s) from audit/kit/probes/")

    # 5. Bring in the reference HDVO dumps (kept apart from fresh outputs).
    ref_dir = SANDBOX / "_reference_dumps"
    ref_dir.mkdir()
    dumps = sorted((KIT / "data").glob("*.npy"))
    for p in dumps:
        shutil.copy2(p, ref_dir / p.name)
    log(f"copied {len(dumps)} reference dump(s) into _reference_dumps/")

    # 6. Create the testbot output dir (out_path default: ./tests/vtest_data/output/).
    out_dir = SANDBOX / "tests" / "vtest_data" / "output"
    out_dir.mkdir(parents=True)
    log("created tests/vtest_data/output/")

    # 7. Self-verify the resulting state.
    sandbox_debug = "DEBUG_FLAG = True" in utils_py.read_text(encoding="utf-8")
    root_debug_off = (
        "DEBUG_FLAG = False"
        in (ROOT / "smap" / "utils.py").read_text(encoding="utf-8")
    )
    assert (SANDBOX / "smap" / "smap.py").exists(), "sandbox smap.py missing"
    assert sandbox_debug, "sandbox DEBUG flag not set"
    if not root_debug_off:
        log("WARNING: canonical root smap/utils.py is not DEBUG_FLAG=False")
    log(
        f"VERIFY: sandbox DEBUG=True ({sandbox_debug}); "
        f"root DEBUG=False ({root_debug_off})"
    )
    log("audit sandbox ready.")


if __name__ == "__main__":
    main()
