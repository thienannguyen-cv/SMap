#!/usr/bin/env python3
"""Sync the SMap bug catalogue (active_bugs.md) into a Trello list.

One-way sync: ``active_bugs.md`` is the source of truth, Trello mirrors it.
Idempotent: each bug row is matched to a card by a hidden marker
(``<!-- smap-bug-id: X -->``) stored in the card description, so re-running
updates the existing card instead of creating a duplicate.

Stdlib only (urllib) so the GitHub Action needs no ``pip install``.

Configuration (environment variables):
  TRELLO_KEY         Trello API key                              (required to write)
  TRELLO_TOKEN       Trello API token with write scope           (required to write)
  TRELLO_LIST_ID     Target list id                              (preferred)
  TRELLO_BOARD       Board id or shortLink                       (used with TRELLO_LIST_NAME)
  TRELLO_LIST_NAME   Target list name on that board              (used with TRELLO_BOARD)

Without complete credentials the script runs in dry-run mode (parse + print).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.trello.com/1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUGS_FILE = os.path.join(REPO_ROOT, "active_bugs.md")

MARKER_RE = re.compile(r"<!--\s*smap-bug-id:\s*([A-Za-z0-9]+)\s*-->")
# A bug row's first cell looks like: **A — FUT missing convergence filter**
ROW_TITLE_RE = re.compile(r"^\*\*\s*([A-Za-z0-9]+)\s*[—–-]\s*(.+?)\s*\*\*$")

# Severity string -> (label name, Trello label colour). The sync owns these
# four "bucket" labels; any other labels on a card are left untouched.
BUCKET_NAMES = {"correctness", "precision-only", "auditability", "other"}


def severity_bucket(severity: str) -> tuple[str, str]:
    low = severity.lower()
    if "correctness" in low:
        return ("correctness", "red")
    if "precision" in low:
        return ("precision-only", "yellow")
    if "auditability" in low:
        return ("auditability", "sky")
    return ("other", "black")


def parse_bugs(path: str) -> list[dict]:
    """Extract bug rows from the Markdown table(s) in ``active_bugs.md``.

    The file holds more than one table block and the second block does not
    repeat the header, so we key off the row shape (a 5-cell row whose first
    cell is ``**<id> — <title>**``) rather than table position.
    """
    bugs: list[dict] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            stripped = raw.strip()
            if not stripped.startswith("|"):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if len(cells) < 5:
                continue
            m = ROW_TITLE_RE.match(cells[0])
            if not m:  # header row, separator row, or a non-bug table
                continue
            if len(cells) > 5:
                print(
                    f"WARN line {lineno}: row has {len(cells)} cells "
                    "(unescaped '|' inside a cell?); using the first 5",
                    file=sys.stderr,
                )
            bug_id = m.group(1).upper()
            if bug_id in seen:
                print(f"WARN line {lineno}: duplicate bug id {bug_id}; skipping", file=sys.stderr)
                continue
            seen.add(bug_id)
            bugs.append(
                {
                    "id": bug_id,
                    "title": m.group(2).strip(),
                    "severity": cells[1],
                    "source": cells[2],
                    "effect": cells[3],
                    "fix": cells[4],
                }
            )
    return bugs


def card_name(bug: dict) -> str:
    return f"[SMap Bug {bug['id']}] {bug['title']}"


def card_desc(bug: dict) -> str:
    return (
        f"**Severity:** {bug['severity']}\n\n"
        f"**Source anchor:** {bug['source']}\n\n"
        f"**Effect:** {bug['effect']}\n\n"
        f"**Fix direction:** {bug['fix']}\n\n"
        "---\n"
        "_Synced one-way from `active_bugs.md`. Edit the bug there, not here; "
        "do not remove the marker below._\n\n"
        f"<!-- smap-bug-id: {bug['id']} -->"
    )


def api(method: str, path: str, key: str, token: str, params: dict | None = None):
    query = dict(params or {})
    query["key"] = key
    query["token"] = token
    url = f"{API}{path}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SystemExit(f"Trello API {method} {path} failed: HTTP {exc.code} {detail}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Trello API {method} {path} unreachable: {exc.reason}")


def resolve_board_id(board: str, key: str, token: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{24}", board):
        return board
    info = api("GET", f"/boards/{board}", key, token, {"fields": "id,name"})
    return info["id"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Parse and print; never call Trello.")
    ap.add_argument(
        "--archive-removed",
        action="store_true",
        help="Archive cards whose bug id is no longer present in active_bugs.md.",
    )
    args = ap.parse_args()

    bugs = parse_bugs(BUGS_FILE)
    print(f"Parsed {len(bugs)} bug(s) from {BUGS_FILE}: {', '.join(b['id'] for b in bugs) or '(none)'}")
    if not bugs:
        print("No bug rows found — check the table format in active_bugs.md.", file=sys.stderr)
        return 1

    key = os.environ.get("TRELLO_KEY")
    token = os.environ.get("TRELLO_TOKEN")
    board = os.environ.get("TRELLO_BOARD")
    list_id = os.environ.get("TRELLO_LIST_ID")
    list_name = os.environ.get("TRELLO_LIST_NAME")

    creds_ok = bool(key and token and (list_id or (board and list_name)))
    if args.dry_run or not creds_ok:
        if not creds_ok and not args.dry_run:
            print(
                "Incomplete Trello credentials (need TRELLO_KEY, TRELLO_TOKEN, and "
                "TRELLO_LIST_ID or TRELLO_BOARD+TRELLO_LIST_NAME). Running dry-run.",
                file=sys.stderr,
            )
        for b in bugs:
            bucket, color = severity_bucket(b["severity"])
            print(f"\n=== Bug {b['id']}  (label: {bucket}/{color}) ===")
            print("name:", card_name(b))
            print("desc:")
            print(card_desc(b))
        return 0

    # Resolve the target list and its board.
    if not list_id:
        board_id = resolve_board_id(board, key, token)
        lists = api("GET", f"/boards/{board_id}/lists", key, token, {"fields": "id,name"})
        match = [lst for lst in lists if lst["name"].strip().lower() == list_name.strip().lower()]
        if not match:
            raise SystemExit(
                f"List {list_name!r} not found on board. Available: {[lst['name'] for lst in lists]}"
            )
        list_id = match[0]["id"]
    list_info = api("GET", f"/lists/{list_id}", key, token, {"fields": "idBoard,name"})
    board_id = list_info["idBoard"]
    print(f"Target list: {list_info['name']!r} ({list_id}) on board {board_id}")

    # Labels: load existing, create buckets on demand.
    labels = api(
        "GET", f"/boards/{board_id}/labels", key, token, {"fields": "id,name,color", "limit": "1000"}
    )
    label_by_name = {lab["name"]: lab for lab in labels if lab.get("name")}

    def ensure_label(name: str, color: str) -> str:
        if name in label_by_name:
            return label_by_name[name]["id"]
        created = api("POST", "/labels", key, token, {"name": name, "color": color, "idBoard": board_id})
        label_by_name[name] = created
        return created["id"]

    # Existing cards, indexed by marker.
    cards = api("GET", f"/lists/{list_id}/cards", key, token, {"fields": "id,name,desc,idLabels"})
    by_marker: dict[str, dict] = {}
    for card in cards:
        found = MARKER_RE.search(card.get("desc", ""))
        if found:
            by_marker.setdefault(found.group(1).upper(), card)

    for b in bugs:
        bucket, color = severity_bucket(b["severity"])
        label_id = ensure_label(bucket, color)
        name, desc = card_name(b), card_desc(b)
        existing = by_marker.get(b["id"])
        if existing:
            current = existing.get("idLabels", [])
            severity_ids = {label_by_name[n]["id"] for n in BUCKET_NAMES if n in label_by_name}
            desired = [l for l in current if l not in severity_ids]  # keep non-severity labels
            if label_id not in desired:
                desired.append(label_id)
            labels_changed = set(current) != set(desired)
            if existing["name"] != name or existing.get("desc", "") != desc or labels_changed:
                api(
                    "PUT",
                    f"/cards/{existing['id']}",
                    key,
                    token,
                    {"name": name, "desc": desc, "idLabels": ",".join(desired)},
                )
                print(f"UPDATED  Bug {b['id']} -> card {existing['id']}")
            else:
                print(f"UNCHANGED Bug {b['id']} -> card {existing['id']}")
        else:
            created = api(
                "POST",
                "/cards",
                key,
                token,
                {"idList": list_id, "name": name, "desc": desc, "idLabels": label_id},
            )
            print(f"CREATED  Bug {b['id']} -> card {created['id']}")

    if args.archive_removed:
        present = {b["id"] for b in bugs}
        for bid, card in by_marker.items():
            if bid not in present:
                api("PUT", f"/cards/{card['id']}", key, token, {"closed": "true"})
                print(f"ARCHIVED stale Bug {bid} -> card {card['id']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
