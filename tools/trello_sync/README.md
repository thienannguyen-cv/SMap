# Trello bug sync

One-way sync of the SMap bug catalogue ([`active_bugs.md`](../../active_bugs.md))
into a Trello list. `active_bugs.md` is the **source of truth**; Trello mirrors
it. Edit bugs in the Markdown file — never in Trello.

## How it works

`sync_bugs.py` parses the bug table(s) in `active_bugs.md` and, for each bug,
creates or updates one Trello card:

| Bug field        | Trello card                                   |
| ---------------- | --------------------------------------------- |
| `id` + `title`   | Card name `[SMap Bug A] <title>`              |
| `severity`       | A coloured label (the bucket below)           |
| source / effect / fix | Card description (Markdown)              |

Each card carries a hidden marker `<!-- smap-bug-id: A -->` in its description.
Re-runs match on that marker, so the sync is **idempotent** — it updates the
existing card rather than creating duplicates, and only writes when something
actually changed.

Severity → label colour: `correctness`→red, `precision-only`→yellow,
`auditability`→sky, anything else→black (`other`). The sync owns these four
labels; other labels you add to a card by hand are left untouched.

## One-time setup

1. **Get a Trello API key + token.** Visit <https://trello.com/app-key> for the
   key, then generate a token with **write** scope from that page.
2. **Find the target list id.** Open the board, pick (or create) the list you
   want bugs to land in, then call:

   ```bash
   curl "https://api.trello.com/1/boards/<BOARD_SHORTLINK>/lists?key=<KEY>&token=<TOKEN>&fields=id,name"
   ```

   The board shortLink is the code in the board URL (e.g. `66d545d4e065eebded9a9c8f`).
   Alternatively skip the list id and provide `TRELLO_BOARD` + `TRELLO_LIST_NAME`.
3. **Add repository secrets** (Settings → Secrets and variables → Actions):
   - `TRELLO_KEY`
   - `TRELLO_TOKEN`
   - `TRELLO_LIST_ID` (preferred) — or `TRELLO_BOARD` plus a repo **variable**
     `TRELLO_LIST_NAME`.

## Running

- **Automatic:** the [`trello-sync`](../../.github/workflows/trello-sync.yml)
  workflow runs on every push to `main` that touches `active_bugs.md`.
- **On demand:** Actions tab → *Sync bugs to Trello* → *Run workflow*. Toggle
  `dry_run` to preview, or `archive_removed` to archive cards for deleted bugs.
- **Locally:**

  ```bash
  # Preview without touching Trello:
  python tools/trello_sync/sync_bugs.py --dry-run

  # Real sync:
  export TRELLO_KEY=... TRELLO_TOKEN=... TRELLO_LIST_ID=...
  python tools/trello_sync/sync_bugs.py
  ```

Stdlib only — no `pip install` required.

## Notes / limitations

- **One-way.** Changes made in Trello are not pushed back and will be
  overwritten on the next sync if they touch the synced fields.
- **Removed bugs** are left in place unless you pass `--archive-removed`.
- **Network.** This repo's hosted Claude Code web environment blocks outbound
  traffic to `api.trello.com`, so the sync is designed to run on GitHub-hosted
  runners (open egress) rather than from a web session.
