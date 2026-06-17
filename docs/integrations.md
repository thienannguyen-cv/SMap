# SMap Integrations (Discord / Trello)

How project activity reaches external channels. The design is **hybrid and
in-repo**: GitHub's native webhook carries the high-volume firehose, versioned
GitHub Actions carry curated high-signal events, and a SaaS automation (Zapier)
is kept only for **non-GitHub** sources. Everything that touches the repo lives
here so contributors can audit and improve it.

## Channel map

| Discord channel | Content | Mechanism | Lives in |
| --------------- | ------- | --------- | -------- |
| `#dev-activity` | push, PRs, issues, comments (firehose) | **Native** Discord↔GitHub webhook | GitHub repo settings |
| `#releases`     | published release + PyPI link | Action `discord-release.yml` (`release: published`) | repo |
| `#ci-status`    | Unit Tests / Lint / Commit Standards results | Action `discord-ci.yml` (`workflow_run: completed`) | repo |
| `#bugs`         | `active_bugs.md` changes → Trello cards | Action `trello-sync.yml` (+ optional Discord ping) | repo |
| _(non-GitHub)_  | X/RSS/Trello card moves, etc. | **Zapier** (kept) | Zapier |

**One owner per event** — never let two mechanisms post the same thing. Native
owns the firehose; Actions own releases and CI; Zapier owns only sources GitHub
cannot emit.

## Secrets & variables

| Name | Type | Used by | Notes |
| ---- | ---- | ------- | ----- |
| `DISCORD_WEBHOOK_URL`    | secret | `discord-release.yml` | Webhook of `#releases` |
| `DISCORD_CI_WEBHOOK_URL` | secret | `discord-ci.yml` | Webhook of `#ci-status` (set equal to the above to use one channel) |
| `TRELLO_KEY` / `TRELLO_TOKEN` / `TRELLO_LIST_ID` | secret | `trello-sync.yml` | See `tools/trello_sync/README.md` |

A Discord webhook URL is a **bearer credential** — anyone holding it can post.
Keep it in repo secrets, never echo it in logs, and if it leaks, delete the
webhook in Discord and create a new one. Fork PRs cannot read repo secrets, so
the URL is never exposed to outside contributors.

## Setup — native firehose webhook (no code)

1. **Discord:** target channel → *Edit Channel → Integrations → Webhooks → New
   Webhook* → **Copy Webhook URL**.
2. **GitHub:** repo *Settings → Webhooks → Add webhook*
   - **Payload URL:** `<discord-webhook-url>/github`  ← the `/github` suffix is required
   - **Content type:** `application/json`
   - **Events:** *Let me select individual events* → push, pull requests, issues,
     issue comments, releases (pick what you want in the firehose)
   - **Active:** on
3. Test by pushing / opening an issue. Debug via the webhook's **Recent
   Deliveries** panel.

## Setup — curated Actions

1. Create webhooks for `#releases` and `#ci-status` (same steps as above, **without**
   the `/github` suffix — these are raw Discord webhooks the Action posts to).
2. Add `DISCORD_WEBHOOK_URL` and `DISCORD_CI_WEBHOOK_URL` as repo secrets.
3. Merge `discord-release.yml` and `discord-ci.yml` to `main`.
   - `workflow_run` only fires for the copy on the **default branch**, so CI
     notifications start working *after* merge.
   - Each workflow no-ops cleanly if its secret is unset, so it is safe to merge
     before the secrets exist.

### Behaviour notes
- **Release:** posts on every published release; pre-releases are coloured/labelled
  differently. Body is trimmed to 3500 codepoints (safe for multibyte UTF-8).
- **CI:** posts every **failure**, but only **successes on `main`** (noise gate —
  adjust the `if` in `discord-ci.yml`). Uses `workflow_run`, so it also reports CI
  for **fork PRs** while still having access to the webhook secret.
- The workflows use `curl` + `jq` only (no marketplace action). `jq -n --arg`
  is injection-safe against arbitrary release/commit text.

## Zapier (kept, scoped down)

Zapier is retained **only** for sources GitHub cannot emit (e.g. posting releases
to X/Twitter, RSS, Trello card movements). **Disable any Zap that mirrors a
GitHub event** once the native webhook / Actions cover it, to avoid double-posting.
If multi-source automation grows, consider migrating these flows to a
self-hosted **n8n** (open-source, no per-task cost, no SaaS lock-in) for a more
sustainable footprint.

## Limits to keep in mind
- Discord webhook: ~30 messages/min per webhook; ≤10 embeds/message; embed
  description ≤4096 chars.
- Webhook failures are silent — check Recent Deliveries (native) or the Action
  run log (curl uses `-f`, so an HTTP error fails the step).
- Keep each workflow's `permissions:` minimal (`contents: read`).
