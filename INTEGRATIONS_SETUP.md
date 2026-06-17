# SMap Integrations — Setup & Handover

> Tài liệu bàn giao tổng hợp cho kế hoạch tích hợp **Discord + Trello** của SMap.
> Đọc file này trước; chi tiết kỹ thuật nằm trong [`docs/integrations.md`](docs/integrations.md)
> và [`tools/trello_sync/README.md`](tools/trello_sync/README.md).

Ngày bàn giao: 2026-06-17 · Branch: `claude/stoic-babbage-whas6g`

---

## 1. Kế hoạch & quyết định đã chốt

Thiết kế là **hybrid, in-repo** (mọi thứ chạm tới repo đều versioned trong repo để
contributor audit được):

- **Native Discord↔GitHub webhook** gánh "firehose" (push/PR/issue/comment).
- **GitHub Actions** (versioned) gánh các sự kiện high-signal: release, CI, bug.
- **Zapier** *giữ lại nhưng thu hẹp* — chỉ cho nguồn **không phải GitHub**
  (X/Twitter, RSS, Trello card moves). Tắt mọi Zap trùng với sự kiện GitHub.
- **#bugs ping**: dùng **secret riêng** `DISCORD_BUGS_WEBHOOK_URL`; chỉ ping khi
  `active_bugs.md` thực sự đổi trong push.

Nguyên tắc xuyên suốt: **một chủ sở hữu cho mỗi sự kiện** — không để hai cơ chế
cùng đăng một thứ.

---

## 2. Bản đồ kênh (channel map)

| Kênh Discord   | Nội dung | Cơ chế | Nằm ở |
| -------------- | -------- | ------ | ----- |
| `#dev-activity` | push, PR, issue, comment (firehose) | Native webhook | GitHub repo settings |
| `#releases`     | release published + link PyPI | `discord-release.yml` | repo |
| `#ci-status`    | Unit Tests / Lint / Commit Standards | `discord-ci.yml` (`workflow_run`) | repo |
| `#bugs`         | `active_bugs.md` đổi → Trello card + ping | `trello-sync.yml` | repo |
| _(non-GitHub)_  | X/RSS/Trello moves | Zapier (giữ) | Zapier |

---

## 3. Secrets & variables cần tạo

Vào **Settings → Secrets and variables → Actions**.

| Tên | Loại | Dùng bởi | Ghi chú |
| --- | ---- | -------- | ------- |
| `DISCORD_WEBHOOK_URL`      | secret | `discord-release.yml` | Webhook kênh `#releases` |
| `DISCORD_CI_WEBHOOK_URL`   | secret | `discord-ci.yml` | Webhook kênh `#ci-status` (đặt bằng cái trên nếu dùng chung 1 kênh) |
| `DISCORD_BUGS_WEBHOOK_URL` | secret | `trello-sync.yml` | Webhook kênh `#bugs` — **tùy chọn**, không đặt thì bước ping tự no-op |
| `TRELLO_KEY`               | secret | `trello-sync.yml` | <https://trello.com/app-key> |
| `TRELLO_TOKEN`             | secret | `trello-sync.yml` | token scope **write** |
| `TRELLO_LIST_ID`           | secret | `trello-sync.yml` | id list đích (ưu tiên) |
| `TRELLO_BOARD`             | secret | `trello-sync.yml` | thay thế: dùng cùng `TRELLO_LIST_NAME` |
| `TRELLO_LIST_NAME`         | **variable** | `trello-sync.yml` | tên list nếu không có `TRELLO_LIST_ID` |

> Webhook URL của Discord là **bearer credential** — ai có là đăng được. Giữ trong
> repo secrets, không echo ra log; nếu lộ thì xóa webhook trong Discord và tạo lại.
> Fork PR không đọc được repo secrets nên người ngoài không thấy URL.

---

## 4. Checklist triển khai (theo thứ tự)

1. **Native firehose webhook** (không cần code)
   - Discord: kênh `#dev-activity` → *Edit Channel → Integrations → Webhooks → New
     Webhook* → **Copy Webhook URL**.
   - GitHub: *Settings → Webhooks → Add webhook* → Payload URL =
     `<discord-webhook-url>/github` (**bắt buộc** hậu tố `/github`), Content type
     `application/json`, chọn events push/PR/issues/comments/releases.

2. **Webhook cho Actions** (`#releases`, `#ci-status`, `#bugs`)
   - Tạo webhook từng kênh **không** kèm `/github` (đây là raw webhook để Action POST tới).
   - Thêm 3 secret: `DISCORD_WEBHOOK_URL`, `DISCORD_CI_WEBHOOK_URL`,
     `DISCORD_BUGS_WEBHOOK_URL`.

3. **Trello**
   - Lấy `TRELLO_KEY` + `TRELLO_TOKEN` (write), tìm `TRELLO_LIST_ID`
     (xem `tools/trello_sync/README.md`), thêm secrets.

4. **Merge branch `claude/stoic-babbage-whas6g` vào `main`.**
   - `workflow_run` (CI) và trigger theo path chỉ chạy từ **bản trên default branch**,
     nên CI/bug notification chỉ hoạt động *sau khi merge*.
   - Mỗi workflow tự no-op nếu thiếu secret → merge trước khi có secret vẫn an toàn.

5. **Dọn Zapier**: tắt mọi Zap đang mirror sự kiện GitHub (đã được native/Actions phủ).
   Giữ lại Zap cho X/RSS/Trello moves.

---

## 5. Hành vi cần biết

- **Release:** đăng mỗi release published; pre-release tô màu/nhãn khác. Body cắt còn
  3500 codepoint (an toàn UTF-8 đa byte).
- **CI:** đăng **mọi failure**, nhưng chỉ đăng **success trên `main`** (cổng chống ồn —
  chỉnh `if` trong `discord-ci.yml`). Dùng `workflow_run` nên báo được cả CI của
  **fork PR** mà vẫn truy cập được secret.
- **Bugs:** sau khi sync Trello, `trello-sync.yml` ping `#bugs` **chỉ khi `active_bugs.md`
  thực sự đổi** trong push (so `git diff` theo dải push, bỏ qua run chỉ đổi
  `sync_bugs.py`/workflow). Chạy tay `workflow_dispatch` **không** ping. Không đặt
  `DISCORD_BUGS_WEBHOOK_URL` thì bỏ qua ping (Trello sync vẫn chạy).
- Các workflow chỉ dùng `curl` + `jq` (không marketplace action); `jq -n --arg`
  an toàn trước injection từ text release/commit tùy ý.

---

## 6. Giới hạn

- Discord webhook: ~30 msg/phút/webhook; ≤10 embed/msg; description ≤4096 ký tự.
- Webhook fail im lặng — kiểm tra Recent Deliveries (native) hoặc log Action
  (`curl -f` nên lỗi HTTP làm fail step).
- Trello sync **một chiều**: sửa trong Trello sẽ bị ghi đè ở lần sync sau. Bug bị xóa
  chỉ được archive khi truyền `--archive-removed`.
- Môi trường web của repo chặn egress tới `api.trello.com` → sync thiết kế để chạy
  trên GitHub-hosted runner (egress mở), không chạy từ web session.

---

## 7. Danh mục tệp trong gói

```
INTEGRATIONS_SETUP.md                   ← file này (tổng hợp)
docs/integrations.md                    ← tài liệu chi tiết
.github/workflows/discord-release.yml   ← release → #releases
.github/workflows/discord-ci.yml        ← CI → #ci-status (workflow_run)
.github/workflows/trello-sync.yml       ← active_bugs.md → Trello + ping #bugs
tools/trello_sync/sync_bugs.py          ← parser/sync (stdlib, idempotent)
tools/trello_sync/README.md             ← hướng dẫn Trello
smap-integrations.patch                 ← diff origin/main..branch (git apply)
```

Áp patch vào một bản clone sạch: `git apply smap-integrations.patch`
(hoặc copy trực tiếp các tệp trên vào đúng đường dẫn).
