# Publishing guide — đăng bài lên GitHub / dev.to / Medium

Ba nền tảng xử lý Markdown **rất khác nhau**. Bảng dưới cho biết file nào dùng ở đâu và cần chỉnh gì.

| Nền tảng | Bảng (table) | Ảnh | Trạng thái file `.md` | Việc cần làm |
|---|---|---|---|---|
| **GitHub** (repo / Gist) | ✅ render tốt | ✅ đường dẫn tương đối `fig/*.png` chạy trong repo | **Sẵn sàng dán/đẩy nguyên trạng** | Đẩy cả thư mục `fig/` lên cùng repo |
| **dev.to** | ✅ render tốt | ⚠️ cần URL tuyệt đối hoặc upload | Cần **front-matter** ở đầu | Dán front-matter (bên dưới) + upload 2 ảnh, thay đường dẫn |
| **Medium** | ❌ **không hỗ trợ table** | ⚠️ phải upload thủ công | Cần xử lý bảng | Xem mục Medium bên dưới |

---

## 1. GitHub — sẵn sàng ngay

Đẩy các file `.md` + thư mục `fig/` lên repo. Đường dẫn ảnh tương đối (`fig/fig1-cost-crossover.png`) hoạt động trực tiếp. Bảng GFM render đẹp. Không cần chỉnh gì.

- Bản dài VI: `bai-luan-khoang-cach-kiem-chung.md`
- Bản dài EN: `synchronization-trap-essay-EN.md`
- Op-ed VI: `op-ed-bay-dong-bo-VI.md`

---

## 2. dev.to — dán front-matter rồi đăng

Chèn khối này vào **đầu** file (trước dòng `# tiêu đề`), rồi xoá dòng `# H1` trùng tiêu đề trong thân:

**Bản tiếng Anh:**
```yaml
---
title: "Code Got Cheap. Trust Didn't."
published: false
description: "The Synchronization Trap: software's unnamed gap in the AI era, and the verification discipline that closes it."
tags: ai, softwareengineering, architecture, productivity
cover_image: https://<raw-github-url>/fig/fig1-cost-crossover-en.png
---
```

**Bản tiếng Việt:**
```yaml
---
title: "Code đã rẻ. Niềm tin thì chưa."
published: false
description: "Bẫy Đồng bộ: khoảng cách chưa được gọi tên trong kỉ nguyên phần mềm AI, và kỷ luật kiểm chứng dùng để khép nó lại."
tags: ai, softwareengineering, architecture, productivity
cover_image: https://<raw-github-url>/fig/fig1-cost-crossover.png
---
```

**Ảnh trên dev.to:** hoặc (a) đẩy `fig/` lên GitHub rồi dùng URL `https://raw.githubusercontent.com/<user>/<repo>/main/preso/fig/fig1-cost-crossover.png` thay cho `fig/...`; hoặc (b) dùng nút upload ảnh trong trình soạn dev.to rồi dán URL nó trả về. (`tags`: tối đa 4, chữ thường, không dấu.)

---

## 3. Medium — bảng là vấn đề chính

Medium **không render bảng Markdown**. Hai lựa chọn:

- **Cách nhanh (khuyên dùng):** đăng bản đẹp nhất là từ **file PDF/`.docx`** — chụp ảnh từng bảng trong PDF (`synchronization-trap-essay-*.pdf`) và chèn như hình. Các bảng ở đây vốn cô đọng, làm-thành-ảnh đọc tốt.
- **Cách markdown:** dùng công cụ *import* của Medium từ một URL đã đăng (ví dụ bài GitHub/dev.to), rồi sửa tay các bảng (Medium sẽ biến chúng thành văn bản dính liền — phải chuyển thủ công thành danh sách hoặc ảnh).

**Ảnh trên Medium:** upload trực tiếp 2 file trong `fig/` (`fig1-cost-crossover[-en].png`, `fig2-fork[-en].png`) qua trình soạn. Đường dẫn tương đối không hoạt động.

> Gợi ý: nếu Medium là kênh chính, bản **op-ed** (ít bảng hơn) sẽ "lên hình" mượt hơn bản dài; giữ bản dài cho GitHub/dev.to.

---

## File typography (.docx/.pdf)

`reference-custom.docx` là *reference-doc* pandoc đã tuỳ biến (thân **Cambria** serif, tiêu đề **Calibri** đậm màu navy `#1E2761`, giãn dòng 1.15 — đều phủ tiếng Việt đầy đủ). Tạo lại bản `.docx` bất kỳ:

```bash
pandoc <file>.md -o <out>.docx --reference-doc=reference-custom.docx --resource-path=.
```
