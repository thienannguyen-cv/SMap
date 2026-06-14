# Kế hoạch Tích hợp Cơ chế Zoom vào smap_simulator.jsx (bản cập nhật)

Plan này áp dụng lên hai file vừa được khôi phục: `smap_simulator.jsx` (802 dòng, fork hiện tại) và `smap_simulator_spec.md`. Đây là **plan cho thay đổi tiếp theo**, chưa phải code.

---

## Phần 0: Trả lời câu hỏi nghiệm thu (đọc mã nguồn, có trích dòng)

Đây là phần bắt buộc để chứng minh hiểu mã nguồn trước khi sửa.

### 0.1 "Vị trí hiển thị KHÔNG phải lúc nào cũng là phép chiếu xyz" — đúng, và đây là lý do

Vị trí hiển thị của một điểm = **ô (h, w) trên lưới** nơi điểm đó hiện diện. Vị trí này do **routing quyết định**, và routing là **detached + rời rạc + giới hạn trong 3×3**:

```
smap.py L45:  ind = torch.max(-key_query, dim=2, keepdim=True).indices
smap.py L46:  ind_mask = F.one_hot(ind, num_classes=3*3)...
smap.py L47:  weights_b = where(ind_mask>0, ones, zeros)
```

`ind` chọn neighbor (trong 9 ô của 3×3) có `key_query` nhỏ nhất — tức ô gần nhất với nơi xyz muốn điểm đến. Nhưng phép chọn này **chỉ chọn được trong 3×3**, nên mỗi forward pass tại một level, điểm dịch **tối đa 1 ô** ở độ phân giải của level đó.

Phép chiếu xyz (`query_x, query_y` từ `to_3d3x3`) là đại lượng **liên tục** chỉ ra nơi điểm "muốn" đến — có thể rất xa. Vị trí hiển thị là đại lượng **rời rạc** chỉ tiến về phía đó từng-ô-một mỗi level.

→ **Hai thứ này khác nhau:** vị trí hiển thị là xấp xỉ rời rạc, TRỄ so với phép chiếu liên tục của xyz.

**Trả lời gợi ý 1 ("nếu vị trí hiển thị lúc nào cũng là phép chiếu xyz thì còn cần zoom làm gì?"):** Nếu vị trí hiển thị = phép chiếu, điểm sẽ "teleport" tức thì đến nơi xyz chỉ, và routing/zoom thành vô nghĩa. Thực tế routing chỉ cho dịch 1 ô lân cận mỗi level. Cơ chế zoom tồn tại CHÍNH XÁC để cho phép vị trí hiển thị (rời rạc, giới hạn lân cận) **bắt kịp** phép chiếu (liên tục, có thể xa): tại level l, một bước-lân-cận phủ 2^l pixel ở độ phân giải gốc. Không có zoom → điểm chỉ dịch 1 pixel/pass → không bao giờ tới target xa.

### 0.2 "detach KHÔNG có nghĩa là xyz không bao giờ nhận gradient" — xyz nhận gradient ở đâu, khi nào

Detach trong `SMap3x3.forward` chỉ áp lên **trọng số chọn ô** (selection weights), KHÔNG áp lên **giá trị** xyz:

```
smap.py L130:  new_x_z_value = einsum('bcsthw,bcstzhw->bcstzhw', weights_x.detach()..., x_z_value)
smap.py L132:  new_r_mask    = einsum('bcsthw,bcstzhw->bcstzhw', weights_m.detach()..., r_mask)
```

`weights_x.detach()` và `weights_m.detach()` → gradient KHÔNG chảy qua "chọn ô nào". Nhưng `x_z_value` (chứa x,y,z) và `r_mask` (chứa m) KHÔNG bị detach → gradient VẪN chảy qua **giá trị**.

**Nguồn gradient cho xyz nằm trong `calculate_key_query`** (utils.py):

```
utils.py L174:  grouped_key_x = x_value + zeros_like(...)        # KHÔNG detach → x_value nhận gradient
utils.py L175:  grouped_key_y = y_value + zeros_like(...)        # KHÔNG detach → y_value nhận gradient
utils.py L176:  query_x = (updated_key_z[...]).detach()...        # DETACH (phép chiếu back-projection)
utils.py L177:  query_y = (updated_key_z[...]).detach()...        # DETACH
utils.py L180:  diff_x = sign(grouped_key_x - query_x).detach() * (grouped_key_x - query_x)
utils.py L181:  diff_y = sign(grouped_key_y - query_y).detach() * (grouped_key_y - query_y)
utils.py L182:  key_query = diff_x + diff_y
```

`diff_x` là dạng `|grouped_key_x − query_x|` (vì `sign(.).detach()` × hiệu): gradient của nó với `x_value` là hằng số `sign`, đẩy `x_value` về phía `query_x` (phép chiếu từ z). Vậy **xyz nhận gradient qua key_query**, mục tiêu là làm cho tọa độ lưu trữ khớp với phép chiếu back-project từ depth.

**Tại sao m cũng nhận gradient dù cơ chế "khá giống":** m nhận gradient qua `pre_mask_grdf` / `weights_grdf` trong `rectificate_flow`, dùng cùng kỹ thuật straight-through `(grdf − grdf.detach())`. Khác biệt là **nguồn**: xyz lấy gradient từ `key_query_grdf` (sai khác vị trí–phép chiếu), m lấy từ `pre_mask_grdf` (sai khác trạng thái active–target). Cùng cơ chế truyền (straight-through), khác tín hiệu nguồn.

**Trả lời gợi ý 2:** detach chỉ chặn gradient qua việc-chọn-ô (routing), không chặn gradient qua giá-trị (xyz, m). xyz được tối ưu qua `key_query` (utils.py L174-182), m được tối ưu qua `pre_mask_grdf` (rectify.py rectificate_flow). Việc tối ưu là có thật và cần thiết — nó điều chỉnh xyz để phép chiếu tiến về target, sau đó routing+zoom mới thực sự dời ô hiển thị theo.

### 0.3 Hệ quả cho mô phỏng

Mô phỏng phải tách bạch HAI thứ:
1. **Trạng thái điểm** `(x, y, z, m)` ở lưới gốc — được gradient cập nhật (x,y,z qua key_query, m qua mask flow).
2. **Vị trí hiển thị** (ô nào, channel nào, ở level nào) — do routing+zoom quyết định, là hàm rời rạc của phép chiếu xyz, dịch tối đa 1 ô lân cận mỗi level.

Fork hiện tại **gộp hai thứ này làm một** (chỉ có m scalar, ô cố định) — đó là vì sao nó tương đương zoom=0 và chưa thể hiện được sự di chuyển.

---

## Phần 1: Sai lệch của fork hiện tại so với yêu cầu

| # | Fork hiện tại (802 dòng) | Yêu cầu | Mức độ |
|---|--------------------------|---------|--------|
| F1 | Chỉ có `m` scalar mỗi ô | Cần `(x, y, z, m)` đầy đủ (chúng đi cùng nhau trong mã nguồn) | Sai cốt lõi |
| F2 | Lưới 8×8 phẳng, một level | Tích hợp zoom NGAY trên cùng màn: kích cỡ ô đổi theo level | Thiếu |
| F3 | Không có trượt channel | Trượt giữa các channel `C_zoom` (dim=1) | Thiếu |
| F4 | `selected = {r, c}` cố định trên lưới hiển thị | Selected = điểm VẬT LÝ `(orig_r, orig_c)`, hiện ở ô/channel khác nhau theo level | Sai ngữ nghĩa |
| F5 | `dirX/dirY` chỉ là mũi tên tới target lân cận | Mũi tên theo gradient (x,y) thực | Cần sửa |
| F6 | Cập nhật `m += lr×grad` | Cập nhật xyz (qua key_query) + m; routing quyết định ô | Sai cốt lõi |
| F7 | (không có) | Bất biến chống corrupt khi spam trượt zoom | Thiếu |

---

## Phần 2: Nguyên tắc thiết kế chống corrupt (F7 — yêu cầu quan trọng nhất)

**Single Source of Truth (SST) = lưới điểm ở độ phân giải GỐC (finest).** Mọi level zoom chỉ là **view phái sinh**, tính bằng FOLD on-the-fly khi render. Trượt zoom = đổi tham số view (`zoomLevel`, `channelIdx`), TUYỆT ĐỐI không ghi lại state.

Lý do bất biến này đúng (trích mã nguồn):
- FOLD (smap.py L216): `reshape → permute → reshape` — bijection thuần túy, không mất mát.
- UNFOLD (smap.py L235): nghịch đảo chính xác của FOLD.
- Vì FOLD∘UNFOLD = identity, quay về level cũ luôn cho đúng vị trí ban đầu.

→ Nếu app lưu state riêng từng level (sai), reshape lặp lại sẽ tích lũy lệch → điểm trượt mất kiểm soát khi spam. Lưu SST ở finest + view phái sinh → spam trượt không đổi state → không corrupt. Đây là bất biến app phải bảo vệ và test được.

---

## Phần 3: Mô hình dữ liệu mới (theo quyết định: giữ đơn giản, không thêm batch)

```js
// SST — lưới điểm ở finest resolution (FINE_H × FINE_W, lũy thừa của 2)
points[r][c] = { x, y, z, m }   // r,c ∈ [0, FINE_H/W), tọa độ GỐC
target[r][c] ∈ {0, 1}           // ở finest resolution

// View state (KHÔNG phải data — chỉ tham số nhìn)
zoomLevel    // l ∈ [0, L], L = log2(FINE_H)
channelIdx   // ∈ [0, 4^l)
selectedOrig // {r, c} — điểm VẬT LÝ được chọn (tọa độ GỐC, bất biến qua zoom)
mode, lrMode, lr  // giữ nguyên từ fork
```

`FINE_H = FINE_W = 8` (≥ kích thước lưới fork cũ), `L = log2(8) = 3`. Không thêm batch_size.

### Ánh xạ FOLD (view tại level l), khớp smap.py L216 + app_spec.md:
```
Tại level l: lưới hiển thị có (FINE_H/2^l) × (FINE_W/2^l) ô, và 4^l channels.
Ô (ph, pw) ở channel k của level l ↔ pixel gốc:
    sub_r = k // 2^l ,  sub_c = k % 2^l
    orig_r = ph * 2^l + sub_r
    orig_c = pw * 2^l + sub_c
```

### Channel của điểm selected (để giữ viền đỏ theo điểm vật lý — F4):
```
position_h = orig_r // 2^l ;  position_w = orig_c // 2^l
sub_r = orig_r % 2^l ;  sub_c = orig_c % 2^l
channelOfSelected = sub_r * 2^l + sub_c
```
Viền đỏ chỉ hiện khi `channelIdx === channelOfSelected`. Nếu đang ở channel khác → hiện chỉ báo "Điểm đang ở channel K, level l — trượt để thấy".

---

## Phần 4: Cơ chế di chuyển trong mô phỏng (theo mã nguồn, tách 2 loại)

### 4.1 Loại A — Gradient cập nhật xyz (KHÔNG đổi ô), mỗi bước Markov:
```
key_query_x = sign(x - proj_x(z, pixel)) → gradient đẩy x về proj_x   [utils.py L174,180]
key_query_y = sign(y - proj_y(z, pixel)) → gradient đẩy y về proj_y   [utils.py L175,181]
m_grad từ mask flow (giữ logic fork hiện tại + filter CURRENT/CORRECT)
update: x += lr×gx ; y += lr×gy ; z += lr×gz ; m += lr×gm   (batch, cùng lr)
```
Phép chiếu `proj(z, pixel)` dùng camera identity (app_spec.md): `proj_x = z × x_im`, `proj_y = z × y_im`, với `(x_im, y_im)` là tọa độ pixel chuẩn hóa của ô. Inspector hiển thị công thức + ma trận camera (identity, dùng chung toàn cục — smap.py L15).

### 4.2 Loại B — Routing dời ô hiển thị (KHÔNG gradient), khi chuyển/chạy zoom:
```
Tại level l, mỗi điểm xét 3×3 lân cận, chọn ô có key_query nhỏ nhất [smap.py L45].
Điểm dời TỐI ĐA 1 ô ở độ phân giải level l = 2^l pixel gốc.
```
Khi user chạy "zoom step" ở level l, mô phỏng áp routing tại level đó: điểm di chuyển về ô lân cận gần phép chiếu xyz hơn. Vì shape ở level cao nhỏ hơn (đã ÷2 mỗi chiều), 1 bước ở level cao = bước xa ở level thấp.

**Quan trọng:** Loại B chỉ thay đổi **vị trí hiển thị suy ra từ SST**, không ghi đè SST. Vị trí hiển thị = hàm của xyz qua routing, tính lại mỗi render. Spam trượt không tích lũy lỗi (Phần 2).

---

## Phần 5: Giữ + thích nghi cơ chế cũ

| Cơ chế cũ (fork) | Giữ/Sửa | Cách thích nghi |
|------------------|---------|-----------------|
| CURRENT/CORRECT toggle | Giữ | Áp cho m-flow filter như cũ |
| Meta-state coloring | Giữ | Tính ở finest, hiển thị theo view level |
| Per-step commentary | Giữ + mở rộng | Thêm thông tin level, channel, channelOfSelected |
| Mũi tên hướng (dirX/dirY) | Sửa | Đổi từ "hướng tới target" sang gradient (x,y) thực; xoay theo (gx, gy) |
| Inspector m editor | Giữ + mở rộng | Thêm sliders x, y, z |
| Manual/Auto LR | Giữ | Không đổi |
| Batch update | Giữ | Cập nhật mọi điểm (mọi channel/level) cùng lr |
| 3 presets | Giữ | Default = response.txt config, ở finest |

---

## Phần 6: UI mới

- **Zoom slider/buttons**: chọn `zoomLevel` ∈ [0, 3]. Hiển thị "Level l: (H/2^l)×(W/2^l) ô, 4^l channels".
- **Channel slider**: chọn `channelIdx` ∈ [0, 4^l). Hiển thị "Channel k / 4^l". Khi l=0, ẩn (chỉ 1 channel).
- **Grid**: kích cỡ ô tự co/giãn theo số ô của level (CSS grid-template-columns động).
- **Selected indicator**: viền đỏ theo điểm vật lý; nếu ở channel khác → banner "trượt đến channel K".
- **Inspector**: thêm x, y, z sliders + công thức chiếu + camera matrix. Giữ m, t.
- **Zoom step button**: áp routing 1 bước tại level hiện tại (Loại B), xem điểm dời ô.

---

## Phần 7: Checklist mới (thêm vào reasoning.md)

| # | Tiêu chí | Cách thỏa mãn |
|---|----------|---------------|
| C21 | Zoom tích hợp cùng màn, ô co giãn | zoomLevel + grid động, không tạo view tách biệt |
| C22 | Trượt channel C_zoom | channelIdx slider; tên mã nguồn: C_zoom (4^l), C_zoom_2 (2^l), dim=1 |
| C23 | Selected = điểm vật lý qua mọi zoom | selectedOrig (orig_r,orig_c) + channelOfSelected |
| C24 | Chống corrupt khi spam trượt | SST ở finest, view phái sinh, FOLD/UNFOLD bijection |
| C25 | Thêm x,y,z bên cạnh m | points[r][c]={x,y,z,m}, gradient cập nhật cả 4 |
| C26 | Vị trí hiển thị ≠ phép chiếu xyz | Routing rời rạc giới hạn 3×3 (L45), trễ so với phép chiếu liên tục |
| C27 | xyz nhận gradient (detach chỉ chặn selection) | key_query L174-182, einsum giá trị không detach (L130) |
| C28 | Zoom nhị phân = khoảng cách bất kỳ | 1 ô tại level l = 2^l pixel; ∑ b_l×2^l |
| C29 | Giữ cơ chế cũ (mũi tên, toggle, commentary) | Phần 5 |
| C30 | Mũi tên = gradient (x,y) thực, không phải hướng target | Sửa dirX/dirY thành (gx, gy) |

---

## Phần 8: Xác nhận hiểu mã nguồn (tóm tắt để đánh giá nhanh)

1. **Tên channel trượt:** `C_zoom` (= 4^level), `C_zoom_2` (= 2^level), là dim=1 trong `[BATCH, C, 4, H, W]`.
2. **Vị trí hiển thị ≠ phép chiếu xyz:** routing (L45) rời rạc, giới hạn 3×3, detached → vị trí hiển thị TRỄ so với phép chiếu liên tục; zoom giúp bắt kịp.
3. **xyz nhận gradient ở đâu:** `calculate_key_query` utils.py L174-182 (grouped_key không detach, query detach); truyền qua einsum giá trị không detach (smap.py L130). Detach chỉ ở selection weights (weights_x/weights_m.detach()).
4. **m vs xyz:** cùng straight-through, khác nguồn (m: pre_mask_grdf; xyz: key_query_grdf).
5. **SST chống corrupt:** finest grid + FOLD/UNFOLD bijection (L216/L235).
6. **Camera:** một ma trận chung toàn cục, identity trong app (smap.py L15, utils.py to_3d3x3).
