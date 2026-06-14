# Sổ tay thực hành: Kiểm định độc lập như một mặc định

**Cách áp dụng quy trình kiểm định/cải thiện của SMap cho các dự án khác — phân loại theo loại dự án, viễn cảnh, tình huống và edge case.**

> Sổ tay này tách **quy tắc bất biến** (dùng được cho mọi dự án) khỏi **ví dụ cụ thể** (SMap — một hiện thân). Mục tiêu không phải bắt bạn theo case của SMap, mà đưa cho bạn một *thang trưởng thành*: vào ở đúng nấc dự án bạn đang đứng, leo cao chỉ ở chỗ thật sự cần. Đọc kèm tiểu luận *Khoảng Trống Lý Thuyết* (phần "tại sao") và `SMap-verification-paradigm.pptx` (phần "định vị").
>
> Các khối **▸ SMap (hiện thân)** chỉ để bạn *thấy* hình dạng của quy tắc trong một dự án thật — cái bạn mang đi là **hình dạng**, không phải miền bài toán.

---

## 0. Bài này dành cho ai — và KHÔNG dành cho ai

Kỷ luật này tốn công. Nó **đáng** đúng ở nơi đặt cược cao và **lãng phí** ở nơi oracle rẻ. Trước khi đọc tiếp, định vị dự án của bạn:

| Nếu dự án của bạn… | Thì… |
|---|---|
| Prototype dùng một lần, throwaway, rủi ro thấp | Dừng ở **L0–L1** (mục 11). Áp L3–L4 lên đây là tự giết lợi thế tốc độ. |
| CRUD / app nghiệp vụ, oracle rẻ (kiểu, snapshot, so khớp trực tiếp) | **L0–L2** là đủ. |
| Hệ thống sống lâu, brownfield, logic phức tạp, nhiều người chạm vào | **L3** trở lên cho phần rủi ro cao. |
| Oracle thật sự khó: "đúng là gì" *chưa được định nghĩa* (ML kernel, hệ số học/khoa học, hệ phân tán, an toàn-trọng yếu, reverse-engineering) | **L4** ở đúng phần khó. Đây là vùng SMap sống. |
| AI sinh phần lớn code, không ai giữ lý thuyết | Bất kể loại nào: nâng một nấc, vì bạn đang ở "legacy ngay từ khi sinh". |

**Quy tắc vàng:** leo thang chỉ ở **chỗ có oracle problem thật**. Không có dự án nào phải áp L4 cho toàn bộ.

---

## 1. Sáu nguyên lý nền (bất biến — học thuộc phần này)

Mọi thứ còn lại chỉ là cách hiện thực hóa sáu điều này:

1. **Source-of-truth (SST) là đối tượng nghiên cứu, không phải thứ bắt code phục tùng.** Trong hệ AI-in-the-loop, "đúng là gì" thường phải *khám phá*. Spec/test KHÔNG định nghĩa cái đúng — chúng là dụng cụ dò.
2. **Tam giác hóa ≥ 3 biểu diễn.** "Đúng" = nhiều biểu diễn suy ra *độc lập* cùng hội tụ — không phải so sánh hai chiều "code pass test của chính nó".
3. **Độc lập/làm mù là load-bearing.** Bên viết kiểm định không được thấy đáp án; phải suy từ ngữ nghĩa; phải coi sản phẩm là *dự đoán để bị bác bỏ*.
4. **Lập trường falsification.** Mục tiêu là *phá*, không *xác nhận*. Một "pass" chỉ tạm thời cho đến khi được bên độc lập tái lập.
5. **Tách probe ≠ verdict.** Script *đo đạc* (quan sát hành vi) và test *phán xử* (đúng/sai) là hai artifact, do hai bên khác nhau tạo.
6. **Giọng assumed-vs-actual.** Mọi trạng thái đạt được nhờ *giả định* phải hiển thị là **giả thuyết**, không được trình bày như sự thật.

> ▸ **SMap (hiện thân):** SST = mã nguồn PyTorch; biểu diễn 2 = chứng minh giải tích trong `math_model.md`; biểu diễn 3 = mô phỏng chạy được (`smap_simulator.jsx` + `_repro_check.mjs`) và probe chạy source thật (`t_*.py`). Load-bearing = **dấu và điểm-zero** của gradient (không phải magnitude). Giọng assumed-vs-actual = các trạng thái "đã sửa" chỉ đạt nhờ bật công-tắc-giả-định đều hiển thị là HYPOTHESIS.

---

## 2. Phân loại dự án → chọn nấc thang

Bốn trục quyết định bạn leo cao tới đâu và cần những artifact gì:

| Trục | Đầu rẻ (leo thấp) | Đầu khó (leo cao) |
|---|---|---|
| **Độ khó của oracle** | "đúng" định nghĩa được trước (kiểu, snapshot, công thức đã biết) | "đúng" phải khám phá; output ngẫu nhiên/không khả vi; không có ground truth |
| **Tuổi thọ & đặt cược** | throwaway, rủi ro thấp | sống lâu, brownfield, sự cố tốn kém |
| **Vị trí lý thuyết** | bạn viết, lý thuyết còn trong đầu | AI viết / legacy — không ai giữ lý thuyết |
| **Loại việc** | thêm tính năng có spec rõ | debug/correctness/proof/reverse-engineering |

**Cách dùng:** chấm mỗi trục. Càng nhiều trục lệch "đầu khó" → càng leo cao và càng cần đủ ba đỉnh tam giác. Bảng ánh xạ cụ thể ở **mục 9**.

---

## 3. Dựng tam giác: ba biểu diễn độc lập

Trái tim của kỷ luật. Bạn cần **≥ 3 biểu diễn của *cùng một sự thật*, suy ra độc lập, hội tụ ở mức mang tải**. Hai chiều (code ↔ test) không đủ vì nếu cả hai đến từ cùng một prior, chúng *tương quan lỗi* (xem mục 10).

**Bước 1 — Định nghĩa "tương đương ở mức mang tải".** Đừng so ở bề mặt. Hỏi: *thuộc tính nào sai thì kết quả sai, thuộc tính nào khác chỉ là nhãn?* Chỉ đòi hội tụ ở phần mang tải.

> ▸ **SMap:** mang tải = dấu + zero của gradient; magnitude là nhãn. Hai biểu diễn "tương đương" khi cùng dấu và cùng zero/non-zero, dù số tuyệt đối khác.

**Bước 2 — Chọn ba đỉnh.** Tùy domain, các đỉnh khả dụng:

| Đỉnh | Là gì | Dùng khi |
|---|---|---|
| **SST / mã nguồn** | bản thân hệ thống (đối tượng nghiên cứu) | luôn luôn — đây là chân lý triển khai |
| **Mô hình giải tích / chứng minh** | đặc tả toán về tính chất bất biến/hội tụ | có cấu trúc toán; cần lý do "vì sao đúng" |
| **Triển khai tham chiếu độc lập** | impl thứ hai viết bởi người/agent khác (differential testing) | có thể viết lại rẻ; không có ground truth |
| **Quan hệ metamorphic** | "với phép biến đổi T, output phải đổi theo cách Q" | không biết output đúng nhưng biết *quan hệ* (xoay ảnh 360° → ảnh gốc) |
| **Property-based** | "với mọi input thỏa P, output thỏa Q" | bất biến phát biểu được dù không biết giá trị cụ thể |
| **Mô phỏng / mô hình quan sát được** | tái dựng hành vi để quan sát từng cơ chế | hệ động/khó nội quan; cần thấy cơ chế chạy |
| **Oracle tham chiếu** | dữ liệu vàng, hệ trước đó, lý thuyết đã chứng | có nguồn chân lý ngoài |

Chọn ba đỉnh **suy ra độc lập nhất có thể**. Ba đỉnh cùng rút từ một prior = một đỉnh đội ba mũ (mục 10).

---

## 4. Quy trình lặp (6 bước — tổng quát hóa từ SMap)

1. **Duy trì một đặc tả-đúng *gần source nhất*.** Mô hình bám theo source, KHÔNG phải ngược lại. Mỗi lệch giữa source và đặc tả = một ứng viên bug, không phải lý do sửa đặc tả.
2. **Dựng một "gương" quan sát được, phản chiếu *hai chiều*.** Forward: một bug trở nên *quan sát được*. Backward: làm việc với cơ chế đó *để lộ* trạng thái source + hướng sửa. Phản chiếu một chiều là một khuyết tật.
3. **Mỗi lệch *đã xác nhận từ source* = một bug có hồ sơ + một công tắc SOURCE(lỗi)/FIXED(sửa), gated đúng chỗ áp dụng.** KHÔNG bịa bug.
4. **Dùng gương để phát hiện/sửa** và yêu cầu điều chỉnh gương.
5. **Kiểm định hai tầng** (mục 6): (a) tĩnh, agent không bị mớm; (b) tương tác, lái hệ sống — khi có nghi ngờ blocker.
6. **Lặp 1–5 đến khi đúng.**

> ▸ **SMap:** "gương" = app React; phản chiếu hai chiều = bug thành toggle quan sát được *và* thao tác với cơ chế làm lộ trạng thái source + hướng sửa.

**Khái quát cho dự án không có "app":** "gương" có thể là một CLI in ra trạng thái trung gian, một test harness ghi lại quyết định nội bộ, một notebook trực quan hóa, hay một log có cấu trúc. Yêu cầu bất biến: *mỗi cơ chế then chốt phải quan sát được từ bên ngoài*.

---

## 5. Definition of Done: "equivalence-complete"

Một cơ chế chỉ được coi là **đã kiểm định** (không chỉ "trông có vẻ đúng") khi đủ **bốn** điều — đây là rào chắn chống tự lừa:

```
equivalence-complete  ⟺  (1) scenario quan sát được   [app_scenario]
                        ∧ (2) test độc lập trên SST    [source_test, do bên khác viết]
                        ∧ (3) tương đương ở mức mang tải [equivalence: sign/zero…]
                        ∧ (4) tái lập được             [reproducible]
```

**Schema hồ sơ cho mỗi cơ chế** (chép template này):

```
mechanism: <tên>
  source_anchor:   <file:line / hàm — chốt vào source, re-grep sau mỗi sửa>
  semantic_claim:  <cơ chế này làm gì, phát biểu kiểm được>
  scenario:        <cách làm cho nó quan sát được trên "gương">
  independent_test:<probe/test do bên ĐỘC LẬP viết trên SST>
  equivalence:     <tiêu chí "bằng nhau" ở mức mang tải>
  independence_note:<ai viết test, có bị mớm đáp án không>
  residual_risk:   <điều vẫn có thể sai sau khi pass>
```

**Bảng triage trưởng thành** (để biết cơ chế nào *thật sự* xong):

| Ký hiệu | Nghĩa |
|---|---|
| ✅ complete | đủ 4 điều, đã tái lập độc lập |
| ◑ near | thiếu 1 điều (vd: chưa có agent độc lập thứ hai) |
| ◑ partial | chỉ kiểm được cấu trúc/proxy, chưa chạm source thật |
| ✗ blocked | proxy ≠ source, hoặc thiếu harness/dữ liệu |
| ✗ unspecced | chưa có scenario + test |

> ▸ **SMap:** chỉ R9/R13 (lực-mask S1) đạt ✅ complete — và chỉ sau khi một agent độc lập **tự viết probe của nó** (`t_indep_matched.py`) và tái lập. Lần trước đó tái dùng probe cũ của chính mình → **không tính độc lập**. Đây là bài học cốt lõi (mục 6, edge case).

---

## 6. Hợp đồng vai trò agent (làm mù) — kèm prompt mẫu

"Ai được biết gì, khi nào" là **biến kỹ thuật hạng nhất**. Ba vai, ba hợp đồng:

**(A) Validator tĩnh, KHÔNG bị mớm.** Đưa *chỉ* source + đặc tả + gương; **giấu** danh sách bug/đáp án. Yêu cầu:
> "Độc lập từ SOURCE, hãy tái dựng [bất biến / tiêu chí đúng] và lập luận tính đúng. Quyết định đặc tả có phải là phiên bản đúng *gần source nhất* không, và chứng minh có đứng vững không. Quyết định gương có phản chiếu *mọi* cơ chế then chốt không; thách thức mọi chỗ trượt. FALSIFY: một tín hiệu hiển thị mà đọc-từ-source lại lệch; một đại lượng không truy được về source; một bug source thật mà gương bỏ sót; một bug bịa. Chạy source thật nếu có thể. Xuất: mỗi claim ACCEPT/FALSIFY kèm file:line, bug mới, residual risk."

**(B) Validator tương tác.** Đưa hệ đang chạy + bộ công cụ quan sát + các scenario như *fixture*, **giấu** `expected`/`equivalence`. Agent lái hệ sống, báo ACCEPT/CONCERN/FALSIFY theo giá trị *quan sát được*. Rào chắn: probe chỉ phơi **trạng thái render + chuỗi thao tác**, không bao giờ phơi đáp án.

**(C) Bug-fix agent (kích hoạt thủ công).**
> "Sửa MỘT bug đã xác nhận trong SOURCE để khớp đặc tả. (1) RE-DERIVE từ source rằng lệch là thật (re-check line numbers). (2) Sửa source TỐI THIỂU. (3) VERIFY: re-derive; chạy source thật; không bất biến faithfulness nào regress. (4) PHẢN CHIẾU NGƯỢC: retire/flip toggle trên gương + diff đặc tả + flip badge chuẩn. (5) Bàn cho validation không-bị-mớm. Xuất: diff (file:line), bằng chứng từ source, cập nhật gương/đặc tả, residual risk."

**Rào chắn độc lập (quan trọng nhất):** bất kỳ test-hook/probe/fixture nào cũng chỉ phơi *trạng thái người dùng quan sát được + chuỗi thao tác*, **không bao giờ** phơi source-truth / bug-list / đáp án kỳ vọng.

---

## 7. Chống overclaim: probe ≠ verdict + giọng assumed-vs-actual

- **Tách đo đạc khỏi phán xử.** Script quan sát hành vi và test phán xử đúng/sai là *hai* artifact, do *hai* bên tạo. Trộn chúng = thẩm phán tự chấm bài mình.
- **Giọng assumed-vs-actual.** Mọi "xanh/đã sửa/đạt chuẩn" có được nhờ *bật một giả định* phải đọc là **giả thuyết**: "certificate hợp lệ (giả định fix)", "FIX (nếu áp dụng)". Chỉ trạng thái đạt qua *chứng minh / chạy thật / source-faithful* mới giữ giọng khẳng định.

> ▸ **SMap:** badge tự-audit mặc định "⚠2" (2 mục mở); xanh đọc là "✓ chuẩn đạt · N giả định"; chữ "instrument faithful" chỉ hiện khi *thật sự* giải quyết bằng proof/source, không phải bằng assume-toggle.

---

## 8. Tính liên tục: handoff + cold-start audit

Lý thuyết của LLM phù du, dựng lại mỗi phiên. Để kỷ luật sống xuyên phiên/agent, viết một **handoff tự chứa** (effective-verbal-context):

**Phải có:** mục tiêu phát biểu được không cần chat cũ; inventory artifact + cách chạy; thuật ngữ định nghĩa bằng nghĩa quan-sát-được/source-level; ranh giới giả định (được phép / KHÔNG được phép); open-work ưu tiên kèm *file/lệnh/test cụ thể*; bug catalogue đã xác nhận; **các anchor (file:line) kèm cảnh báo re-grep sau sửa**.

**Cold-start audit** — chạy khi handoff cũ, hoặc việc kế tiếp rủi ro cao:

| Câu hỏi | Kiểm gì |
|---|---|
| Mục tiêu tái lập được không cần chat cũ? | có đoạn "recovery note" gọn |
| Artifact tìm + chạy được? | inventory + recipe |
| Thuật ngữ định nghĩa kiểu quan-sát-được? | không có thuật ngữ "treo" |
| Ranh giới giả định rõ? | sẽ không chứng minh nhầm thứ |
| Open-work hành động được? | có file/lệnh/test kế tiếp |
| Rò "this/above/latest"? | mọi dòng tự chứa |
| Residual risk? | điều không tái dựng được từ handoff |

> ▸ **SMap:** handoff phân biệt rõ "mask-optimization regime (đã chứng)" vs "m-fixed camera-calibration regime (phải re-derive)" — một **Setting assumption** tường minh ngăn việc áp chứng minh sai phạm vi.

---

## 9. Phân loại chi tiết theo loại dự án

Với mỗi loại: **oracle**, **ba đỉnh nên dùng**, **nấc thang**, **tình huống**, **edge case riêng**.

### A. CRUD / app nghiệp vụ — *oracle rẻ*
- **Oracle:** "đúng" định nghĩa được trước (so khớp DB, snapshot UI, quy tắc nghiệp vụ viết ra).
- **Ba đỉnh:** thường không cần — code ↔ test ↔ snapshot là đủ. **Thang: L0–L2.**
- **Tình huống:** thêm field, sửa form, endpoint mới.
- **Edge case:** khi một "quy tắc nghiệp vụ" hóa ra *chưa ai định nghĩa rõ* (vd: làm tròn tiền tệ, múi giờ, quyền hạn chồng chéo) → cục bộ đó nhảy lên "oracle khó", áp L3 *chỉ cho cục bộ đó*.

### B. Thư viện / API thuần (hàm thuần) — *oracle vừa*
- **Oracle:** bất biến phát biểu được dù không biết mọi giá trị.
- **Ba đỉnh:** property-based + differential (impl tham chiếu) + đặc tả. **Thang: L2–L3.**
- **Tình huống:** parser, serializer, thư viện ngày-giờ, thuật toán chuẩn.
- **Edge case:** *metamorphic* cứu khi không có giá trị đúng (parse→serialize→parse phải bằng nhau). Cẩn thận encoding/locale làm "bằng nhau ở bề mặt" ≠ "bằng nhau ở mức mang tải".

### C. Hệ số học / khoa học / ML kernel — *oracle khó* (vùng SMap)
- **Oracle:** "đúng" *phải khám phá*; output ngẫu nhiên/không khả vi; không có ground truth điểm-điểm.
- **Ba đỉnh:** SST ↔ chứng minh giải tích ↔ mô phỏng/probe chạy được; hội tụ ở **mức mang tải** (dấu/zero của gradient, bất biến hội tụ), không phải giá trị tuyệt đối. **Thang: L4 ở phần khó.**
- **Tình huống:** training "chạy được nhưng output bẩn"; nghi một cơ chế cohesive có bug latent.
- **Edge case:**
  - *Faithfulness ≠ correctness:* một gương phản chiếu *trung thực* một thuật toán *có bug* vẫn pass faithfulness. Phải chứng minh correctness *bên trên* faithfulness.
  - *Premature read:* kết luận "stall/đứng" từ quan sát 3–4 bước có thể sai — chạy đủ dài (vd headless tới hội tụ) trước khi gọi tên. (SMap: "far/k_const stall" hóa ra là *rate effect*, không phải correctness stall.)
  - *Regime shift:* chứng minh đúng ở regime này phải **re-derive** ở regime khác (SMap: m-fixed camera-calibration). Luôn ghi *Setting assumption*.

### D. Hệ phân tán / concurrent — *oracle khó (phi tất định)*
- **Oracle:** phi tất định; "đúng" = bất biến an toàn + tiến triển dưới mọi interleaving.
- **Ba đỉnh:** đặc tả hình thức (TLA+/model checking) ↔ differential giữa replica ↔ fault-injection/chaos quan sát được. **Thang: L3–L4.**
- **Tình huống:** consensus, cache invalidation, exactly-once.
- **Edge case:** lỗi chỉ hiện dưới interleaving hiếm — probe phải *ép* lịch trình, không chờ may rủi. Đừng nhầm "chưa thấy lỗi" với "không có lỗi".

### E. Reverse-engineering / debug hệ "chạy được nhưng output bẩn" — *debugging arc* (đúng cung thật của SMap)
- **Oracle:** SST = chính hệ đang chạy; "đúng" là một hành vi mong đợi *chưa được đặc tả đầy đủ*.
- **Quy trình:** (1) khoanh vùng cohesive nghi ngờ; (2) instrument trạng thái trung gian để *quan sát được*; (3) dựng đặc tả-đúng-gần-source; (4) tam giác hóa để xác nhận lệch là *bug source*, không phải hiểu nhầm; (5) thu hẹp regime (chỉ debug regime đang lỗi). **Thang: L3–L4.**
- **Edge case:** "bug" thường là *hiểu nhầm regime*. Tách rõ **source fact** khỏi **diễn giải bug**, và luôn có *đường falsification* trước khi đề xuất sửa.

### F. Brownfield / legacy (lý thuyết đã mất) — *theory-reconstruction trước*
- **Oracle:** behavior hiện tại *là* SST tạm thời (nhưng có thể chính nó đang sai).
- **Quy trình:** trước khi sửa, **xây lại lý thuyết** từ behavior thật (không từ tài liệu lỗi thời): đọc execution path, phân biệt code load-bearing vs incidental, đọc test như documentation, lộ implicit contract. **Thang: L3 cho phần rủi ro cao.**
- **Edge case:** "characterization test" chốt behavior *hiện tại* (kể cả khi nghi sai) để mọi thay đổi sau đó có điểm tựa; nhưng đánh dấu rõ cái nào là "đúng mong muốn" vs "chỉ là hiện trạng".

### G. Greenfield / prototype throwaway — *tối thiểu*
- **Thang: L0–L1.** Đừng phí kỷ luật.
- **Edge case quan trọng nhất:** *khoảnh khắc nó thôi là throwaway*. Dấu hiệu: có người thứ hai phụ thuộc vào nó; nó lên production; nó sống quá 1 lần dùng. Lúc đó **nâng nấc ngay**, vì theory debt đã bắt đầu tích.

---

## 10. Edge case & failure mode xuyên suốt (đọc kỹ)

| Failure mode | Triệu chứng | Cách chặn |
|---|---|---|
| **Lỗi tương quan (Knight–Leveson)** | ba "biểu diễn độc lập" cùng sai một chỗ → *hội tụ giả* | đa dạng *thật*: người/agent khác, phương pháp suy luận khác, model khác; cảnh giác khi cả ba cùng từ một LLM |
| **Pattern retrieval** | validator trả lời đúng-generic nhưng bỏ qua ràng buộc *rất cụ thể* của bạn; hỏi toàn câu "hệ nào cũng cần" | bắt nó *liệt kê assumption nó tự đặt*; hỏi câu đặc thù cho *bất thường cụ thể*; đối chiếu lại artifact thật |
| **Faithfulness ≠ correctness** | gương trung thực với một thuật toán có bug vẫn "xanh" | chứng minh correctness *trên* faithfulness; faithfulness chỉ là điều kiện cần |
| **Premature read** | kết luận từ quan sát quá ngắn | chạy đủ dài tới trạng thái ổn định trước khi gọi tên |
| **Regime shift** | chứng minh đúng ở phạm vi A bị dùng cho phạm vi B | ghi *Setting assumption*; re-derive khi đổi regime |
| **Stale anchors** | line number/đường dẫn trôi sau khi sửa | re-grep trước khi dựa vào anchor |
| **Probe leakage** | đáp án rò vào probe → "độc lập" giả | probe chỉ phơi trạng thái quan sát được + thao tác |
| **Overclaim qua assume-toggle** | "đã xong" trong khi mới chỉ giả định | giọng assumed-vs-actual; mặc định hiển thị số giả định đang mở |
| **Tái dùng probe của chính mình** | "đã kiểm định độc lập" nhưng dùng lại test cũ của mình | đòi agent thứ hai *tự viết probe của nó*; nếu không → ◑ near, chưa ✅ |

---

## 11. Thang trưởng thành (vào ở đúng nấc, leo chỉ chỗ cần)

| Nấc | Thêm gì | Dành cho |
|---|---|---|
| **L0** | Test tự viết, cùng tác giả | đa số dự án hôm nay |
| **L1** | Tách *đo đạc* khỏi *phán xử* (probe ≠ verdict) | khi muốn ngừng nhầm "quan sát" với "xác nhận" |
| **L2** | Giọng *assumed-vs-actual*: hết overclaim "đã xong" | khi muốn rõ giả thuyết vs sự thật |
| **L3** | Một agent *độc lập, làm mù* viết lại kiểm định cho phần rủi ro cao | brownfield, hệ sống lâu, logic phức tạp |
| **L4** | Tam giác hóa ≥ 3 biểu diễn với model diversity | chỉ khi oracle thật sự khó |

**Phát hành kỷ luật, không phát hành framework:** đừng bắt dự án khác theo case của bạn — đưa cho họ *thang* này để vào ở đúng nấc họ đang đứng.

---

## 12. Checklist một trang

**Trước khi bắt đầu**
- [ ] Chấm 4 trục (mục 2) → chọn nấc thang mục tiêu.
- [ ] Định nghĩa "tương đương ở mức mang tải" — cái gì mang tải, cái gì là nhãn?
- [ ] Chọn ba đỉnh tam giác *độc lập nhất có thể*.

**Trong khi làm**
- [ ] Đặc tả-đúng bám source (mô hình theo source, không ngược lại).
- [ ] Mỗi cơ chế then chốt *quan sát được* từ bên ngoài (gương hai chiều).
- [ ] Mỗi bug *đã xác nhận từ source* mới được ghi; không bịa bug.
- [ ] Probe chỉ phơi trạng thái quan sát được — không rò đáp án.

**Definition of Done cho mỗi cơ chế**
- [ ] scenario quan sát được ∧ test độc lập trên SST ∧ tương đương mức mang tải ∧ tái lập được.
- [ ] Agent thứ hai *tự viết probe của nó* và tái lập (nếu không → ◑ near).

**Trước khi nói "đã xong"**
- [ ] Trạng thái xanh nào do giả định? → hiển thị là giả thuyết.
- [ ] Re-grep mọi anchor đã sửa.
- [ ] Liệt kê residual risk + Setting assumption (regime nào áp dụng).
- [ ] Handoff đủ để một phiên/agent mới tiếp tục không cần chat cũ.

---

*Sổ tay này là tầng "scaffolding + maturity ladder" của bộ Khoảng Trống Lý Thuyết. Quy tắc bất biến tách khỏi ví dụ SMap có chủ đích — cái chuyển giao là hình dạng, không phải miền bài toán.*
