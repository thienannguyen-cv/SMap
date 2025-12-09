# Kiến trúc HDVO: Mô hình Hỗ trợ Ngữ nghĩa AI

Kiến trúc **Tối ưu hóa và Kiểm định Giả thiết có Con người tham gia (Human-in-the-Loop Hypothesis-Driven Validation and Optimization - HDVO)** là khuôn khổ nằm sau các ứng dụng như `[Fold/Unfold app](https://hdvo.vercel.app/)` và `Prediction/Optimization app`.

## I. Kiến trúc HDVO

| Lớp | Thành phần Tương ứng | Vai trò Cốt lõi |
| :--- | :--- | :--- |
| **L1: Lớp Giả thiết Ngữ nghĩa/Logic** | Hàm JavaScript/AI (do kỹ sư định nghĩa) | **Mã hóa Tri thức:** Chuyển đổi logic nghiệp vụ (Hypothesis) thành một hàm logic có thể thực thi được. |
| **L2: Lớp Thực thi Liên môi trường** | `exec/exec.py` (Python), `exec/execute_ai_function.js` (Node.js) | **Cầu nối Thực thi:** Đóng gói dữ liệu (Tensor/Numpy) từ môi trường Python và thực thi hàm Logic (L1) trong môi trường JavaScript. |
| **L3: Lớp Tối ưu hóa và Phản hồi** | `[Fold/Unfold app](https://hdvo.vercel.app/)` | **Vòng lặp Con người tham gia:** So sánh kết quả thực thi Giả thiết (L1) với Dữ liệu Chuẩn/Thực tế (Gold Standard/Ground Truth) và chỉ ra các **điểm không khớp (Mismatches)** để kỹ sư tinh chỉnh. |

## II. Về Kiến trúc HDVO

Kiến trúc HDVO cho phép dự án mở rộng ra ngoài việc kiểm thử gradient để trở thành một phương pháp luận kiểm thử AI cấp cao, linh hoạt và có khả năng giải thích:

| Ứng dụng | Mô tả | Công cụ Tương ứng |
| :--- | :--- | :--- |
| **Kiểm thử Gradient** | **Thẩm định Logic Vị trí:** Kiểm tra tính đúng đắn ngữ nghĩa của gradient dựa trên các ràng buộc vị trí kernel (ví dụ: $3 \times 3$ slices). | `[Fold/Unfold app](https://hdvo.vercel.app/)` |
| **Tối ưu hóa Dự đoán** | **Kiểm tra Logic Miền thời gian:** Tối ưu hóa hàm dự đoán dựa trên các quy tắc thời gian hoặc điều kiện lịch sử đã biết (ví dụ: dự đoán mực nước ngầm 7 ngày, quy tắc vật lý/kinh tế). | *[Ví dụ mới]* |
| **Tối ưu hóa Quyết định Phân loại** | **Kiểm tra Giả thiết Ranh giới:** Mã hóa các quy tắc logic xác định ranh giới phân loại. Sử dụng HDVO để kiểm tra xem các quyết định của mô hình có vi phạm các ranh giới logic đã định nghĩa hay không. | *[Ví dụ mới]* |
| **Thẩm định Logic Hậu xử lý** | **Kiểm tra Luồng Dữ liệu Logic:** Xác minh rằng các hàm hậu xử lý (ví dụ: lọc nhiễu, chuẩn hóa ngưỡng) tuân thủ các điều kiện logic phức tạp được mã hóa trong L1. | *[Ví dụ mới]* |

------------------------------------------------------------------------

# Giá trị

HDVO Framework giải quyết **Khoảng cách Ngữ nghĩa (The Semantic Gap)** mà chuẩn công nghiệp hiện tại bỏ qua.

| Đặc điểm | Chuẩn Công nghiệp Hiện tại | Quy trình AI/HDVO |
| :--- | :--- | :--- |
| **Trọng tâm kiểm thử** | **Tính chính xác Toán học** (`gradcheck`) và **Hiệu suất Đầu-cuối** (Accuracy/Loss). | **Tính đúng đắn Ngữ nghĩa** và sự nhất quán Logic nghiệp vụ. |
| **Kiểm soát Logic** | **Thụ động:** Chỉ kiểm tra những gì mã nguồn hiện tại tạo ra. | **Chủ động:** Mã hóa logic nghiệp vụ vào Lớp Giả thiết và kiểm tra chéo Dữ liệu Chuẩn. |
| **Giá trị Cốt lõi** | Đảm bảo tính đúng đắn của phép toán. | Đảm bảo logic nghiệp vụ, đặc biệt là các ràng buộc vị trí và miền tri thức, được duy trì. |

# Disclaimer

Mặc dù dự án này được phát hành dưới Giấy phép Apache 2.0, thư mục tools/hdvo/ chứa Logic Nghiệp vụ Độc quyền và bị ràng buộc bởi các điều khoản Giấy phép Độc quyền đi kèm.