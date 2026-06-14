# PROPRIETARY LICENSE / DISCLAIMER

All HDVO materials distributed in `tools/testing/hdvo/` and its subdirectories, including source code, documentation, configuration, and session files, are the **Proprietary Intellectual Property** of Nguyễn Lê Thiện Ân.

The use, copying, modification, and distribution of these files are strictly prohibited, except with explicit written permission from Nguyễn Lê Thiện Ân.

These files contain Core Business Logic and Semantic Hypotheses, which are considered **Trade Secrets**.

**© 2025 Nguyễn Lê Thiện Ân (Thien An L. Nguyen). All Rights Reserved.**

---

# HDVO Architecture: AI Semantic-Aided Model

The **Human-in-the-Loop Hypothesis-Driven Validation and Optimization (HDVO)** architecture is the framework behind applications like the [`Fold/Unfold app`](https://hdvo.vercel.app/) and the `Prediction/Optimization app`.

## I. The HDVO Layered Architecture

| Layer | Corresponding Component | Core Role |
| :--- | :--- | :--- |
| **L1: Semantic Hypothesis/Logic Layer** | JavaScript/AI Function (defined by the engineer) | **Knowledge Encoding:** Translates the core business logic (Hypothesis) into an executable function. |
| **L2: Cross-Environment Execution Layer** | `exec/exec.py` (Python), `exec/execute_ai_function.js` (Node.js) | **Execution Bridge:** Packages data (Tensor/Numpy) from the Python environment and executes the Logic function (L1) in the JavaScript environment. |
| **L3: Optimization and Feedback Layer** | [`Fold/Unfold app`](https://hdvo.vercel.app/) | **Human-in-the-Loop:** Compares the Hypothesis execution results (L1) against the Gold Standard/Ground Truth data and highlights **Mismatches** for the engineer to refine. |

## II. Beyond Testing: Ensuring Core Semantic Integrity

In complex data processing systems, particularly those involving **CNN Gradient Flow** and advanced processes like **SMap Optimization**, traditional unit or integration tests often fall into the **Test Coverage Paradox**.

While high **Code Coverage** is attainable, covering the entirety of the **Input Space Explosion** is practically impossible. This creates a significant risk: the codebase might **pass all tests** but still harbor deep **Semantic Bugs**, violating the Core Business Logic encoded in the Semantic Hypothesis Layer (L1). Uncontrolled logical changes over time lead to **Semantic Drift**, turning maintenance into a **Maintainability Nightmare**.

HDVO is designed to directly address **The Semantic Gap** that current industry standards overlook. Instead of focusing solely on **Mathematical Accuracy** (`gradcheck`), HDVO focuses on whether the code performs **correctly according to the established logic (Semantic Integrity)**.

## III. HDVO: Mechanism for Maintaining Semantic Integrity via Markov Chain

The **SMap Optimization** process is conceptualized as a **Markov Chain (MC)**, where the state (`State`) of the source code (L1 Logic) is refined with each iteration. HDVO's goal is to ensure that every **state transition** adheres to the core **Semantic Hypothesis**.

HDVO maintains the core semantic integrity of the SMap Optimization code through the following structured **Semantic-Reinforced Iterative Learning** cycle:

### 1. Error State Sampling and Diagnosis (L3 & L2 Layers)

This step isolates the **State Vector** where the Markov Chain is violating semantic integrity.

-   **Error Hotspot Analysis:** The operating engineer uses feedback from the L3 Layer to identify the area with the most concentrated mismatches (e.g., the area with the highest **Error Entropy**).
-   **State Vector Extraction:** The Cross-Environment Execution Layer (L2 - `exec.py`) is used to precisely sample that error state:
    ```bash
    python "tools\testing\hdvo\exec\exec.py" -o DEPconfig0.json "tools\testing\hdvo\fold_unfold_session.json"
    ```
    This command generates `DEPconfig0.json`, the **Full State Vector** containing context, targets, and conditional blocks for AI analysis.

### 2. Transition Analysis and Hypothesis Construction (L1 Layer)

This is the core phase where the AI is trained to capture the entire information about the **Markov Chain** and propose a **complete hypothesis for the SMap Optimization process**. To ensure **Accumulated Knowledge**, HDVO utilizes a sequence of structured prompt templates (analyzed in reverse order: Component 3, 2, 1):

#### a. Component 3: Denoising Mechanism and Markov Chain Whole (Analysis Order: 1st)

**Goal:** To understand the `Fold/Unfold` mechanism as a **gradual denoising process**, guiding the point towards its true target position (Ground Truth) until it reaches the **Stationary State** of the Markov Chain.

**Analysis Requirement:** The AI must demonstrate an understanding of all three components and their interaction within the whole Markov Chain. All accumulated information must be stored solely as a **Hypothesis/Theory** (L1's unique memory).

**Prompt Context:**
> "Về tổng quan, cơ chế fold/unfold giúp (thông qua quá trình huấn luyện) mang một điểm ở vị trí lân cận đến vị trí mà nó nên được hiển thị (tức là vị trí đang được so sánh với mục tiêu). Như vậy, gradient ở đây chính là gradient cho điểm tương ứng với vị trí hiện tại so với vị trí gốc (thể hiện qua thứ tự của điểm trong một slice, slice 0 tương ứng với việc dịch chuyển một ô lên phía trên về bên trái, slice 8 tương ứng với việc dịch chuyển xuống dưới và về bên phải, các slice khác có thể suy ngược ra vị trí dịch chuyển). Tuy nhiên, cũng cần biết rằng mục tiêu đang được so sánh chưa phải là vị trí đầu ra cuối cùng của điểm (tức vị trí thực tế, ground truth) nhưng thông qua quá trình huấn luyện mà điểm sẽ dần đi về vị trí thực tế của nó như một quá trình khử nhiễu khi điểm (gần như chắc chắn) sẽ đạt được trạng thái dừng của Markov chain tương ứng với phân phối có thể được dùng để mô tả trạng thái di chuyển của điểm trong quá trình huấn luyện. Để hiểu được toàn bộ Markov chain này cần hiểu 3 thành phần xây dựng nên nó mà ở đây là đang nói về thành phần thứ 3. Vì bạn cần phải hiểu chính xác mỗi thành phần trong tổng thể nên khi phân tích bạn cần thể hiện hiểu biết của mình về cả 3 thành phần cũng như cách chúng tương tác với nhau trong tổng thể toàn bộ Markov chain, nếu như thành phần nào vẫn chưa được cung cấp thông tin phân tích thì hãy nói rõ rằng "Thông tin để phân tích thành phần thứ X vẫn chưa được cung cấp". Nhắc lại rằng ở đây chúng ta chỉ đang làm việc với các cấu thành của thành phần thứ 3 (tức là thông tin lỗi không bao gồm hai thành phần còn lại) và mỗi lần chúng ta chỉ tiếp cận được với lỗi của mỗi thành phần (theo giới hạn thiết kế của quy trình phân tích lỗi này) nhưng vì quá trình sẽ được lặp lại và bạn sẽ được tiếp cận lần lượt với các thành phần khác để có thể tích lũy đầy đủ thông tin và xây dựng tri thức về toàn bộ Markov chain. Cuối cùng, tất cả mọi thông tin tích lũy được đều phải được lưu ở dạng giả thuyết/học thuyết vì chỉ có thông tin này mới được giữ lại ở mỗi lần lặp và là trí nhớ duy nhất. "

#### b. Component 2: Gradient Gating and Point Dynamics (Analysis Order: 2nd)

**Goal:** Analyze the construction logic of array `X` from `comp_2_pre_X` and the **Gradient Gating** mechanism. The AI must construct a coherent logic for the entire Markov Chain, detailing how a point moves or stops.

**Key Concepts:**
- **Movement:** Point movement is based on `comp_2_pre_X` values: moved if `>.5`, or optimized for movement if `<.5` using the gradient technique (`coord_tensor - coord_tensord.detach()`).
- **Gating and Stopping Logic:** The AI must predict the stopping condition (related to Component 1) and clarify whether gradient blocking is **universal** (first slice only) or **specific** (all 9 slices have meaning).

**Prompt Context:**
> "Nhắc lại rằng ở lượt phân tích này chúng ta chỉ đang làm việc với các cấu thành của thành phần thứ 2 (tức là thông tin lỗi không bao gồm hai thành phần còn lại) và mỗi lần chúng ta chỉ tiếp cận được với lỗi của mỗi thành phần (theo giới hạn thiết kế của quy trình phân tích lỗi này) nhưng vì quá trình sẽ được lặp lại và bạn sẽ được tiếp cận lần lượt với các thành phần khác để có thể tích lũy đầy đủ thông tin và xây dựng tri thức về toàn bộ Markov chain. Thành phần thứ 2 này của Markov chain sẽ tương tác với mảng comp_2_pre_X thể hiện logic đã được dùng để xây dựng mảng X. Thông qua các tấm thành phần của mảng comp_2_pre_X việc đóng/mở gradient tại một vị trí dựa trên giá trị của comp_2_pre_X tại vị trí đó cho phép "dịch" một điểm đến vị trí tương ứng với tọa độ đang được gán tại điểm đó. Như vậy, giá trị của comp_2_pre_X sẽ thể hiện hai điều. Một là, điểm đã dịch đến vị trí tương ứng rồi (nếu giá trị là >.5). Hai là, điểm vẫn chưa dịch đến vị trí đó và việc tối ưu giá trị tại điểm sẽ đồng thời dịch điểm đến vị trí đó (việc này có thể đạt được thông qua kỹ thuật xây dựng giá trị với một tổng có giá trị 0, nhưng có chứa thông tin về tọa độ tương ứng với vị trí, là coord_tensor-coord_tensord.detach(), vừa giúp không làm thay đổi giá trị và vừa có thể giúp cho việc nhận gradient). Nói tóm lại, bạn cần nắm được cơ chế đóng/mở gradient dựa trên thông tin không chỉ về vị trí hiện tại của điểm mà còn về các vị trí khả dĩ (giá trị <.5) được quy định bởi thứ tự của 9 tấm thành phần (trên-trái, trên-giữa, ..., dưới-phải). Ngoài ra, còn một logic nữa mà bạn sẽ phải dự đoán, liên quan đến việc một điểm đã có thể dừng được chưa (đã dịch đến một điểm kích hoạt trên mục tiêu mà điểm đó chỉ trùng với một mình nó chưa). Logic này sẽ được đề cập đến ở thành phần thứ nhất của Markov chain. Để cho dễ hình dung thì khi một điểm đã thỏa mãn điều kiện để dừng thì đầu ra mà bạn dự đoán sẽ là 0 nhưng tấm X thì sẽ được xây dựng bằng cách trượt giá trị 0 đó ra các vị trí "dịch" khả dĩ tương ứng để chặn cách gradient dẫn đến việc dịch chuyển điểm đó. Việc của bạn không chỉ đơn thuần là tối ưu lỗi nhưng còn phải xây dựng được một logic mạch lạc về toàn bộ Markov chain, về cách mà một điểm đạt trạng thái dừng hoặc di chuyển và về cách mà một lỗi ảnh hưởng đến sự tiến hóa của toàn hệ thống. Cuối cùng là về một lưu ý rằng việc block gradient có thể là chung cho toàn bộ các vị trí dịch khả dĩ (khi đó đầu ra cần tối ưu sẽ là ở tấm thành phần đầu tiên, còn các tấm còn lại có giá trị 0 tượng trưng) hoặc có thể riêng cho mỗi vị trí dịch (lúc này đầu ra của 9 tấm thành phần đều sẽ có ý nghĩa). "

#### c. Component 1: Physical Entity and State Unification (Analysis Order: 3rd - Final)

**Goal:** Capture how information from 9 possible translation positions (`comp_1_pre_X` slices) is **Aggregated** to create the **Unified State** of a **Physical Entity**. The result is then `unfold`ed to generate `comp_2_pre_X`.

**Knowledge Validation:** The AI must provide a **concrete example** (array name, value, output, positional meaning) and specify which component is being analyzed. The engineer uses induced errors (e.g., value -17) as a feedback mechanism to ensure compliance with the quality requirement.

**Prompt Context:**
> "Nhắc lại rằng ở lượt phân tích này chúng ta chỉ đang làm việc với các cấu thành của thành phần thứ nhất (tức là thông tin lỗi không bao gồm hai thành phần còn lại) và mỗi lần chúng ta chỉ tiếp cận được với lỗi của mỗi thành phần (theo giới hạn thiết kế của quy trình phân tích lỗi này) nhưng vì quá trình sẽ được lặp lại và bạn sẽ được tiếp cận lần lượt với các thành phần khác để có thể tích lũy đầy đủ thông tin và xây dựng tri thức về toàn bộ Markov chain. Thành phần này của Markov chain sẽ tương tác với mảng comp_1_pre_X thể hiện logic đã được dùng để xây dựng mảng comp_2_pre_X . Thông qua các tấm thành phần của mảng comp_1_pre_X trạng thái của một điểm vật lý có thể được xây dựng dựa trên thông tin về 9 vị trí "dịch" khả dĩ của nó quanh vị trí hiện tại của nó. Như vậy, thứ tự của các tấm thành phần của mảng này có thể quy đổi sang offset tương ứng với việc dịch chuyển đi một ô chung quanh vị trí hiện tại. Điều này cho phép thống nhất (thông qua aggregation và sau đó unfold để tản ra các vị trí xung quanh) trạng thái đóng/mở gradient tại các vị trí khả dĩ và thể hiện sự tồn tại của điểm đó là một thể hiện vật lý hơn là một tập hợp các logic độc lập. Như vậy, bạn không chỉ hiểu về thông tin tại một vị trí sau khi trượt (fold) mà còn phải hiểu cách mà thông tin về các vị trí khả dĩ của một điểm được tổng hợp để tạo nên thông tin nhất quán về trạng thái của điểm vật lý đằng sau các giá trị tại một vị trí trên mảng X rồi sau đó lại được trượt (unfold) để tạo ra mảng comp_2_pre_X và góp phần xây dựng nên mảng X. Việc tối ưu lỗi giờ đây không chỉ đơn thuần là đưa ra một hàm lý giải cố định mà còn dựa trên đánh giá của tôi về khả năng hiểu vấn đề (về toàn bộ Markov chain) của bạn. Như vậy, ngoài việc tối ưu lỗi, bạn cần cho tôi một ví dụ cụ thể (tên mảng, giá trị, đầu ra tương ứng với mỗi thành phần của Markov chain cũng như làm rõ về ý nghĩa của vị trí của các giá trị tương ứng trên mảng) để tôi có thể thay đổi đầu vào tương ứng với ví dụ cũng như chỉ định thành phần nào của Markov chain đang được phân tích và/hoặc ví dụ có đang được sử dụng, hay chỉ đơn giản là dùng nội dung của các mảng đã cho (vì khi ví dụ được sử dụng thì bạn cần phải khai báo các giá trị liên quan đến ví dụ trong hàm, thay cho giá trị lấy từ các mảng tham số thì mới có thể tối ưu được), để bạn có thể đưa ra đề xuất phù hợp. Các lỗi sẽ được tôi sử dụng để thông báo đánh giá của tôi về mức độ đáp ứng của đề xuất, ví dụ nếu tôi không thấy ví dụ cụ thể trong đề xuất thì tôi sẽ chỉnh giá trị kì vọng để bạn có thể thấy rằng có một ô đang có lỗi với giá trị là -17 và cứ như vậy thì bạn sẽ không thể tối ưu lỗi này nếu không đảm bảo yêu cầu này hơn là chỉ đưa ra một hàm đề xuất tối ưu đơn thuần. "

---

# Value Proposition

The HDVO Framework addresses **The Semantic Gap** that current industry standards neglect.

| Feature | Current Industry Standard | AI/HDVO Process |
| :--- | :--- | :--- |
| **Testing Focus** | **Mathematical Accuracy** (`gradcheck`) and **End-to-End Performance** (Accuracy/Loss). | **Semantic Correctness** and Business Logic consistency (Maintaining **Semantic Integrity**). |
| **Logic Control** | **Passive:** Only checks what the current code produces. | **Active:** Encodes business logic into the Hypothesis Layer (L1) and cross-validates against the Gold Standard using the **Markov Chain** model. |
| **Core Value** | Ensuring the correctness of mathematical operations. | Ensuring business logic, especially positional constraints and domain knowledge, is maintained. |

---

# HDVO Application Examples

| Testing Scenario | HDVO Logic Focus | Usage in Project |
| :--- | :--- | :--- |
| **Standard Operation Check** | **High-Precision Operations:** Verifies that complex functions (e.g., convolution/deconvolution) adhere to the defined precision. | *[New Example]* |
| **Positional Consistency Check** | **Positional Constraint Validation:** Verifies that computations do not violate physical or logical positional constraints within the grid. | *[New Example]* |
| **Boundary Hypothesis Validation** | **Encoding logic rules defining classification boundaries.** Uses HDVO to check if model decisions violate defined logical boundaries. | *[New Example]* |
| **Post-processing Logic Audit** | **Logic Data Flow Check:** Verifies that post-processing functions (e.g., denoising, threshold normalization) comply with complex logical conditions encoded in L1. | *[New Example]* |
