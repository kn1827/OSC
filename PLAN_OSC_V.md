# Kế hoạch OSC-V: Quan sát → Dừng / Kiểm chứng → Bảo đảm

Mục tiêu: nâng OSC từ "dừng sớm có bảo đảm" thành một bài toán **dừng tối ưu có hành động thu thông tin tốn phí**. Tại mỗi mốc của debate, hệ chọn một trong ba hành động: **STOP** (trả lời ngay), **VERIFY** (gọi verifier một lần) hoặc **DEBATE** (chạy tiếp). Hệ đi kèm hai loại bảo đảm có số cụ thể:
- bảo đảm hữu hạn mẫu khi huấn luyện;
- bảo đảm anytime-valid khi triển khai.

---

## A. OSC chạy như thế nào (đọc phần này trước)

**Trả lời ngắn:** kết quả chính của bài **vẫn tính offline trên log đủ vòng**, giống cách làm của DUS cũ. Chỉ bước thu log và bước verifier cần gọi model. Chạy OSC thật (dừng giữa chừng) là bước **xác nhận và triển khai**, không bắt buộc để có số trong bảng kết quả.

### A.1 Bốn bước

| Bước | Lệnh | Gọi model? | Đầu ra |
|---|---|---|---|
| 1. Thu log đủ vòng | `python MAD.py --stop none --log_root results/logs_osc` | **có** | mỗi câu hỏi chạy đủ 6 vòng; lưu đáp án, confidence, lập luận, verdict của critic, token của từng lượt gọi |
| 1b. Verifier offline | `python -m osc.verify_offline ...` | **có** (khoảng 3–4 lần gọi/câu hỏi) | verdict pass/fail cho mỗi cặp (câu hỏi, đáp án) từng xuất hiện trong log |
| 2a. Huấn luyện + cam kết | `python -m osc.train ...` | không | file policy: trọng số observer, T, ngưỡng dừng, (τ, κ, ℓ) nếu có VERIFY, kèm cam kết ε/δ |
| 2b. Đánh giá (replay) | `python -m osc.evaluate ...` | không | **các con số của bài**: accuracy, chi phí, harm của OSC / OSC-V so với fixed-depth, consensus, majority, verify-at-stop |
| 3. Chạy thật | `python MAD.py --stop osc --osc_policy ...` | **có** | debate thật sự dừng giữa chừng; token tiết kiệm thật; đáp án cuối |
| 4. Tự kiểm | `python -m osc.monitor ...` | không (dùng các câu audit của bước 3) | báo động nếu tỉ lệ dừng sai vượt ε |

### A.2 Vì sao đánh giá offline (replay) là hợp lệ

Dừng sớm **không làm thay đổi các vòng trước điểm dừng**. Một debate mà OSC dừng ở vòng k có vòng 0..k giống hệt vòng 0..k của debate chạy đủ 6 vòng (cùng model, prompt, `critic_mode`, nhiệt độ). Vì vậy:
- "OSC dừng ở vòng k" chính là "cắt log đủ vòng tại vòng k";
- chi phí = token của các vòng đã chạy (cộng token verifier nếu có VERIFY);
- log đủ vòng còn cho biết debate **sẽ** trả lời gì nếu chạy tiếp, nên đo được harm ("chạy đủ thì đúng, dừng sớm thì sai"). Chạy thật không đo được con số này vì nó không thấy phần sau điểm dừng.

Replay chỉ đúng khi mọi thiết lập giữ nguyên. **Đổi model, prompt hoặc `critic_mode` thì phải thu lại log và huấn luyện lại.** Policy lưu `task` và `max_rounds`, và orchestrator từ chối chạy khi hai giá trị này không khớp.

**Verifier cũng theo đúng nguyên tắc này.** Khi chạy thật (bước 3), verifier là một hành động **online** bên trong debate: orchestrator gọi nó ngay tại mốc mà chính sách chọn VERIFY, tối đa 1 lần mỗi câu, rồi dùng verdict để quyết định dừng hay chạy tiếp. Bước 1b chỉ là **dữ liệu** để học τ và để replay. Verdict cho một cặp (câu hỏi, đáp án) không phụ thuộc vào thời điểm hỏi, vì verifier luôn nhận cùng một input: câu hỏi, đáp án, và lập luận lúc đáp án xuất hiện lần đầu (cả code offline lẫn orchestrator dùng chung quy tắc này), với temperature 0. Nên việc hỏi trước cho mọi đáp án ứng viên rồi tra lại khi replay cho kết quả giống hệt việc hỏi lúc chạy thật, trong khi rẻ hơn nhiều: mỗi cặp chỉ hỏi 1 lần, dùng chung cho mọi mốc, mọi seed và mọi cấu hình (a, κ, ℓ) được thử.

### A.3 Khác gì DUS cũ

| | DUS cũ | OSC |
|---|---|---|
| Thu log đủ vòng rồi phân tích offline | có | có (bước 1–2) |
| Điểm dùng để dừng | điểm bất định tự đặt công thức | xác suất đúng của từng đáp án, học từ log (mục B) |
| Bảo đảm bằng số | không | "dừng sớm làm hỏng ≤ ε câu, độ tin 1 − δ" |
| Dùng được khi chạy thật | không (chỉ mô phỏng trên log) | có (bước 3), cùng một file policy |
| Kiểm lại khi triển khai | không | có (bước 4) |
| Hành động VERIFY | không | có (mục 2.3–2.5) |

### A.4 Con số nào lấy từ đâu

| Con số trong bài | Nguồn |
|---|---|
| Accuracy / chi phí / harm, so sánh với fixed-depth cùng chi phí | bước 2b (replay, nested 5 fold) |
| Cam kết ε/δ và kiểm tra tính đúng của nó (E2) | bước 2a, lặp trên 200 lần chia ngẫu nhiên |
| τ₁, τ₀, vùng verify, blind spot (E3) | bước 1b + 2b |
| Token tiết kiệm thật, thời gian chạy | bước 3 |
| Replay khớp chạy thật | bước 3 trên câu hỏi / seed mới, so với replay trên cùng các câu đó |
| Độ tin cậy khi triển khai (E5) | bước 4 |

---

## B. 15 thông tin của mỗi đáp án (đầu vào của observer)

### B.1 Đáp án ứng viên và các ký hiệu

Tại mỗi mốc, **đáp án ứng viên** là mọi đáp án đã từng xuất hiện trong phần debate đã quan sát (ví dụ `18` và `20`). Mỗi ứng viên được mô tả bằng 15 con số, chỉ dùng thông tin đã có tới mốc đó. Code: `osc/features.py`, hàm `candidate_features`.

Ký hiệu:
- **Agent:** solver A, solver B, critic (đáp án tự giải của critic).
- **c:** confidence mà agent tự khai cho đáp án của mình, trong khoảng 0–1.
- **Vai trò của một lượt nói** (so với vòng ngay trước):
  - **R0:** vòng đầu tiên, trước khi agent đọc ai;
  - **HOLD:** agent giữ nguyên đáp án của chính mình ở vòng trước;
  - **COPY:** agent đổi sang một đáp án mà agent **khác** đã nêu ở vòng trước;
  - **NOVEL:** agent đổi sang một đáp án chưa ai nêu ở vòng trước.
- **Sớm / muộn:** chia các vòng đã quan sát làm đôi. Vòng t là "muộn" nếu t ≥ (số vòng đã quan sát)/2, tối thiểu là 1.
- **Mốc giữa vòng:** 2 solver đã trả lời nhưng critic chưa, nên ô của critic trong vòng đó để trống và không đóng góp gì.

### B.2 Bảng 15 thông tin

| # | Tên trong code | Cách tính cho đáp án a | Ý nghĩa |
|---|---|---|---|
| 1 | `first_round_confidence` | tổng c của các agent chọn a ở vòng 0 | ý kiến độc lập ban đầu, chưa bị ai ảnh hưởng |
| 2 | `hold_early` | tổng c của các lượt HOLD chọn a, ở nửa đầu | giữ ý sớm |
| 3 | `hold_late` | như trên, ở nửa sau | giữ ý muộn; lặp lại nên thường mang ít bằng chứng mới |
| 4 | `copy_early` | tổng c của các lượt COPY sang a, nửa đầu | bị agent khác thuyết phục sớm |
| 5 | `copy_late` | như trên, nửa sau | bị thuyết phục muộn |
| 6 | `novel_early` | tổng c của các lượt NOVEL ra a, nửa đầu | đáp án mới nảy ra sớm |
| 7 | `novel_late` | như trên, nửa sau | đáp án mới nảy ra muộn |
| 8 | `mass_solver_a` | tổng c của solver A cho a, qua mọi vòng | để observer học độ tin cậy riêng của solver A |
| 9 | `mass_solver_b` | tổng c của solver B cho a | như trên, cho solver B |
| 10 | `mass_critic` | tổng c của critic cho a | như trên, cho critic |
| 11 | `critic_verdict_mass` | mỗi lượt solver chọn a: **+c_verdict** nếu critic nói agree với solver đó, **−c_verdict** nếu disagree; mỗi lượt critic tự chọn a: +c của critic | critic xác nhận hay bác bỏ đáp án |
| 12 | `share_of_agents_backing` | số agent (trong 3) từng chọn a, chia 3 | độ phủ: bao nhiêu agent khác nhau từng ủng hộ |
| 13 | `first_appearance` | vòng a xuất hiện lần đầu, chia số vòng đã quan sát | 0 = có từ đầu; gần 1 = vừa mới xuất hiện |
| 14 | `held_in_latest_round` | 1 nếu ở vòng mới nhất còn ít nhất một agent chọn a, ngược lại 0 | đáp án còn "sống" không |
| 15 | `rounds_as_majority` | tỉ lệ số vòng mà a là đáp án đa số (hoà thì chọn đáp án có confidence cao nhất) | ổn định theo thời gian |

Nhóm 1–7 tách **cùng một lượng confidence theo cách agent đi đến đáp án**. Đây là điểm khác cốt lõi với bỏ phiếu: bỏ phiếu cộng mọi lượt như nhau, còn observer học được rằng một lượt HOLD lặp lại đáng ít hơn một ý kiến độc lập ở vòng đầu.

### B.3 Ví dụ tính tay (khớp với output của code)

Debate 2 vòng, câu hỏi có đáp án đúng là 18:

| | Solver A | Solver B | Critic | Critic nói về A | Critic nói về B |
|---|---|---|---|---|---|
| Vòng 0 | 18 (c = 0.90) | 20 (0.80) | 18 (0.95) | agree (0.90) | disagree (0.80) |
| Vòng 1 | 18 (0.90) — HOLD | 18 (0.85) — COPY | 18 (0.95) — HOLD | agree (0.90) | agree (0.85) |

Có 2 vòng nên vòng 1 là "muộn".

| # | Thông tin | Đáp án 18 | Đáp án 20 | Tính |
|---|---|---|---|---|
| 1 | first_round_confidence | 1.85 | 0.80 | 0.90 + 0.95 ; 0.80 |
| 3 | hold_late | 1.85 | 0 | A giữ 0.90 + critic giữ 0.95 |
| 5 | copy_late | 0.85 | 0 | B chép sang 18 |
| 8 | mass_solver_a | 1.80 | 0 | 0.90 + 0.90 |
| 9 | mass_solver_b | 0.85 | 0.80 | vòng 1 ; vòng 0 |
| 10 | mass_critic | 1.90 | 0 | 0.95 + 0.95 |
| 11 | critic_verdict_mass | 4.55 | −0.80 | (0.90 + 0.95) + (0.90 + 0.85 + 0.95) ; B bị disagree 0.80 |
| 12 | share_of_agents_backing | 1.00 | 0.33 | 3/3 ; 1/3 |
| 13 | first_appearance | 0 | 0 | cả hai có từ vòng 0 |
| 14 | held_in_latest_round | 1 | 0 | vòng 1 không còn ai chọn 20 |
| 15 | rounds_as_majority | 1.00 | 0 | 18 là đa số cả 2 vòng |

Các ô còn lại (2, 4, 6, 7) bằng 0.

### B.4 Từ 15 con số ra xác suất

1. **Chuẩn hoá:** mỗi thông tin được trừ trung bình và chia độ lệch chuẩn tính trên dữ liệu train, để 15 con số cùng thang đo.
2. **Điểm:** `điểm(a) = w₁·x₁(a) + … + w₁₅·x₁₅(a)`.
3. **Xác suất:** `P(a) = e^(điểm(a)/T) ÷ Σ_b e^(điểm(b)/T)`. Các ứng viên cạnh tranh với nhau, tổng xác suất bằng 1.
4. **Học w:** chọn w sao cho tổng `ln P(đáp án đúng)` trên các debate cũ lớn nhất (conditional logit, phạt L2 0.001). Bài toán lồi, nên nghiệm tối ưu là duy nhất.
5. **Chọn T:** trên debate không dùng để học w, chọn T sao cho "observer nói 80%" nghĩa là đúng khoảng 80%.
6. Mỗi mốc có bộ w và T riêng: 12 mốc × 15 = 180 trọng số.

Confidence là con số model **tự khai** và thường bị thổi phồng. Observer không dùng nó trực tiếp: trọng số w học từ đáp án đúng/sai thật, nên một model hay tự tin quá mức sẽ tự động bị giảm trọng số.

---

## 1. Các claim của bài

- **C1, dừng có bảo đảm.** `P(harm ≤ ε) ≥ 1 − δ`, với harm = "chạy đủ thì đúng, dừng sớm thì sai". OSC phải thắng **fixed-depth hull ngẫu nhiên hoá ở cùng chi phí**, và khoảng tin cậy bootstrap theo câu hỏi không chứa 0 trên ít nhất 2/3 task.
- **C2, chỉ kiểm chứng khi đáng giá.** Vùng VERIFY có dạng đóng (Mệnh đề 1). OSC-V đạt accuracy bằng always-verify với ít lần gọi verifier hơn rõ rệt, hoặc đạt accuracy cao hơn ở cùng chi phí.
- **C3, bảo đảm còn đúng khi triển khai.** Monitor dùng e-process (anytime-valid): có thể kiểm sau mỗi câu mà xác suất báo động sai vẫn ≤ δ. Audit có trọng số cho khoảng tin cậy của risk hẹp hơn audit đều khi cùng số lần chạy đủ vòng.
- **C4, phân tích.** Lượng hoá mức đếm thừa phiếu bằng `ρ_k = θ_{k,HOLD} / θ_{k,R0}`, và chỉ ra khi nào giả định bỏ phiếu i.i.d. sai.

---

## 2. Toán

### 2.1 Ký hiệu

- Debate `D` có các mốc `t = 1..K`, với K = 12: 6 vòng × {sau 2 solver, cuối vòng}.
- `𝓕_t` là thông tin đã quan sát tới mốc t.
- `𝒜_t` là tập đáp án ứng viên đã xuất hiện, `y` là đáp án đúng.
- Observer cho `P_t(a) = P(y = a | 𝓕_t)`. Đáp án dẫn đầu là `â_t = argmax P_t`, và `p_t = P_t(â_t)`, `m_t = P_t(â_t) − P_t(second)`.
- `c_t` là chi phí đã tiêu tới mốc t (phần của chi phí chạy đủ). `c_V` là chi phí một lần gọi verifier.
- Loss harm: `L = 1[â_K đúng] · 1[â_τ sai]`, với τ là mốc dừng.
- Loss nhãn-tự-do: `L' = 1[â_τ ≠ â_K]`.

### 2.2 Observer (giữ nguyên)

```
P_t(a) = exp(w_tᵀ x_t(a) / T_t) / Σ_{b∈𝒜_t} exp(w_tᵀ x_t(b) / T_t)
```

- `x_t(a)` là 15 thông tin ở mục B; `w_t` học bằng conditional logit với phạt L2.
- `T_t` chọn trên held-out để giảm ECE.
- Mô hình lồi theo `w_t`, nên nghiệm tối ưu là duy nhất.

### 2.3 Mô hình verifier và cập nhật Bayes

Verifier chấm đáp án dẫn đầu, trả về `v ∈ {pass, fail}`. Lỗi của verifier được phép tương quan với lỗi của debate (blind spot) thông qua trạng thái `s = (m_t, số agent đồng ý, t)`:

```
τ₁(s) = P(v = pass | â đúng, s)          τ₀(s) = P(v = pass | â sai, s)
```

`τ₁` và `τ₀` là hai logistic regression theo `s`, học trên dữ liệu có nhãn. Cập nhật:

```
logit p⁺ = logit p + log(τ₁/τ₀)                  (pass)
logit p⁻ = logit p + log((1−τ₁)/(1−τ₀))          (fail)
```

Thông tin mà verifier mang lại được đo bằng
`I(s) = τ₁(s) − τ₀(s)` (Youden). Blind spot là vùng `s` mà `I(s) ≈ 0`, thường là lúc các agent đồng thuận.

### 2.4 Mệnh đề 1: vùng VERIFY

**Giả thiết.**
- Có 2 ứng viên, `p = P(â đúng) ≥ ½`.
- Gọi verifier tốn `κ = λ·c_V`, với λ là giá của một đơn vị chi phí tính theo đơn vị accuracy.
- Sau khi verify thì phải dừng: pass thì chọn â, fail thì chọn đáp án còn lại.
- `τ₁ > τ₀`.

**Mệnh đề.** Lợi ích kỳ vọng của việc verify so với STOP ngay là

```
G(p) = p·τ₁ + (1−p)(1−τ₀) − p = (1−p)(1−τ₀) − p(1−τ₁)
```

Hàm này tuyến tính và giảm theo p. Suy ra VERIFY tốt hơn STOP khi và chỉ khi

```
p ≤ u(s) := (1 − τ₀(s) − κ) / (2 − τ₀(s) − τ₁(s))
```

**Chứng minh.**
- STOP cho accuracy p.
- Khi VERIFY: đáp án cuối đúng nếu (â đúng và pass), xác suất `pτ₁`; hoặc (â sai và fail), xác suất `(1−p)(1−τ₀)`.
- Trừ đi p rồi giải bất phương trình `G(p) ≥ κ` theo p, được ngưỡng u.
- Quy tắc "fail thì đổi đáp án" là hợp lý khi và chỉ khi `p⁻ < ½`, tức `p < (1−τ₀)/(2−τ₀−τ₁) = u|_{κ=0}`. Điều này nhất quán với vùng verify. ∎

**Hệ quả 1 (blind spot làm mất vùng verify).**
- Nếu `τ₀ → τ₁` thì `u → (1−τ₀−κ) / (2(1−τ₀)) < ½`, tức là không bao giờ nên verify.
- Nếu verifier hoàn hảo (`τ₁ = 1`, `τ₀ = 0`) thì `u = 1 − κ`, tức là verify mọi debate chưa chắc chắn tuyệt đối.

**Hệ quả 2 (cấu trúc 3 vùng).** Kết hợp với ngưỡng dừng `a_t` của OSC, chính sách có dạng:

```
STOP    nếu  m_t ≥ a_t  và  p_t > u(s_t)
VERIFY  nếu  ℓ_t ≤ p_t ≤ u(s_t)     (mỗi debate verify tối đa một lần)
DEBATE  nếu ngược lại
```

Với nhiều ứng viên, khi fail thì không đổi đáp án mà chạy tiếp debate, và cập nhật `P_t(â) ← p⁻` (chuẩn hoá lại các ứng viên khác). Trường hợp này không có dạng đóng. Họ chính sách được tham số hoá bởi `(a, b, κ, ℓ)`. Giá trị κ vẫn quyết định u(s) qua công thức trên, nên có ý nghĩa kinh tế thay vì chỉ là một hyperparameter.

### 2.5 Bảo đảm lúc huấn luyện (LTT theo đường Pareto)

- Chia câu hỏi thành hai nửa **H₁** và **H₂**; trong mỗi câu hỏi chỉ lấy một debate để đảm bảo độc lập.
- Trên H₁, tìm đường Pareto (chi phí, harm) trên lưới `(a, b, κ, ℓ)`, rồi sắp xếp các điểm trên đường từ an toàn nhất tới mạo hiểm nhất. Thứ tự này cố định trước khi nhìn H₂.
- Trên H₂, kiểm lần lượt `H₀^{(j)}: R_harm(π_j) > ε` bằng
  `p_j = BinomCDF(k_j; n, ε)`. Dừng ở lần đầu tiên `p_j > δ`.
- **Định lý (fixed-sequence testing).** Chính sách được chọn thoả `P(R_harm > ε) ≤ δ`. Kết quả đúng mà không cần harm đơn điệu theo j, vì thứ tự được cố định từ H₁.
- **Ràng buộc thứ hai:** chi phí verify trung bình không vượt ngân sách B_V. Ràng buộc này kiểm được bằng thống kê trực tiếp vì không phải là risk cần bảo đảm.
- **Cỡ mẫu:** chạy `assert n_H₂ ≥ ln δ / ln(1−ε)` trước khi train, và cảnh báo sớm thay vì âm thầm trả về `a = None`.

### 2.6 Bảo đảm khi triển khai: e-process

- Audit i: câu hỏi vẫn được chạy đủ vòng, rồi ghi `L'_i = 1[â_τ ≠ â_K]`.
- Giả thuyết cần phát hiện là drift. `H₀: E[L'] ≤ ε`.

```
K_n = Π_{i≤n} (1 + μ_i (L'_i − ε)),     μ_i ∈ [0, 1/ε], chỉ phụ thuộc dữ liệu trước i
```

- Dưới H₀, `E[K_n | 𝓕_{n−1}] ≤ K_{n−1}` và `K_n ≥ 0`, nên K là supermartingale không âm.
- **Ville:** `P(∃n: K_n ≥ 1/δ) ≤ δ`. Có thể báo động bất cứ lúc nào `K_n ≥ 1/δ`, kiểm sau mỗi câu, mà không bị lạm phát lỗi do nhìn trộm.
- Cách chọn μ_i (Kelly, dạng plug-in): `μ_i = clip((r̂_{i−1} − ε) / (ε(1−ε)), 0, 1/(2ε))`, với `r̂` là tỉ lệ loss trung bình đã làm trơn. Đây là nghiệm của `max_μ E log(1 + μ(L−ε))` khi `L ~ Bern(r)`.
- Khi có báo động: tăng `a` (và giảm `u`) theo lưới, replay trên các câu đã audit, rồi khởi động lại e-process.

### 2.7 Audit có trọng số (Horvitz–Thompson, phân bổ Neyman)

- Xác suất audit `π(s_τ)` phụ thuộc trạng thái tại lúc dừng, với `π ≥ π_min`.
- Ước lượng `L̂_i = L'_i · A_i / π(s_i)` không chệch và bị chặn trong `[0, 1/π_min]`, nên đưa thẳng được vào e-process ở 2.6 thay cho `L'_i`. Điều kiện `1 − μ_i ε ≥ 0` vẫn đúng.
- **Tối ưu.** Cực tiểu `Var(L̂) = E[r(s)/π(s)] − r̄²` với ràng buộc `E[π(s)] = B`. Dùng nhân tử Lagrange: `−r(s)/π(s)² + ν = 0`, suy ra **`π*(s) ∝ √r(s)`**, với `r(s) = P(L' = 1 | s)` ước lượng từ dữ liệu train.
- Ý nghĩa: audit nhiều ở những lần dừng có margin thấp và ít ở những lần dừng rất chắc chắn.

### 2.8 Phân tích đếm thừa (dùng mô hình đệ quy làm công cụ đo)

```
λ_t(a) = γ λ_{t−1}(a) + Σ_{k∈{A,B,C}} Σ_{r∈{R0,HOLD,COPY,NOVEL}} θ_{k,r} c_{k,t} 1[k ủng hộ a ở t với vai trò r]
                      + θ_v Σ_{k∈{A,B}} sgn(verdict_{k,t}) c^v_{k,t} 1[k ủng hộ a]
```

- Với γ cố định, log-likelihood lõm theo θ.
- Báo cáo `ρ_k = θ_{k,HOLD} / θ_{k,R0}` với khoảng tin cậy bootstrap theo câu hỏi.
- So sánh held-out NLL với mô hình ràng buộc "vote3" (`θ_{k,r} = θ_k`, không verdict). Đây là kiểm định thực nghiệm cho giả định bỏ phiếu i.i.d. Không dùng χ², vì likelihood gộp nhiều mốc của cùng một debate nên các mẫu không độc lập.

---

## 3. Việc cần làm trong code

| # | File | Nội dung | Kiểm thử | Trạng thái |
|---|---|---|---|---|
| 1 | `osc/certify.py`, `osc/train.py` | Kiểm tra cỡ mẫu ở 2.5 (`min_questions`, `smallest_eps`): cảnh báo trước khi train và in ra ε nhỏ nhất khả thi | `test_sample_size_floor` | **xong** |
| 2 | `src/agents/verifier.py`, `src/verification/rules.py`, `tasks/base.py` | Verifier: pass/fail/unknown + confidence + đáp án tự giải; chế độ `reasoning` và `blind`; kiểm tra bằng luật (AST an toàn, không `eval`); prompt chung ở `BenchmarkConfig.build_verifier_prompt` | `test_verifier_*`, `test_*_rules*` | **xong** |
| 3 | `osc/verify_offline.py` | Chạy verifier offline; mỗi cặp (câu hỏi, đáp án) 1 lần, dùng chung giữa các seed; chạy lại = tiếp tục; `--max_questions` cho pilot G1 | `test_offline_pairs_are_deduplicated_across_seeds` | **xong** |
| 4 | `osc/verify.py` | `TauModel` (τ₁, τ₀ logistic theo trạng thái s, mỗi debate trọng số 1); `verify_band_upper` (Mệnh đề 1); `log_lr` | `test_proposition1_*`, `test_bayes_*`, `test_tau_model_*` | **xong** |
| 5 | `osc/verify.py`, `osc/train.py` (`fit_policy_v`), `osc/policy.py` | Chính sách 3 hành động; mô phỏng vector hoá (margin sau verify tính sẵn cho mọi mốc); Pareto testing: τ và đường Pareto trên H₁, fixed-sequence trên H₂; "không kiểm" nằm trong lưới | `test_never_verify_*`, `test_informative_*`, `test_certify_v_*` | **xong** |
| 6 | `src/orchestrator.py`, `MAD.py` | VERIFY lúc chạy thật, tối đa 1 lần/câu; log `osc.verify` và `osc.verify_tokens`; `--verifier_config` | `test_orchestrator_verifies_once_and_counts_tokens` | **xong** |
| 7 | `osc/monitor.py` | E-process 2.6 thay cho binomial lặp; `--audit weighted` theo 2.7; log `π_i` cho từng câu | mô phỏng: không drift thì tỉ lệ báo động sai ≤ δ; có drift thì đo độ trễ phát hiện | chưa |
| 8 | `src/orchestrator.py` | `_is_audited` dùng `π(s_τ)` tại lúc dừng (hash tất định theo câu hỏi, so với π) | test tần suất audit ≈ E[π] | chưa |
| 9 | `osc/evaluate.py` | `--verify_logs`: OSC-V và verify-at-stop, nested, chi phí có cả token verifier, so với fixed-depth hull và với OSC; τ₁/τ₀ trên fold test theo đồng thuận / không đồng thuận | smoke test trên log cũ với verdict giả lập | **xong** (còn thiếu rescue/hurt) |
| 10 | `osc/analysis_evidence.py` | Mô hình ở 2.8 (chuyển từ script pilot `pilot_rec.py`, `RecCache` + `fit_shared`): bảng ρ_k và so sánh vote3 | – | chưa |
| 11 | `tasks/choice.py`, `tasks/{commonsenseqa,truthfulqa,bbh}/`, `tasks/hf_download.py`, `src/communication/protocol.py` | Benchmark mới: CommonsenseQA (1.221), TruthfulQA MC1 (817), BBH 23 sub-task (1.200); parser đọc chữ cái A–R | `test_letter_extraction`, `test_parser_*`, `test_new_benchmarks_*` | **xong** |
| 12 | `src/agents/base_agent.py`, `MAD.py`, `config/vllm/`, `kaggle/start_vllm.sh` | Provider `vllm` (API tương thích OpenAI, gọi bằng `requests`); seed riêng từng agent (an toàn khi chạy song song); `MAD.py --workers` và `--solver_a_config/--solver_b_config/--critic_config`; script khởi động server cho 2×T4 | `tests/test_backend.py` (server vLLM giả, chạy song song, seed từng phiếu) | **xong** |
| 13 | `baselines/` | `no_debate.py` (chuyển từ `mv_baseline.py`; thêm single agent, `--workers`); `stopping.py`: fixed depth, consensus, Adaptive-Consistency Beta, Wald SPRT, ổn định phân bố, replay trên log; `osc.evaluate --baselines` so ở cùng chi phí | `tests/test_baselines.py` | **xong** |

**Chi phí verifier offline:** trên log GSM8K cũ có 3.648 cặp (câu hỏi, đáp án) cho 978 câu hỏi, tức khoảng 3,7 lần gọi mỗi câu hỏi (mọi seed dùng chung). Pilot G1 với 300 câu hỏi cần khoảng 1.100 lần gọi.

---

## 4. Thu dữ liệu

- **Model:** chạy `--critic_mode live` và `--stop none`, mỗi bộ model lưu vào một thư mục log riêng. Hai lựa chọn:

  | | Solver A | Solver B | Critic | Verifier | Chạy ở đâu |
  |---|---|---|---|---|---|
  | Kaggle, miễn phí (mặc định) | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct | Mistral-7B-Instruct-v0.3 (Gemma-2 không chạy fp16 trên T4) | Phi-4 14B | vLLM, AWQ 4-bit, 2×T4 (`config/vllm/`, `kaggle/start_vllm.sh`) |
  | Qua API, trả phí | GPT-4o-mini | DeepSeek | Llama-3.3-70B (Groq/OpenRouter) | model khác họ | provider `chatgpt` / `deepseek` / `groq` / `openrouter` có sẵn |

  Llama-3.3-70B không chạy được trên GPU Kaggle (bản 4-bit cần khoảng 40GB, 2×T4 chỉ có 32GB). Verifier nên khác họ với cả ba debater để τ₀ thấp.
- **Thời gian trên Kaggle:** quota khoảng 30 giờ GPU/tuần, mỗi phiên tối đa 12 giờ. Đo thời gian thực bằng `--n 20` trước. Nếu không đủ quota cho 6 task × 600 câu thì ưu tiên GSM8K, MMLU, TruthfulQA (3 dạng đáp án khác nhau) rồi mới thêm task khác. Lưu `results/` làm output của notebook và chạy tiếp bằng `--resume`.
- **Cỡ mẫu:** **≥ 600 câu hỏi/task**, để nửa H₂ có ≥ 300 câu và certify được ε = 1% với k ≤ 0 (hoặc ε = 2% với k ≤ 2). Nếu ngân sách chỉ đủ 300 câu thì báo cáo ở ε = 2% và 5%, và chọn hình dạng ngưỡng (b, extra) trên một task khác để khỏi phải chia đôi.

  Lý do: để certify được (khi không có câu nào bị hại) cần `(1−ε)^n ≤ δ`, tức `n ≥ ln δ / ln(1−ε)` câu hỏi độc lập **trong nửa dùng để certify**. Với ε = 1%, δ = 0.1 là 230 câu; 300 câu chia đôi chỉ còn 150 nên không thể certify. `osc.train` in cảnh báo trước khi chạy nếu thiếu. Số câu bị hại tối đa vẫn được chấp nhận (δ = 0.1):

  | ε \ n (nửa certify) | 150 | 300 | 600 | 1200 |
  |---|---|---|---|---|
  | 1% | không thể | 0 | 2 | 7 |
  | 2% | 0 | 2 | 7 | 17 |
  | 5% | 3 | 9 | 22 | 49 |

- **Seed:** 2–3 seed mỗi câu. Seed dùng cho đánh giá và bootstrap, còn certify chỉ lấy 1 debate mỗi câu.
- **Token:** log mới có token của từng lần gọi (`tokens` trong round record), nên chi phí sẽ là token thật thay vì số lần gọi.
- **Verifier:** chạy offline (việc #3) sau khi có log. Dùng cùng họ model hoặc model khác; ghi rõ vì ảnh hưởng tới τ₀.
- **Benchmark:** GSM8K, MMLU, StrategyQA, CommonsenseQA, TruthfulQA, BBH (đã có dữ liệu, cả ba benchmark mới đều ≥ 600 câu). Nên thêm một task có đáp án kiểm được (MATH hoặc SVAMP), nơi verifier có khả năng cao là mang thông tin. TruthfulQA đáng chú ý cho E3: câu hỏi được thiết kế để đáp án phổ biến là sai, nên đây là nơi dễ thấy blind spot (τ₀ cao) nhất.

---

## 5. Thí nghiệm

**Baseline** (chi tiết và lệnh chạy ở README, mục Baseline):

| Nhóm | Baseline | Gọi model thêm? |
|---|---|---|
| Không debate | single agent; self-consistency@3; ensemble vote 3 model | có (`baselines/no_debate.py`) |
| Debate, dừng cố định | fixed-depth 1–6 vòng; fixed-depth ngẫu nhiên hoá cùng chi phí | không (replay) |
| Debate, dừng theo luật | consensus; Adaptive-Consistency (Beta); Wald SPRT; ổn định phân bố đáp án | không (replay, `baselines/stopping.py`) |
| Debate + kiểm chứng | verify-at-stop (luôn verify đáp án lúc dừng) | không (replay với verdict offline) |
| Hệ thống debate có verifier bên ngoài | chạy repo gốc của hệ đó với cùng model và cùng câu hỏi | có (ngoài repo này) |

Mọi so sánh dừng sớm đều ở **cùng chi phí trung bình**. Luật dừng được chọn cách pha trộn tham số tốt nhất ngay trên dữ liệu đánh giá, nên so sánh nghiêng về phía baseline.

| ID | Câu hỏi | Cách đo | Thành công khi |
|---|---|---|---|
| E1 | OSC / OSC-V có thắng fixed-depth **và các luật dừng** ở cùng chi phí không? | `osc.evaluate --baselines`: đường cost–accuracy; Δ so với hull ngẫu nhiên hoá của từng baseline; bootstrap 2000 lần theo câu hỏi | CI không chứa 0 trên ≥ 2/3 task, với mọi luật dừng |
| E2 | Bảo đảm có đúng không? | 200 lần chia H₁/H₂ ngẫu nhiên; tỉ lệ `harm > ε` | ≤ δ |
| E3 | Verifier có thông tin không, blind spot nằm ở đâu? | Đường τ₁(s), τ₀(s), I(s) theo margin và đồng thuận; vùng u(s) | I(s) > 0.2 ở vùng margin trung bình |
| E4 | VERIFY có đáng tiền không? | OSC-V so với always-verify và never-verify; số lần gọi verifier; rescue/hurt | Cùng accuracy với ≤ 50% số lần gọi, hoặc +acc ở cùng chi phí |
| E5 | Monitor | Mô phỏng drift (đổi model solver, trộn task, nhiễu confidence); không drift thì đo báo động sai | Báo động sai ≤ δ; có drift thì phát hiện trong vài trăm câu |
| E6 | Audit có trọng số | Độ rộng CI của risk khi π đều so với π* ∝ √r, cùng số lần chạy đủ vòng | Hẹp hơn rõ rệt |
| E7 | Đếm thừa | Bảng ρ_k; NLL của mô hình vai trò so với vote3 | ρ_k < 1 có ý nghĩa ở ≥ 1 agent |

**Metric báo cáo:** accuracy, token trung bình, số vòng trung bình, harm, rescue (dừng/verify sửa được câu sai), hurt (làm hỏng câu đúng), blind-spot rate (đồng thuận sai mà verifier vẫn pass), số lần gọi verifier.

---

## 6. Điểm quyết định (gate)

- **G0, trước khi thu dữ liệu:** việc #1 xong, và số câu hỏi đủ theo bảng cỡ mẫu ở mục 4.
- **G1, sau E3 trên khoảng 300 debate đầu tiên:** nếu `I(s) < 0.1` trên toàn bộ vùng margin có debate dừng, hoặc `u(s) < ℓ` ở mọi nơi với κ thực tế, thì **bỏ VERIFY**. Khi đó bài gồm C1 + C3 + C4.
- **G2, sau E1:** nếu OSC không thắng hull ở task nào thì dừng lại, xem lại observer trước khi viết.
- **G3, trước khi viết:** E2 đạt trên mọi task được báo cáo.

---

## 7. Lịch

| Tuần | Việc |
|---|---|
| 1 | Việc #1, #2, #3 và #6 (khung); thu 300 debate đầu tiên cho GSM8K |
| 2 | Việc #4; chạy E3 và quyết định G1; tiếp tục thu dữ liệu |
| 3 | Việc #5, #9; E1, E2, E4 trên GSM8K |
| 4 | Việc #7, #8; E5, E6 (mô phỏng, không cần gọi thêm model) |
| 5 | Chạy đủ 3–4 task; việc #10 và E7 |
| 6 | Bảng, hình, chứng minh đầy đủ trong phụ lục; bản nháp |

---

## 8. Rủi ro

| Rủi ro | Dấu hiệu | Cách xử lý |
|---|---|---|
| Verifier chung blind spot với debate | τ₀ ≈ τ₁ ở vùng đồng thuận | Verifier khác họ model; verifier dạng công cụ (chạy code / tính lại số) cho GSM8K |
| Không đủ câu hỏi để certify | assert ở việc #1 báo lỗi | Nới ε, chọn hình dạng ngưỡng trên task khác, thu thêm |
| Critic live làm observer đổi hành vi | ECE tăng so với log cũ | Học lại w, T trên log mới; không trộn với log cũ |
| StrategyQA không có tín hiệu | AUC của margin ≈ 0.6 như log cũ | Báo cáo trung thực là giới hạn; bảo đảm vẫn đúng, chỉ là không tiết kiệm được |
| Bảo đảm là của quy trình, không phải của đúng bộ w đang triển khai | – | Học w chỉ trên H₁ cho bản certify; monitor (2.6) kiểm lại trên đúng bộ w đó |
