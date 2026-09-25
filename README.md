# OSC — Quan sát → Dừng / Kiểm chứng → Bảo đảm

Dừng sớm có bảo đảm cho multi-agent debate (2 solver + 1 critic), có thêm hành động **kiểm chứng** (VERIFY) bằng một verifier riêng. Kế hoạch chi tiết và phần toán nằm ở `PLAN_OSC_V.md`.

Cấu trúc: `MAD.py` chạy debate · `src/` agent (solver, critic, verifier), orchestrator, `src/verification/rules.py` kiểm tra bằng luật · `tasks/` prompt + chấm điểm từng benchmark · `data/` câu hỏi · `config/` model (`config/vllm/` cho vLLM) · `osc/` framework dừng · `baselines/` các baseline · `kaggle/` script chạy trên Kaggle · `tests/`.

Quyết định sau mỗi mốc của debate: **dừng** (trả lời bằng đáp án dẫn đầu) hay **chạy tiếp**, kèm cam kết bằng số:
"dừng sớm làm hỏng tối đa ε câu, với độ tin 1 − δ".

## Cách hoạt động

**1. Quan sát** (`features.py`, `observer.py`). Mỗi đáp án ứng viên có 15 con số mô tả (ai nêu ở vòng đầu, bao nhiêu lần được giữ nguyên / chép theo / nêu mới, critic đồng ý hay phản đối, …).
- `điểm(a) = w₁·x₁(a) + … + w₁₅·x₁₅(a)`
- `%(a) = e^(điểm(a)/T) ÷ tổng các e^(điểm(b)/T)`
- Trọng số w học từ debate cũ: làm cho tổng ln(% gán cho đáp án đúng) lớn nhất, trừ 0.001 × tổng bình phương w.
- T chọn trên debate *không dùng để học w*, sao cho "nói 80% thì đúng khoảng 80%".
- Mỗi mốc có một bộ w, T riêng.

**2. Dừng** (`certify.py`, `policy.py`).
- Có 2 mốc mỗi vòng: `rXs` (sau 2 solver, trước critic) và `rXf` (cuối vòng).
- `khoảng cách = %(đáp án cao nhất) − %(đáp án cao nhì)`. Dừng khi khoảng cách ≥ ngưỡng.
- Ngưỡng: `a − b·(số thứ tự vòng)` ở cuối vòng và `a − b·(số thứ tự vòng) + extra` ở giữa vòng. Mốc cuối cùng luôn dừng.

**3. Bảo đảm** (`certify.py`, `train.py`).
- Câu "bị hại" = chạy đủ thì đúng nhưng dừng sớm lại sai (`--risk disagree`: dừng sớm khác đáp án chạy đủ).
- Chia câu hỏi làm hai nửa: nửa 1 chọn b và extra; nửa 2 chọn a bằng phép thử
  `P = Σ_{i=0..k} C(n,i)·ε^i·(1−ε)^(n−i) ≤ δ`
  với n câu (mỗi câu hỏi lấy 1 debate), k câu bị hại. Thử a từ an toàn nhất xuống, dừng ở lần trượt đầu tiên.
- Nếu dữ liệu quá ít để cam kết thì policy không bao giờ dừng sớm (`certified_a = null`). Với ε = 1%, δ = 10% cần ít nhất 230 câu hỏi khác nhau ở nửa 2.

**4. Kiểm chứng (VERIFY)** (`src/agents/verifier.py`, `osc/verify.py`). Verifier chấm **một** đáp án: `pass` / `fail` (không đọc được thì `unknown`, coi như không có bằng chứng), kèm đáp án tự giải và kết quả kiểm tra bằng luật (không tốn token: phép tính sai, đáp án không có trong lập luận, kết luận mâu thuẫn). Có 2 chế độ: `reasoning` (xem cả lập luận) và `blind` (chỉ xem câu hỏi + đáp án).
- Không tin verdict nguyên văn: học `τ₁(s) = P(pass | đúng, s)` và `τ₀(s) = P(pass | sai, s)` theo trạng thái debate `s` (τ₀ cao khi verifier cùng "điểm mù" với debate).
- Verdict cộng `log(τ₁/τ₀)` (pass) hoặc `log((1−τ₁)/(1−τ₀))` (fail) vào logit của đáp án được kiểm, giữ đến hết debate.
- Chỉ kiểm khi đáng: `ℓ ≤ p ≤ u(s) = (1 − τ₀ − κ)/(2 − τ₀ − τ₁)` (Mệnh đề 1). Mỗi câu kiểm tối đa 1 lần.
- (a, κ, ℓ) được cam kết cùng lúc (Pareto testing). "Không bao giờ kiểm" nằm trong lưới, nên nếu verifier vô ích thì kết quả quay về OSC thường.

**5. Tự kiểm khi chạy thật** (`monitor.py`, nâng cấp N4). `--audit_rate` (mặc định 5%) số câu vẫn chạy đủ vòng và ghi lại chỗ OSC *định* dừng. Monitor đếm số câu mà đáp án lúc dừng khác đáp án chạy đủ, dùng cùng phép thử P. Nếu trượt thì tăng a và có thể ghi policy mới bằng `--update`.

## Quy trình

```bash
# 1) thu debate chạy đủ 6 vòng (critic mặc định là live), để riêng thư mục cho bộ model mới
python MAD.py --task all --n 300 --seed 1 --stop none --log_root results/logs_osc
#    lặp lại với --seed 2..5

# 2) đánh giá nested: OSC so với fixed-depth cùng chi phí, consensus, majority
python -m osc.evaluate --logs "results/logs_osc/**/debate_full_*.jsonl" --task gsm8k --out results/osc/eval_gsm8k.json

# 3) huấn luyện + cam kết policy dùng thật
python -m osc.train --logs "results/logs_osc/**/debate_full_*.jsonl" --task gsm8k --out results/osc/policy_gsm8k.json --eps 0.01 --delta 0.10

# 4) chạy thật với OSC ({task} tự thay theo --task)
python MAD.py --task all --n 300 --seed 6 --stop osc --osc_policy results/osc/policy_{task}.json --log_root results/logs_osc_live

# 5) kiểm lại cam kết trên các câu tự kiểm
python -m osc.monitor --logs "results/logs_osc_live/**/debate_full_*.jsonl" --policy results/osc/policy_gsm8k.json --eps 0.02
```

Tắt dừng giữa vòng (N2): thêm `--no_mid` cho cả `osc.train` và `osc.evaluate`.

### Thêm hành động VERIFY

```bash
# a) chạy verifier offline trên log đủ vòng (mỗi cặp câu hỏi-đáp án 1 lần; chạy lại = tiếp tục)
python -m osc.verify_offline --logs "results/logs_osc/**/debate_full_*.jsonl" --task gsm8k --mode reasoning --out results/verify/gsm8k_reasoning.jsonl
#    thử trước ~300 câu để quyết định có làm tiếp không (gate G1): thêm --max_questions 300

# b) đánh giá nested: OSC-V so với OSC, verify-at-stop và fixed-depth cùng chi phí
python -m osc.evaluate --logs "results/logs_osc/**/debate_full_*.jsonl" --task gsm8k --verify_logs "results/verify/gsm8k_*.jsonl" --out results/osc/eval_v_gsm8k.json

# c) huấn luyện + cam kết policy có VERIFY
python -m osc.train --logs "results/logs_osc/**/debate_full_*.jsonl" --task gsm8k --verify_logs "results/verify/gsm8k_*.jsonl" --eps 0.02 --out results/osc/policy_v_gsm8k.json

# d) chạy thật (verifier chỉ được gọi khi policy có khối verify)
python MAD.py --task gsm8k --stop osc --osc_policy results/osc/policy_v_gsm8k.json --verifier_config config/model_config_verifier.yaml
```

`config/model_config_verifier.yaml` hiện là model giữ chỗ. Nên chọn họ model **khác** các debater để τ₀ thấp.

## Baseline (`baselines/`)

| Baseline | Loại | Cách chạy |
|---|---|---|
| Single agent (1 model, 1 lần gọi) | gọi model | `python MAD.py --majority_voting --sc_model a --mv_votes 1` |
| Self-consistency@k (1 model, k mẫu) | gọi model | `python MAD.py --majority_voting --sc_model a --mv_votes 3` |
| Ensemble vote (3 model, không debate) | gọi model | `python MAD.py --majority_voting` |
| Debate fixed-depth 1–6 vòng, dừng khi đồng thuận | replay log | `python -m baselines.stopping ...` / `osc.evaluate --baselines` |
| Adaptive-Consistency (luật Beta) | replay log | như trên |
| Wald SPRT | replay log | như trên |
| Dừng khi phân bố đáp án ổn định | replay log | như trên |
| OSC, verify-at-stop | replay log | `osc.evaluate` (`--verify_logs` cho verify-at-stop) |

- Baseline "gọi model" nằm ở `baselines/no_debate.py` (trước đây là `mv_baseline.py`), chạy qua `MAD.py` trên đúng tập câu hỏi của debate (cùng `--task --n --seed`). Log ở `results/logs_mv/` và `results/logs_sc/`.
- Baseline "replay log" nằm ở `baselines/stopping.py`: không gọi model, chạy trên log đủ vòng như OSC.
- `python -m osc.evaluate ... --baselines` so mỗi kết quả OSC / OSC-V với từng luật dừng **ở cùng chi phí**. Luật dừng được chọn cách pha trộn tham số tốt nhất ngay trên dữ liệu đánh giá, nên so sánh nghiêng về phía baseline, tức là thận trọng cho OSC. Nếu OSC rẻ hơn mức rẻ nhất mà luật đạt được, dòng kết quả ghi "(baseline spends more)".

## Chạy trên Kaggle (vLLM, GPU T4 ×2)

Ollama xử lý từng request một; trên Kaggle nên dùng vLLM (server tương thích OpenAI, gom nhiều request thành batch) và chạy nhiều câu hỏi song song (`--workers`).

```bash
# 0) Notebook: Accelerator = GPU T4 x2, Internet = On
pip install -q vllm pyyaml scipy scikit-learn pandas pyarrow requests

# 1) debate: Qwen2.5-7B + Llama-3.1-8B trên GPU 0, Mistral-7B-v0.3 (critic) trên GPU 1 (đều AWQ 4-bit)
bash kaggle/start_vllm.sh debate
python MAD.py --task gsm8k --n 600 --seed 1 --stop none --workers 16 \
    --solver_a_config config/vllm/solver_qwen.yaml --solver_b_config config/vllm/solver_llama.yaml \
    --critic_config config/vllm/critic_mistral.yaml --log_root results/logs_osc

# 2) verifier (Phi-4, khác họ với cả ba debater)
bash kaggle/start_vllm.sh stop && bash kaggle/start_vllm.sh verify
python -m osc.verify_offline --logs "results/logs_osc/**/debate_full_*.jsonl" --task gsm8k \
    --verifier_config config/vllm/verifier_phi4.yaml --workers 16 --out results/verify/gsm8k_reasoning.jsonl
```

- Không dùng Gemma-2 làm critic trên T4: vLLM từ chối Gemma-2 ở fp16 ("numerical instability"), T4 không có bf16, và bản fp32 không vừa bộ nhớ.
- `MAD.py` và `osc.verify_offline` kiểm tra mọi server vLLM trong config trước khi chạy, và dừng ngay nếu server chưa lên hoặc đang chạy model khác. `MAD.py` cũng dừng nếu mọi câu của một task đều lỗi.
- Mọi thứ sau bước 2 (`osc.train`, `osc.evaluate`, `baselines.stopping`) chạy trên CPU, không cần GPU.
- Phiên Kaggle tối đa 12 giờ và thư mục làm việc bị xoá khi hết phiên. Lưu `results/` làm output của notebook, phiên sau chép lại vào rồi chạy tiếp với `--resume`.
- Trước khi chạy lớn, thử `--n 20` để đo thời gian mỗi câu, rồi nhân lên cho đủ số câu cần.
- Token của provider `vllm` = token prompt + token sinh ra (giống các provider API). Provider Ollama chỉ đếm token sinh ra, nên không trộn log của hai backend khi so sánh chi phí.

## Benchmark

| task | dạng | số câu | tạo dữ liệu |
|---|---|---|---|
| gsm8k | số | 1319 | `tasks/gsm8k/gen_gsm.py` |
| strategyqa | yes/no | 687 | `tasks/strategyqa/gen_strategyqa.py` |
| mmlu | A–D | 2000 | `tasks/mmlu/gen_mmlu.py` |
| commonsenseqa | A–E | 1221 (validation) | `python -m tasks.commonsenseqa.gen_commonsenseqa` |
| truthfulqa | MC1, 4 lựa chọn | 817 | `python -m tasks.truthfulqa.gen_truthfulqa` |
| bbh | 23 sub-task, (A)…(R) | 1200 | `python -m tasks.bbh.gen_bbh` |

Ba benchmark mới dùng chung `tasks/choice.py` và tải thẳng file parquet của HuggingFace, không cần cài `datasets`. `--task all` chạy cả 6 benchmark.

## Kiểm thử

```bash
python tests/test_osc.py
python tests/test_verify.py
python tests/test_backend.py
python tests/test_baselines.py
```