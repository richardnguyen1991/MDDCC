# MDDCC — Multi-Dimensional Deep Convolutional Classifier

Tái hiện **Wang K., Fu Y., Duan X., Liu T., "Detection and mitigation of DDoS attacks
based on multi-dimensional characteristics in SDN"**, *Scientific Reports* 14:16421 (2024),
DOI [10.1038/s41598-024-66907-z](https://doi.org/10.1038/s41598-024-66907-z).

Wavelet DB4 (SWT, level 3) → 4 nhánh CNN độc lập → compose cộng → FC → Softmax,
loss `MSE + λ·Σσ(w)`, huấn luyện trên **CIC-DDoS2019** (Parquet) bằng **Kaggle CPU**,
checkpoint trên **AWS S3**, điều phối bằng **GitHub Actions**.

> ⚠️ README này mô tả trạng thái hiện tại của pipeline. Hướng dẫn chạy lại đầy đủ
> (kể cả bước thêm secret trên Kaggle và cách khôi phục khi run hỏng) sẽ hoàn thiện
> ở Bước 6 theo thứ tự bàn giao.

---

## Trạng thái bàn giao

| Bước | Nội dung | Trạng thái |
|---|---|---|
| 1 | Tên repo/notebook/bucket/prefix, cây thư mục, `configs/mddcc.yaml` | ✅ xong |
| 2a | Discovery Kaggle Dataset (`scripts/discover_dataset.py`) | ✅ xong — **đã chạy thật trên Kaggle 2026-08-18** |
| 2b | `data.py`, `wavelet.py`, chống rò rỉ split, loại cột, test SWT | ✅ xong |
| 2c | `stage1_switch_stats.py` — công thức (1)(2)(3) + 3-sigma (§3.G) | ✅ xong — **ngoài phạm vi đánh giá** |
| — | **Tổng test** | **242/242 pass** |
| 3 | `model.py`, `train.py`, `checkpoint.py`, `s3io.py` (resume giữa epoch từ S3) | ✅ xong |
| 4 | `evaluate.py`, `viz.py`, `make_report.py`, `explain.py` | ✅ xong |
| 5 | `kernel/`, `.github/workflows/run-kaggle.yml`, chống vòng lặp vô hạn | ✅ xong |
| 6 | README đầy đủ, chạy thử 2 epoch + kiểm tra resume, rồi chạy đủ 100 epoch | ⏳ chưa làm |

---

## Cấu hình đã chốt

Toàn bộ siêu tham số nằm trong [`configs/mddcc.yaml`](configs/mddcc.yaml). Các giá trị cốt lõi:

| Tham số | Giá trị | Nguồn |
|---|---|---|
| epochs | **100 chính xác**, không early stopping | yêu cầu thí nghiệm |
| batch_size | 4096 | yêu cầu thí nghiệm |
| learning_rate | **0.01** hằng số, không scheduler | **đúng bài báo** |
| optimizer | SGD, `momentum=0`, `weight_decay=0` | bài báo |
| loss | MSE(softmax, one-hot) + `λ_std=1.0 · Σ σ(w)` | công thức (8)(9) |
| wavelet | DB4, level 3, **SWT** (không downsampling) | mục "Two-stage attack detection" |
| subband | đúng 4: `cD1, cD2, cD3, cA3` | χ(3), n+1 = 4 |
| CNN mỗi nhánh | 32 → 64 → 32, dropout 0.2 / 0.3 / 0.2, ReLU | Table 3 |
| compose | `sum` (cộng phần tử) | công thức (10) |
| device | CPU bắt buộc | ràng buộc hạ tầng |
| feature selection | **none** — dùng tất cả feature | yêu cầu thí nghiệm (bài báo: 48/80) |
| imbalance handling | **none** | bài báo cũng không xử lý |

---

## Ngân sách tính toán (đo thực tế, không phải ước lượng giấy)

Đo trên CPU 4 luồng, PyTorch 2.12+cpu, kiến trúc MDDCC đầy đủ (151.571 tham số):

| Chỉ số | Giá trị |
|---|---|
| Train step (bs=4096, fwd+bwd+SGD) | 1,441 s → **2.842 mẫu/s** |
| Eval step (forward, `no_grad`) | 0,523 s → **7.825 mẫu/s** |
| Dataset | 70.427.637 hàng, 18 file Parquet (2,88 GB), **18 lớp** (sau gộp), 1 schema duy nhất |
| Split | train 41,9M / val 7,4M / test 21,1M |
| 1 epoch | train 4,10 h + val 0,26 h = **4,36 h** (10.231 step) |
| **100 epoch** | **436 h ≈ 18 ngày → 39 session Kaggle 11h20m** |
| Cache feature float32 | 22,8 GB (81 cột) |

**Quyết định phương án C:** giữ trọn 70,4M hàng, không cắt dữ liệu. Kéo theo:
`max_restarts` của GitHub Actions phải đặt ~60 (không phải 40).

**Không cache subband.** Cache `[N,4,9,9]` float32 = **91,3 GB**, vượt đĩa Kaggle.
Chỉ cache feature đã Min-Max `[N,F]` = 22,5 GB, SWT tính on-the-fly theo lô —
chi phí đo được 0,2 h/epoch, khoảng 5% so với 4,1 h compute.

---

## Hình học wavelet

Số đo thật sau discovery: Parquet có 90 cột, loại 9 cột định danh/provenance → **F = 81**,
tất cả đều `double`, **không còn cột phi số** nên không cần one-hot.

```
F = 81 feature  ──pad reflect──>  F_swt = ceil(81/8)*8 = 88
                ──pywt.swt(db4, level=3, axis=1)──>  4 subband × 88 (cùng độ dài chuỗi gốc)
                ──pad 0 lên S*S, S = max(8, ceil(sqrt(88))) = 10──>  ảnh [4, 10, 10]
```

`F` sẽ giảm tiếp sau bước loại cột hằng số / trùng lặp (xem mục dưới), và hình học được
tính lại theo `F` cuối cùng.

Mỗi subband đi vào một CNN riêng (**không chia sẻ trọng số**), `z = Σ zᵢ`, flatten → `Linear → Softmax`.

### Vì sao `pool_ceil_mode: true`

Table 3 có 3 lần `MaxPool2d(2×2)`. Với `S = 10` và pooling mặc định (`floor`):
`10 → 5 → 2 → 1`, tức toàn bộ 4 nhánh bị nén còn **32 số** trước lớp FC cho 19 lớp —
thắt cổ chai nghiêm trọng. Với `ceil_mode=True`: `10 → 5 → 3 → 2` → **128 chiều**.
Đây là sai khác nhỏ nhất có thể so với bài báo (giữ nguyên công thức padding và
Table 3, chỉ đổi chế độ làm tròn của pooling) và đã ghi vào `deviations_from_paper`.

### Ràng buộc `min_final_map` — chặn sụp đổ ngầm

Riêng `ceil_mode` **chưa đủ**. Công thức `S = max(8, ceil(sqrt(F_swt)))` cho `S = 8` với
mọi `F ≤ 64`, và `8 → 4 → 2 → 1` vẫn sụp về 32 chiều kể cả khi bật `ceil_mode`.
Số cột đặc trưng không cố định — nó phụ thuộc kết quả loại cột hằng số/trùng lặp — nên
tình huống này có thể xảy ra mà không ai để ý.

`compute_geometry` vì vậy **nâng `S` lên** cho tới khi feature map cuối ≥ `min_final_map`
(mặc định 2), ghi lại giá trị gốc vào `side_bumped_from`, và **từ chối** `force_side`
vi phạm ràng buộc. Với `F = 81` hiện tại thì `S = 10` đã an toàn, không cần nâng.

---

## Phân bố lớp thật (discovery 2026-08-18)

Dữ liệu thô có 19 nhãn; sau khi gộp `UDP-lag` → `UDPLag` còn **18 lớp**,
mất cân bằng **1 : 45.746** giữa `TFTP` và `WebDDoS`:

| Lớp | Số mẫu | % | | Lớp | Số mẫu | % |
|---|---:|---:|---|---|---:|---:|
| TFTP | 20.082.580 | 28,52 | | DrDoS_SSDP | 2.610.611 | 3,71 |
| Syn | 6.473.789 | 9,19 | | DrDoS_LDAP | 2.179.930 | 3,10 |
| MSSQL | 5.787.453 | 8,22 | | LDAP | 1.915.122 | 2,72 |
| DrDoS_SNMP | 5.159.870 | 7,33 | | DrDoS_NTP | 1.202.642 | 1,71 |
| DrDoS_DNS | 5.071.011 | 7,20 | | UDPLag † | 368.334 | 0,52 |
| DrDoS_MSSQL | 4.522.492 | 6,42 | | Portmap | 186.960 | 0,27 |
| DrDoS_NetBIOS | 4.093.279 | 5,81 | | **BENIGN** | **113.828** | **0,16** |
| UDP | 3.867.155 | 5,49 | | | | |
| NetBIOS | 3.657.497 | 5,19 | | **WebDDoS** | **439** | **0,0006** |
| DrDoS_UDP | 3.134.645 | 4,45 | | | | |

Ba lớp cần theo dõi riêng khi đọc kết quả:

- **`WebDDoS` chỉ có 439 mẫu.** Chia 59,5/10,5/30 → khoảng 261 train / 46 val / 132 test.
  Macro-F1 của lớp này sẽ rất nhiễu; đừng diễn giải quá mức một con số dựa trên 132 mẫu.
- **† `UDPLag` = `UDP-lag` (366.461) + `UDPLag` (1.873) = 368.334.** Dữ liệu thô có hai
  nhãn riêng cho cùng một loại tấn công, do hai ngày thu thập (`01-12` và `03-11`) đặt tên
  khác nhau. Đã gộp theo yêu cầu — xem mục "Gộp nhãn" bên dưới.
- **`BENIGN` chỉ chiếm 0,16%.** Binary view (BENIGN vs ATTACK) để so Table 9 vì thế cực kỳ
  mất cân bằng — đúng lý do bài báo ghi nhận FPR 8,18%.

---

## Gộp nhãn `UDP-lag` → `UDPLag`

CIC-DDoS2019 chứa **hai nhãn khác nhau cho cùng một loại tấn công**: ngày `01-12` ghi
`UDP-lag` (366.461 mẫu), ngày `03-11` ghi `UDPLag` (1.873 mẫu). Pipeline gộp chúng trước
khi đánh mã lớp → **18 lớp** thay vì 19.

Khai báo trong [`configs/mddcc.yaml`](configs/mddcc.yaml), không hardcode:

```yaml
data:
  label:
    merge_map:
      "UDP-lag": "UDPLag"     # đặt {} để giữ nguyên 19 lớp như dữ liệu thô
```

Việc gộp diễn ra ở `scan_labels()`, **trước** khi chia tập — nên phân tầng, `sample_manifest`
và mọi metric đều nhất quán trên 18 lớp. `label_mapping.json` ghi lại đầy đủ để truy nguyên:

```json
"label_merge": {
  "applied": true,
  "map": {"UDP-lag": "UDPLag"},
  "merged_into": {"UDPLag": ["UDP-lag"]},
  "raw_counts_before_merge": {"UDP-lag": 366461, "UDPLag": 1873, ...}
}
```

Nếu `merge_map` trỏ tới một nhãn không tồn tại trong dữ liệu, pipeline **fail-fast** thay vì
âm thầm bỏ qua — tránh trường hợp gõ sai tên nhãn mà vẫn chạy như không có gì.

> Đây là **sai khác so với dataset gốc**, đã ghi vào `deviations_from_paper`. Khi báo cáo
> kết quả phải nêu rõ số lớp là 18 và lý do gộp, vì Macro-F1 trên 18 lớp không so trực tiếp
> được với một công bố khác dùng 19 lớp.

---

## Quy tắc loại cột

`feature_selection: none` — **không** chọn tập con theo độ quan trọng như bài báo (48/80).
Chỉ loại những cột không mang thông tin hoặc gây rò rỉ, và mọi cột bị loại đều được ghi
kèm lý do vào `preprocessing.json`:

| Nhóm | Cột | Lý do |
|---|---|---|
| Định danh / rò rỉ | `Flow ID`, `Source IP`, `Destination IP`, `Timestamp`, `Unnamed: 0`, `SimillarHTTP` | giữ lại sẽ làm metric đẹp giả tạo |
| Provenance | `__capture_day`, `__source_file_id`, `__source_row_id` | do bước chuyển Parquet thêm vào, không phải đặc trưng lưu lượng |
| Thiếu > 80% | tính trên **tập train** | §3.C.1 |
| Hằng số tuyệt đối | `min == max` trên train | không mang thông tin |
| Trùng lặp | lọc ứng viên bằng chữ ký thống kê, rồi **đối chiếu giá trị thật** trên mẫu ≤ 50.000 hàng train | §3.B |

Thứ tự bắt buộc: split → fit scaler trên train → **loại cột** → tính hình học wavelet →
dựng cache. Loại cột diễn ra *sau* khi fit scaler vì cả ba tiêu chí đều phải tính chỉ trên
tập train; Min-Max độc lập từng cột nên cắt bớt cột không làm sai thống kê, không cần fit lại.

> Phát hiện trùng lặp được xác minh trên mẫu chứ không trên toàn bộ 70,4M hàng. Hai cột
> giống hệt nhau trên 50.000 hàng train nhưng khác nhau ở phần còn lại sẽ bị bỏ sót —
> đánh đổi có chủ ý để tránh một lượt quét toàn bộ dữ liệu.

---

## Giai đoạn 1 — ngoài phạm vi đánh giá

`src/stage1_switch_stats.py` implement đúng công thức của bài báo:

| | Công thức | Hướng khi bị tấn công |
|---|---|---|
| (1) | `R_Pi = N_FI / N_Pi` | giảm |
| (2) | `R_FI = N_FO / N_FI` | giảm |
| (3) | `ΔN_P = abs(N_PI − N_PO)` | tăng |

Ngưỡng học từ **chỉ lưu lượng bình thường** (bài báo lấy 10.000 bộ mẫu):
`R_Pi` và `R_FI` dùng `μ − 3σ`, `ΔN_P` dùng `μ + 3σ`. Bài báo yêu cầu **cả ba** cùng vượt
ngưỡng mới kết luận có tấn công — phép AND, không phải OR; test đã khoá hành vi này.

**Module không được chạy trên CIC-DDoS2019** và không được báo cáo như đã tái hiện đủ:
dataset là dữ liệu luồng do CIC-FlowMeter trích xuất, không chứa `N_FI`, `N_FO`, `N_Pi`,
`N_PI`, `N_PO`. Hàm `assert_not_applicable_to_cicddos2019()` chặn cứng mọi ý định đó.
Tái hiện đầy đủ cần môi trường Mininet/POX, nằm ngoài phạm vi luận văn.

---

## Loss: `MSE + λ·Σσ(w)` và một chỗ mơ hồ của bài báo

Công thức (8) được cài đúng nguyên văn — `σ(ω) = sqrt( (1/nk)Σω² − ((1/nk)Σω)² )`
tính trên **toàn bộ `nk` phần tử** của ma trận trọng số từng lớp conv/fc, **không tính bias**.
Lưu ý đây là độ lệch chuẩn tổng thể (`ddof=0`), không phải `torch.std` mặc định (`ddof=1`);
test khoá lại điều này.

Nhưng bài báo chỉ viết *"we use the mean squared error as the loss function"* và công thức (9)
là `L = min_ω {f(X,y:ω) + λσ(ω)}` — **không định nghĩa cách rút gọn MSE**. Với 13 lớp
(4 nhánh × 3 conv + 1 FC), `Σσ(w) ≈ 1.064` lúc khởi tạo. Đo thực tế:

| `mse_reduction` | MSE | `λ·Σσ(w)` | Tỷ lệ |
|---|---:|---:|---:|
| `mean_elements` (`nn.MSELoss` mặc định) | 0,0527 | 1,0639 | **20,2×** |
| `mean_batch_sum_class` | 0,9493 | 1,0639 | **1,1×** |

Chọn **`mean_batch_sum_class`** — chế độ duy nhất mà regularizer không át loss chính.
Với `mean_elements`, `σ(w)` chiếm 95% tổng loss và gradient sẽ chủ yếu kéo trọng số về
phía đồng đều thay vì học phân loại. Đây là quyết định kỹ thuật giải quyết chỗ mơ hồ của
bài báo, không phải sai khác — nhưng đã ghi vào `run_config.json` để đối chiếu, và đổi được
qua config nếu muốn chạy biến thể.

`history.json` ghi tách riêng `train_mse_loss` và `train_std_reg` mỗi epoch (mục 2.C) để
kiểm chứng liên tục rằng regularizer không lấn át.

---

## Báo cáo và giải thích

### `make_report.py` chạy độc lập

```bash
python make_report.py --run-dir s3://bucket/prefix/mddcc_20260819-0251 --upload
python make_report.py --run-dir ./_localstore/mddcc_20260819-0251
```

Sinh lại **đủ 14 hình C1–C14** cùng CSV chỉ từ artifact đã lưu, **không cần train lại** —
đã kiểm chứng bằng chạy thật. Lý do tồn tại: Kaggle cắt session bất kỳ lúc nào, và khi cần
sửa nhãn/màu hình cho luận văn thì không được phép huấn luyện lại 39 session.

Mỗi hình xuất **đồng thời 3 file**: `.png` (300 dpi), `.pdf` (vector), `.csv` (dữ liệu đúng
như đã vẽ). Hình nào thiếu dữ liệu đầu vào thì bị bỏ qua kèm cảnh báo, không làm hỏng cả
báo cáo.

### Bước đánh giá cuối là một bước RIÊNG BIỆT

```bash
python -m src.evaluate --config configs/mddcc.yaml
```

Chỉ chạy được khi `training_state.json` báo `is_complete` — chưa đủ 100 epoch thì fail-fast.
Nếu bước này lỗi, checkpoint vẫn nguyên trên S3 và chạy lại được bằng chính lệnh đó.
Idempotent: seed cố định, giữ nguyên thứ tự test, không lấy mẫu ngẫu nhiên cho metric chính.

### Chi phí wavelet được bóc tách (mục 6)

Đây là điểm khác biệt cốt lõi của MDDCC nên `t_swt` không được giấu trong tổng thời gian.
Đo trên CPU 4 luồng, `model.eval()` + `no_grad`, F=81 → S=10:

| batch | `t_scale` | `t_swt` | `t_forward` | `t_total` p50 / p95 | throughput | SWT chiếm |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 3,59 ms | 29,65 ms | 395,91 ms | 427,64 / 438,64 ms | 9.578 mẫu/s | **6,93%** |
| 1 | 0,04 ms | 0,17 ms | 1,99 ms | 2,20 / 3,13 ms | 454 mẫu/s | **7,55%** |

Hai điều đáng chú ý: phân rã wavelet chỉ chiếm ~7% thời gian suy luận, phần còn lại là
forward pass của 4 nhánh CNN. Và độ trễ ở `batch=1` là **2,2 ms**, thoả yêu cầu < 100 ms mà
bài báo nhấn mạnh cho phát hiện thời gian thực.

### Đo mức độ quan trọng của đặc trưng

- **Permutation** là phương pháp chính: hoán vị từng cột **trên không gian feature gốc
  (trước SWT)** rồi tính lại SWT. Hoán vị trực tiếp trên subband đã biến đổi sẽ vô nghĩa vì
  một cột gốc ảnh hưởng tới cả 4 subband qua bộ lọc wavelet. Test kiểm tra dữ liệu được trả
  lại nguyên trạng sau mỗi cột.
- **SHAP** `GradientExplainer` (đã xác minh với `shap==0.52.0`), quy đóng góp từ 4 subband
  về feature gốc bằng cách cộng `|SHAP|` theo vị trí rồi **bỏ các vị trí padding**. Tính
  theo chunk, cộng dồn, không giữ tensor `[sample, class, feature]` trong RAM. Số mẫu thực
  tế đã dùng được ghi vào `shap_meta.json` — nếu yêu cầu nhiều hơn số mẫu có sẵn thì tự
  giảm và báo đúng con số. Thiếu thư viện `shap` thì bỏ qua C13 kèm cảnh báo, không làm
  hỏng cả bước đánh giá.
- **Branch ablation** zero-out lần lượt `cD1/cD2/cD3/cA3`, đo mức giảm Macro-F1 — bằng
  chứng định lượng cho đóng góp của phân rã wavelet đa mức.
- `feature_importance_comparison.csv` giữ **riêng** `rank_permutation` và `rank_shap` kèm cờ
  `top10_consensus`, **không** gộp hai thước đo thành một điểm.

> SHAP và permutation thể hiện **đóng góp dự đoán**, không chứng minh quan hệ nhân quả.
> Các đặc trưng tương quan mạnh có thể chia sẻ importance, làm cả hai cùng thấp một cách
> giả tạo. Ghi chú này được lưu vào `explain_sample_manifest.json` của mỗi run.

---

## Chống mất session

| Cơ chế | Cài đặt |
|---|---|
| Checkpoint theo epoch **và** theo step | mỗi `checkpoint.interval_steps` (mặc định 200) |
| Upload an toàn | ghi key `_tmp/` → kiểm tra size + sha256 → copy → xoá tmp. Session bị cắt giữa lúc upload thì key chính vẫn nguyên vẹn |
| Resume giữa epoch | bỏ qua `steps_done_in_epoch` batch đầu của permutation đã seed theo `(seed, epoch)` |
| Kiểm tra hash | `params_hash`, `feature_schema_hash`, `scaler_hash` — lệch thì **fail-fast**, không âm thầm train lại từ đầu |
| `run_id` cố định | `current_run_id.json` trên S3; session sau đọc lại đúng run |
| Thoát chủ động | còn 20 phút → lưu checkpoint + history + state, `exit_reason="time_guard"`, exit 0 |
| `history.json` | append-only, kiểm tra epoch liên tục 1..n mỗi lần resume |

Epoch bị resume giữa chừng được đánh dấu `train_metrics_partial: true` kèm
`resumed_after_batches` — vì metric train của epoch đó chỉ tính trên phần batch chạy trong
session sau, không phải cả epoch. Không đánh dấu thì learning curve sẽ bị đọc nhầm.

`LocalStore` cho phép chạy thử toàn bộ pipeline không cần AWS (`--local-store <dir>`).
Run thật **bắt buộc** dùng S3 — pipeline in cảnh báo khi không thấy `S3_BUCKET`.

---

## Chạy

```bash
# Cài phụ thuộc (Kaggle đã có sẵn torch/numpy/sklearn/pyarrow)
pip install -r requirements.txt

# Bước 2a — discovery (chạy trên Kaggle, có mount dataset)
python scripts/discover_dataset.py --config configs/mddcc.yaml --count-labels

# Bước 2b — dựng cache + toàn bộ artifact cấu hình
python -m src.data --config configs/mddcc.yaml --out-dir artifacts

# Bước 3 — huấn luyện (S3 thật)
python -m src.train --config configs/mddcc.yaml

# Chạy thử không cần AWS
python -m src.train --config configs/mddcc.yaml     --input-dir <parquet> --cache-dir <tmp> --local-store <dir> --max-epochs 2

# Bước 4 — đánh giá cuối (chạy SAU khi xong 100 epoch)
python -m src.evaluate --config configs/mddcc.yaml

# Sinh lại toàn bộ hình + CSV từ artifact, không train lại
python make_report.py --run-dir s3://$S3_BUCKET/$S3_PREFIX/<run_id> --upload

# Test
python -m pytest tests -q
```

```bash
# Bước 5 — sinh lại notebook sau khi sửa build_notebook.py
python scripts/build_notebook.py

# Push kernel bằng tay (thường để GitHub Actions làm)
kaggle kernels push -p kernel/
```

> `--max-epochs` **chỉ dùng để chạy thử**. Run chính luôn lấy `train.epochs = 100`
> từ config và dừng chính xác ở epoch 100.

> Trên Windows, nếu `C:\Users\<user>\AppData\Local\Temp` bị chặn quyền, chạy
> `python -m pytest tests -q --basetemp=<thư-mục-ghi-được>`.

---

## Tự động hoá: GitHub Actions → Kaggle

### Luồng quyết định

`scripts/kaggle_orchestrator.py` chứa toàn bộ logic; workflow chỉ gọi nó. Hàm `decide()` là
**hàm thuần** không đọc/ghi gì, nên test được từng nhánh:

| Trạng thái | Quyết định |
|---|---|
| `current_epoch ≥ 100` hoặc `status=completed` | `DONE` — không push nữa |
| kernel `running` / `queued` | `WAIT` — không đụng vào |
| kernel `complete` / `error` / `cancelAcknowledged`, epoch < 100 | `PUSH` — mở session mới |
| chưa có `training_state.json` | `PUSH` — khởi động run đầu tiên |
| **3 lần push liên tiếp mà `current_epoch` đứng yên** | `ABORT` + mở GitHub Issue |
| `restarts ≥ max_restarts` (60) | `ABORT` |

`max_restarts = 60` vì ngân sách đo được là **39 session**, cộng biên an toàn.

Đã dry-run thật ba tình huống với `LocalStore`: epoch 37/100 + kernel `complete` → `PUSH`;
kernel `running` → `WAIT`; ba lần push liên tiếp đứng yên ở epoch 37 → `ABORT`.

```bash
# Thử logic quyết định không cần AWS
python scripts/kaggle_orchestrator.py --kernel richardnguyen1991/mddcc     --local-store ./_localstore --kernel-status complete --dry-run
```

### `kernel-metadata.json`

```json
"id": "richardnguyen1991/mddcc",
"dataset_sources": ["dungnguyen28101991/cicddos2019-parquet"],
"enable_internet": true, "enable_gpu": false, "accelerator": "none", "is_private": true
```

**Dòng `dataset_sources` là lỗi hay gặp nhất khi tự động hoá bằng `kaggle kernels push`.**
Thiếu nó thì session do GitHub Actions khởi động sẽ **không có dataset**, notebook chết ngay
ở bước đọc dữ liệu và vòng lặp restart quay vô ích cho tới khi chạm `max_restarts`. Ô code
đầu tiên của notebook vì thế kiểm tra `/kaggle/input` và fail-fast kèm thông báo chỉ đúng
nguyên nhân này.

### Notebook

`kernel/kaggle_notebook.ipynb` được **sinh ra từ `scripts/build_notebook.py`**, không sửa
JSON bằng tay — JSON notebook rất dễ hỏng và không review được trong diff. Có test kiểm tra
file `.ipynb` không lệch khỏi script sinh nó.

Tám ô code: fail-fast dataset → clone repo → cài phụ thuộc → đọc secret → in trạng thái
trước → train → đánh giá cuối (chỉ khi đủ 100 epoch) → in trạng thái sau.

Notebook **idempotent**: chỉ gọi `RunRegistry.get()`, không bao giờ tự tạo `run_id` mới —
có test khẳng định `new_run_id` không xuất hiện trong notebook.

Nếu `data.kaggle_input_dir` trong config không tồn tại (Kaggle đổi cấu trúc mount), notebook
tự dò lại đường dẫn thật, ghi ra `configs/mddcc.runtime.yaml` và dùng file đó — thay vì chết.

### Xử lý `KAGGLE_API_TOKEN` hai dạng

Secret này có thể là **chuỗi key thuần** hoặc **toàn bộ nội dung `kaggle.json`**. Workflow tự
nhận dạng: thử `json.loads` trước, nếu ra dict có khoá `key` thì dùng luôn; nếu không thì
ghép với `KAGGLE_USERNAME`. Ghi ra `~/.kaggle/kaggle.json` với `chmod 600`.

### Hai lưu ý về cron

GitHub **tự tắt cron sau 60 ngày** repo không hoạt động, và cron chỉ chạy "best effort" nên
có thể trễ vài phút. Với ngân sách ~39 session × 11h20m thì cả hai đều không gây vấn đề —
workflow chỉ cần bắt được thời điểm kernel vừa kết thúc, và mỗi lần push đều tạo commit
activity gián tiếp qua Issue/Actions log.

---

## Phân vai lưu trữ — không được nhầm lẫn

- **Kaggle Dataset** = dữ liệu đầu vào, **CHỈ ĐỌC**. Không ghi gì vào `/kaggle/input`.
- **S3** = trạng thái huấn luyện và mọi kết quả. Nơi duy nhất sống sót qua session bị cancel,
  và là nguồn sự thật cho GitHub Actions.
- **GitHub Secrets** = nơi duy nhất giữ credential. **Kaggle không lưu secret nào** —
  credential tạm thời được tiêm vào notebook lúc push, xem mục Secrets.
- Cache cục bộ nằm ở `/kaggle/temp` (không tính vào giới hạn 20 GB của `/kaggle/working`),
  **mất khi session kết thúc** và phải dựng lại mỗi session — chi phí này được đo và ghi
  vào `cache_build_seconds`.

## Secrets — Kaggle không lưu secret nào

Toàn bộ credential nằm trên **GitHub Secrets**. Kaggle không lưu gì.

```
AWS_ACCESS_KEY_ID      AWS_SECRET_ACCESS_KEY   AWS_DEFAULT_REGION
S3_BUCKET              S3_PREFIX
KAGGLE_KERNEL = richardnguyen1991/mddcc
KAGGLE_API_TOKEN       (+ KAGGLE_USERNAME nếu token là key thuần)
```

### Credential đến được Kaggle bằng cách nào

Vấn đề: notebook chạy trên Kaggle cần credential AWS để ghi S3, nhưng GitHub
Secrets không tự có mặt trong runtime Kaggle. Cách giải quyết:

```
GitHub Secrets (khoá dài hạn)
   └─ sts:GetSessionToken (16 giờ)  ← chạy trên runner GitHub
        └─ base64 → tiêm vào bản notebook trong thư mục TẠM
             └─ kaggle kernels push từ thư mục tạm
                  └─ notebook giải mã → biến môi trường → ghi S3
        └─ xoá thư mục tạm (bước `if: always()`)
```

`scripts/prepare_kernel_push.py` làm việc này. Thư mục `kernel/` đã commit
**không bao giờ bị ghi vào** — bản trong git luôn chỉ có placeholder
`__MDDCC_CREDENTIALS_B64__`, và có test khẳng định điều đó cùng với việc file
không chứa `AKIA`/`ASIA`/`SecretAccessKey`.

Token sống 16 giờ (`STS_DURATION_SECONDS`), đủ cho session 11h20m cộng thời gian
Kaggle xếp hàng. Notebook kiểm tra hạn ngay ở mục 4: còn dưới 1 giờ thì cảnh báo,
đã hết hạn thì fail-fast thay vì chạy 11 giờ rồi mất hết checkpoint.

### Đánh giá bảo mật trung thực

**Được:** khoá dài hạn không bao giờ rời khỏi GitHub. Không có secret nào lưu trên
Kaggle. Token tự hết hạn.

**Vẫn còn rủi ro:** token tạm thời **có** nằm trong mã nguồn notebook mà Kaggle lưu
lại. Điều này không tránh được — runtime Kaggle phải có một credential nào đó, và
presigned URL cũng chỉ là một dạng bearer token khác. Kaggle giữ lại các phiên bản
kernel cũ, mỗi phiên bản chứa token của lần đó; sau khi hết hạn thì vô hại.

**Vì vậy bắt buộc:** dùng một **IAM user riêng** (không phải root — script từ chối
khoá root vì chỉ được 3600s và cấp quyền toàn bộ tài khoản), giới hạn policy trong
`arn:aws:s3:::$S3_BUCKET/$S3_PREFIX/*` với đúng bốn quyền `s3:GetObject`,
`s3:PutObject`, `s3:DeleteObject`, `s3:ListBucket`. Khi đó thiệt hại tối đa nếu
token bị lộ trong 16 giờ là giới hạn trong prefix của run này.

Nếu cần chặt hơn nữa: đổi sang `sts:AssumeRole` với inline session policy để thu hẹp
quyền ngay tại thời điểm cấp token.

### Chạy tay khi cần

```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=...
export S3_BUCKET=... S3_PREFIX=...
python scripts/prepare_kernel_push.py --out-dir /tmp/kernel_push
kaggle kernels push -p /tmp/kernel_push
rm -rf /tmp/kernel_push
```

Notebook **không** khởi động được bằng tay trên Kaggle UI nếu chưa tiêm credential —
nó sẽ fail-fast ở mục 4b kèm hướng dẫn.

## Sai khác so với bài báo

Bảng đầy đủ nằm ở mục `deviations_from_paper` trong
[`configs/mddcc.yaml`](configs/mddcc.yaml) và sẽ được ghi vào `run_config.json` mỗi run.
Các điểm chính: dùng tất cả feature (bài báo chọn 48), chạy đủ 100 epoch không early
stopping, chỉ dùng CIC-DDoS2019, chạy CPU, `pool_ceil_mode=true`, SWT tính on-the-fly
thay vì cache subband, và gộp `UDP-lag` → `UDPLag`.

**Learning rate đã đổi về 0.01 đúng bài báo (2026-08-19)** — trước đó dùng 0.001 theo yêu
cầu thí nghiệm. Đây không còn là sai khác.

**Giai đoạn 1 của bài báo (thống kê cổng switch SDN với ngưỡng 3-sigma) và module giảm
thiểu dựa trên đồ thị KHÔNG được đánh giá** vì CIC-DDoS2019 không chứa số liệu cổng switch
và không có môi trường Mininet/POX. Module `src/stage1_switch_stats.py` vẫn sẽ được
implement đúng công thức (1)(2)(3) kèm unit test trên dữ liệu tổng hợp, nhưng không được
báo cáo như đã tái hiện đủ.

## Rủi ro đang theo dõi

### Chẩn đoán: vì sao đổi learning rate về 0.01

Trên cùng dữ liệu tổng hợp, 15 epoch, tách riêng ảnh hưởng của `λ` và `lr`:

| Biến thể | train MSE | train F1 | **val Macro-F1** | val Acc |
|---|---:|---:|---:|---:|
| λ=1.0, lr=0.001 | 0,6648 | 0,2185 | **0,1432** | 0,4012 |
| λ=0.0, lr=0.001 | 0,6647 | 0,2190 | **0,1432** | 0,4012 |
| λ=1.0, lr=0.01 | 0,6515 | 0,2650 | **0,2570** | 0,5083 |
| λ=0.0, lr=0.01 | 0,6436 | 0,2798 | **0,3803** | 0,6865 |
| λ=0.1, lr=0.01 | 0,6445 | 0,2780 | **0,3745** | 0,6770 |

Hai điều đọc được:

1. **Ở `lr=0.001`, bật hay tắt `σ(w)` cho kết quả giống hệt nhau** (0,1432 / 0,4012, MSE
   lệch 0,0001). Regularizer hoàn toàn vô can — nút thắt là learning rate. Nâng lên 0,01
   thì Macro-F1 nhảy 0,1432 → 0,2570.
2. **Nhưng `λ=1.0` vẫn đắt**: ở `lr=0.01` nó làm mất một phần ba Macro-F1
   (0,2570 so với 0,3803). `λ=0.1` lấy lại gần hết (0,3745) mà vẫn giữ cơ chế `σ(w)`.

Điểm (1) là lý do đổi `lr` về 0,01. Điểm (2) **chưa xử lý** — `λ=1.0` là giá trị mặc định
theo công thức (9) và giữ nguyên cho run chính. Nếu sau 100 epoch Macro-F1 thấp, `λ=0.1`
là ứng viên đầu tiên cho một `run_id` biến thể riêng.

> Cảnh báo về quy mô: chẩn đoán này chỉ có 56 step/epoch, run thật có **10.231 step/epoch**.
> Với SGD không momentum, `lr` nhỏ được bù bằng số bước — nên bảng trên **không** chứng minh
> `lr=0.001` sai ở quy mô thật, chỉ chứng minh nó chậm hơn nhiều ở cùng số bước.

### Smoke test 25 epoch đã cho thấy dấu hiệu suy biến

Chạy thử trên dữ liệu tổng hợp (24.000 hàng, 81 cột, 4 lớp tỷ lệ .1/.4/.4/.1, trong đó
20 cột mang tín hiệu thật nên **bài toán chắc chắn học được**), cấu hình đúng như run chính:

| epoch | train MSE | train Macro-F1 | **val Macro-F1** | val Acc | grad_norm | Σσ(w) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0,7879 | 0,1592 | 0,1415 | 0,3948 | 0,828 | 1,0638 |
| 10 | 0,6689 | 0,2231 | 0,1432 | 0,4012 | 0,325 | 1,0555 |
| 18 | 0,6632 | 0,2205 | 0,1432 | 0,4012 | 0,281 | 1,0485 |
| 25 | 0,6623 | 0,2217 | **0,1432** | 0,4012 | 0,290 | 1,0423 |

`val_macro_f1` đứng yên hoàn toàn. Con số 0,1432 không ngẫu nhiên: nếu mô hình luôn dự đoán
một lớp chiếm 0,4 thì `acc = 0,40` và `macro-F1 = 2×0,4/1,4 ÷ 4 = 0,143` — khớp chính xác.
**Mô hình suy biến về lớp đa số** và MSE bão hoà từ khoảng epoch 14.

Đây đúng là rủi ro mục 11.C đã cảnh báo: `MSE + Softmax + SGD lr=0.001` hội tụ chậm hơn
nhiều so với `CrossEntropy + Adam`. Cần lưu ý smoke test chỉ có 56 step/epoch, trong khi
run thật có **10.231 step/epoch** — nhiều hơn ~180 lần số lần cập nhật gradient, nên chưa
thể kết luận chắc chắn cho run thật.

**Không tự ý đổi loss/optimizer/lr giữa chừng.** Nếu sau 100 epoch Macro-F1 vẫn thấp,
kết quả sẽ được báo cáo đúng như vậy kèm panel `grad_norm` / `σ(w)` (hình C1 panel d) làm
bằng chứng chẩn đoán, rồi mới đề xuất một `run_id` RIÊNG cho biến thể — không thay run chính.

Dùng tất cả feature + không xử lý mất cân bằng → FPR có thể cao như bài báo đã ghi nhận
(8,18% trên CIC-DDoS2019). Đây là kết quả cần báo cáo, không phải lỗi cần "sửa".
