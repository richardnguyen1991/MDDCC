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
| — | **Tổng test** | **118/118 pass** |
| 3 | `model.py`, `train.py`, `checkpoint.py`, `s3io.py` (resume giữa epoch từ S3) | ✅ xong |
| 4 | `evaluate.py`, `viz.py`, `make_report.py`, `explain.py` | ⏳ chưa làm |
| 5 | `kernel/`, `.github/workflows/run-kaggle.yml` | ⏳ chưa làm |
| 6 | README đầy đủ, chạy thử 2 epoch + kiểm tra resume, rồi chạy đủ 100 epoch | ⏳ chưa làm |

---

## Cấu hình đã chốt

Toàn bộ siêu tham số nằm trong [`configs/mddcc.yaml`](configs/mddcc.yaml). Các giá trị cốt lõi:

| Tham số | Giá trị | Nguồn |
|---|---|---|
| epochs | **100 chính xác**, không early stopping | yêu cầu thí nghiệm |
| batch_size | 4096 | yêu cầu thí nghiệm |
| learning_rate | 0.001 hằng số, không scheduler | yêu cầu thí nghiệm (bài báo: 0.01) |
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
# Bước 2a — discovery (chạy trên Kaggle, có mount dataset)
python scripts/discover_dataset.py --config configs/mddcc.yaml --count-labels

# Bước 2b — dựng cache + toàn bộ artifact cấu hình
python -m src.data --config configs/mddcc.yaml --out-dir artifacts

# Bước 3 — huấn luyện (S3 thật)
python -m src.train --config configs/mddcc.yaml

# Chạy thử không cần AWS
python -m src.train --config configs/mddcc.yaml     --input-dir <parquet> --cache-dir <tmp> --local-store <dir> --max-epochs 2

# Test
python -m pytest tests -q
```

> `--max-epochs` **chỉ dùng để chạy thử**. Run chính luôn lấy `train.epochs = 100`
> từ config và dừng chính xác ở epoch 100.

> Trên Windows, nếu `C:\Users\<user>\AppData\Local\Temp` bị chặn quyền, chạy
> `python -m pytest tests -q --basetemp=<thư-mục-ghi-được>`.

---

## Phân vai lưu trữ — không được nhầm lẫn

- **Kaggle Dataset** = dữ liệu đầu vào, **CHỈ ĐỌC**. Không ghi gì vào `/kaggle/input`.
- **S3** = trạng thái huấn luyện và mọi kết quả. Nơi duy nhất sống sót qua session bị cancel,
  và là nguồn sự thật cho GitHub Actions.
- Cache cục bộ nằm ở `/kaggle/temp` (không tính vào giới hạn 20 GB của `/kaggle/working`),
  **mất khi session kết thúc** và phải dựng lại mỗi session — chi phí này được đo và ghi
  vào `cache_build_seconds`.

## Secrets

GitHub Actions dùng: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`,
`S3_BUCKET`, `S3_PREFIX`, `KAGGLE_USERNAME`, `KAGGLE_API_TOKEN`, `KAGGLE_KERNEL`.

**GitHub Secrets không tự có mặt trong runtime của Kaggle.** Phải tự thêm thủ công
trên Kaggle (Add-ons → Secrets) đúng 5 secret: `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `S3_BUCKET`, `S3_PREFIX`.
Tuyệt đối không nhét credential vào notebook, `kernel-metadata.json` hay git.

Quyền IAM tối thiểu: `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, `s3:ListBucket`,
giới hạn trong `arn:aws:s3:::$S3_BUCKET/$S3_PREFIX/*`.

---

## Sai khác so với bài báo

Bảng đầy đủ nằm ở mục `deviations_from_paper` trong
[`configs/mddcc.yaml`](configs/mddcc.yaml) và sẽ được ghi vào `run_config.json` mỗi run.
Các điểm chính: learning rate 0.001 (bài báo 0.01), dùng tất cả feature (bài báo chọn 48),
chạy đủ 100 epoch không early stopping, chỉ dùng CIC-DDoS2019, chạy CPU,
`pool_ceil_mode=true`, và SWT tính on-the-fly thay vì cache subband.

**Giai đoạn 1 của bài báo (thống kê cổng switch SDN với ngưỡng 3-sigma) và module giảm
thiểu dựa trên đồ thị KHÔNG được đánh giá** vì CIC-DDoS2019 không chứa số liệu cổng switch
và không có môi trường Mininet/POX. Module `src/stage1_switch_stats.py` vẫn sẽ được
implement đúng công thức (1)(2)(3) kèm unit test trên dữ liệu tổng hợp, nhưng không được
báo cáo như đã tái hiện đủ.

## Rủi ro đang theo dõi

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
