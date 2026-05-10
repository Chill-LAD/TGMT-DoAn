# ĐỀ XUẤT ĐỒ ÁN CUỐI KỲ

## PHÁT HIỆN WATERMARK (WATERMARK DETECTION) TRONG ẢNH SỐ

**Loại đồ án:** Cuối kỳ

---

## I. THÔNG TIN THÀNH VIÊN NHÓM

| STT | Họ và tên            | MSSV     | Email                         | Đóng góp chính              |
| --- | -------------------- | -------- | ----------------------------- | --------------------------- |
| 1   | Đặng Nguyễn Thái Đạt | 23120227 | 23120227@student.hcmus.edu.vn | Dataset, baseline model     |
| 2   | Lê Anh Đức           | 23120236 | 23120236@student.hcmus.edu.vn | Pipeline, cải tiến, báo cáo |

---

## II. GIỚI THIỆU VÀ LÝ DO CHỌN ĐỀ TÀI

### 2.1. Bối cảnh

Trong kỷ nguyên số hóa, việc bảo vệ bản quyền và xác thực nguồn gốc nội dung ảnh trở nên cấp thiết. Watermark là kỹ thuật nhúng thông tin ẩn vào ảnh số nhằm:

- Bảo vệ bản quyền tác phẩm
- Xác thực nội dung và nguồn gốc
- Theo dõi phân phối nội dung trên internet
- Hỗ trợ các vấn đề pháp lý về gian lận hình ảnh

Watermark chia thành hai loại chính:

- **Visible watermark:** Logo, chữ in trên ảnh, nhìn thấy bằng mắt thường
- **Invisible watermark:** Thay đổi pixel ở mức vi mô, không nhìn thấy nhưng giải mã được

### 2.2. Vấn đề nghiên cứu

Các nghiên cứu về watermark embedding đã phát triển mạnh, nhưng bài toán **watermark detection** vẫn còn nhiều thách thức:

1. **Visible watermark:** Tương đối dễ phát hiện nhưng cần xác định vị trí chính xác
2. **Invisible watermark:** Phụ thuộc thuật toán nhúng, khó phát hiện với watermark "unknown"
3. **Watermark yếu:** Bị nén, resize, blur làm giảm tín hiệu đáng kể
4. **Tổng hợp:** Cần mô hình phát hiện cả hai loại với độ chính xác cao

### 2.3. Ý nghĩa thực tiễn

- Hỗ trợ kiểm tra bản quyền hình ảnh tự động
- Phát hiện ảnh bị đánh cắp hoặc sử dụng trái phép
- Xác định nguồn gốc ảnh trong điều tra số
- Ứng dụng trong an ninh mạng và truyền thông

---

## III. MỤC TIÊU ĐỒ ÁN

### 3.1. Mục tiêu tổng quát

Xây dựng mô hình học sâu phát hiện sự tồn tại của watermark (visible và invisible) trong ảnh số:

- Phân loại nhị phân: ảnh có watermark / không có watermark
- Định vị vùng chứa watermark (localization)

### 3.2. Mục tiêu cụ thể

| STT | Mục tiêu                             | Yêu cầu                        | Độ ưu tiên |
| --- | ------------------------------------ | ------------------------------ | ---------- |
| 1   | Khảo sát và tổng hợp các phương pháp | So sánh ≥5 phương pháp         | Bắt buộc   |
| 2   | Xây dựng mô hình baseline            | CNN-based, F1 ≥ 0.85           | Bắt buộc   |
| 3   | Cải tiến mô hình                     | Frequency+Attention, F1 ≥ 0.90 | Nâng cao   |
| 4   | Thực nghiệm ≥2 datasets              | Đa điều kiện test              | Bắt buộc   |
| 5   | Build chương trình hoàn chỉnh        | Giao diện, <100ms/ảnh          | Bắt buộc   |

---

## IV. CÁC NGHIÊN CỨU LIÊN QUAN (RELATED WORK)

### 4.1. Phân loại các hướng tiếp cận

#### A. Phương pháp truyền thống (Traditional Methods)

| Phương pháp     | Nguyên lý                      | Ưu điểm            | Nhược điểm                  |
| --------------- | ------------------------------ | ------------------ | --------------------------- |
| **DCT-based**   | Phân tích hệ số cosine rời rạc | Hiệu quả với JPEG  | Phụ thuộc thuật toán        |
| **DWT-based**   | Phân tích wavelet rời rạc      | Đa độ phân giải    | Khó phát hiện watermark yếu |
| **SVD-based**   | Phân tích giá trị riêng        | Tolerant với nhiễu | Tốn tính toán               |
| **Statistical** | Thống kê histogram, Fourier    | Đơn giản           | Accuracy thấp               |

Công thức DCT 2D:
$F(u,v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x,y) \cos((2x+1)u\pi/2M) \cos((2y+1)v\pi/2N)$

Công thức DWT:
$W_\psi(a,b) = (1/\sqrt{a}) \int_{-\infty}^{\infty} f(t) \psi^*((t-b)/a) dt$

#### B. Phương pháp học sâu (Deep Learning Methods)

| Mô hình                 | Đặc điểm               | Tham chiếu            |
| ----------------------- | ---------------------- | --------------------- |
| **CNN classifiers**     | ResNet, VGG, MobileNet | Binary classification |
| **YOLO-based**          | Object detection       | Localization          |
| **Transformer/ViT**     | Vision Transformer     | Image understanding   |
| **Frequency-aware CNN** | Kết hợp FFT/DCT        | FSNet                 |
| **Self-supervised**     | SSL approaches         | SSL watermarking      |

#### C. Nghiên cứu gần đây (2024-2026)

1. **FSNet (Frequency Shield Network)** - arXiv 2603.06723 (2026): ASPM module phát hiện invisible watermark trong miền tần số, zero-shot detection vượt trội.

2. **InvisMark** - arXiv 2411.07795, WACV 2025: Encoder-decoder cho AI-generated images.

3. **WAM (Watermark Anything Model)** - arXiv 2411.07231, ICLR 2025: Phát hiện và định vị watermark trong vùng cục bộ.

4. **Dual Watermarking** - arXiv 2502.18501, IEEE TAI 2025: Cryptographic + perceptual hash.

5. **DFCL** - Neural Networks 184 (2025), DOI 10.1016/j.neunet.2024.107077: Contrastive Learning cho visible watermark removal.

### 4.2. So sánh các phương pháp

| Tiêu chí             | DCT/DWT    | CNN cơ bản | Frequency-aware | Transformer |
| -------------------- | ---------- | ---------- | --------------- | ----------- |
| Speed                | Cao        | Trung bình | Trung bình      | Trung bình  |
| Accuracy (visible)   | Thấp       | Cao        | Cao             | Cao         |
| Accuracy (invisible) | Trung bình | Thấp       | Cao             | Trung bình  |
| Generalization       | Kém        | Tốt        | Rất tốt         | Tốt         |

### 4.3. Lựa chọn phương pháp

Nhóm chọn hướng **Hybrid CNN + Frequency Domain** vì:

- Kế thừa ưu điểm truyền thống và học sâu
- Cải thiện generalization cho invisible watermark
- Phù hợp khung thời gian 5-6 tuần
- Đáp ứng yêu cầu cải tiến nâng cao

---

## V. PHƯƠNG PHÁP ĐỀ XUẤT

### 5.1. Kiến trúc tổng quát

```
Input Image
    │
    ▼
┌─────────────────────┐
│  Preprocessing      │
│  Branch 1: RGB      │ ── Resize 224×224, Normalize
│  Branch 2: FFT      │ ── Fourier spectrum
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Feature Extraction │
│  Backbone: ResNet18 │ ── RGB features
│  FreqBranch: 1D CNN │ ── Frequency features
│  Fusion: Concat     │ ── Multi-scale fusion
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Classification     │
│  FC Layers          │ ── Binary: Watermark / No-Watermark
│  Channel Attention  │ ── SE-Net style
└─────────────────────┘
    │
    ▼
Output: {Label, Confidence}
```

### 5.2. Chi tiết từng giai đoạn

#### Giai đoạn 1: Tiền xử lý (Preprocessing)

- **Resize:** 224×224
- **Normalization:** ImageNet mean/std
- **Augmentation:** Random flip, rotation (±15°), color jitter, JPEG simulation
- **Frequency branch:** FFT → Log-magnitude → Central crop 224×224
  
  $F_{freq} = \log(|\text{FFT}(I)| + 1)$

#### Giai đoạn 2: Trích xuất đặc trưng

| Component         | Chi tiết                       | Tham số |
| ----------------- | ------------------------------ | ------- |
| RGB Backbone      | ResNet18 (ImageNet pretrained) | 11.7M   |
| Frequency Branch  | 1D CNN (3 layers) + MaxPool    | ~50K    |
| Fusion            | Concat → FC(512→256)           | -       |
| Channel Attention | SE-Net                         | -       |

SE-Net attention:
$s = \sigma(W_{U2}(ReLU(W_{U1}(GAP(z))))$
$z_{att} = s \cdot z$

Trong đó GAP là Global Average Pooling, sigma là sigmoid activation.

#### Giai đoạn 3: Phân loại

- **Binary Classification:** FC(256→128) + ReLU + Dropout(0.5) → FC(128→2) + Softmax

- **Loss:** Cross-entropy
  
  $\mathcal{L}_{CE} = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$

- **Optimizer:** AdamW, lr=1e-4, weight_decay=1e-2

- **Scheduler:** Cosine annealing
  $\eta_t = \eta_{min} + 0.5(\eta_{max} - \eta_{min})(1 + \cos(t\pi/T_{max}))$

Trong đó $\eta_t$ là learning rate tại step t, $T_{max}$ là tổng số epochs.

#### Giai đoạn 4: Cải tiến

| Cải tiến                | Mô tả              | Tác động |
| ----------------------- | ------------------ | -------- |
| **Multi-scale input**   | 224 + 448 ensemble | +2% F1   |
| **TTA**                 | Flip + rotate      | +1% F1   |
| **Semi-supervised**     | Pseudo-labeling    | +3% F1   |
| **Frequency attention** | ASPM-style gating  | +4% F1   |

ASPM (Adaptive Spectral Perception Module) trong FSNet:
$F_{stem} = \Phi_{ASPM}(X) = G_\theta(X) \cdot X$

$G_\theta(X) = \sigma(W_g(\text{AvgPool}(X)))$

### 5.3. Baseline models

| Mô hình        | Mô tả                            |
| -------------- | -------------------------------- |
| **Baseline 1** | ResNet18 (RGB only)              |
| **Baseline 2** | MobileNetV3 (RGB only)           |
| **Ours v1**    | ResNet18 + Frequency branch      |
| **Ours v2**    | ResNet18 + Frequency + Attention |

---

## VI. KẾ HOẠCH THỰC NGHIỆM

### 6.1. Datasets

| Dataset              | Nguồn                                         | Số lượng | Loại WM             |
| -------------------- | --------------------------------------------- | -------- | ------------------- |
| **PITA**             | HuggingFace (bastienp/visible-watermark-pita) | ~20,000  | Visible (logo/text) |
| **CLWD**             | arXiv 2012.07616                              | ~70,000  | Visible (removal)   |
| **COCO + synthetic** | COCO 2017                                     | ~5,000   | Visible/Invisible   |
| **UniFreq-100K**     | arXiv 2603.06723                              | ~100,000 | Invisible           |

### 6.2. Metrics đánh giá

| Metric             | Công thức     | Ý nghĩa              |
| ------------------ | ------------- | -------------------- |
| **Accuracy**       | (TP+TN)/Total | Độ chính xác tổng    |
| **Precision**      | TP/(TP+FP)    | Độ chính xác dự đoán |
| **Recall**         | TP/(TP+FN)    | Độ nhạy              |
| **F1-score**       | 2*P*R/(P+R)   | Cân bằng P/R         |
| **Inference time** | ms/ảnh        | Tốc độ               |

### 6.3. Điều kiện test

| Điều kiện        | Mô tả              |
| ---------------- | ------------------ |
| Clean            | Không distortion   |
| JPEG compression | Quality 70, 80, 90 |
| Resize           | 0.5x, 0.75x, 1.5x  |
| Noise            | Gaussian (σ=5, 10) |
| Mixed            | Kết hợp distortion |

### 6.4. Môi trường

| Thành phần | Phiên bản       |
| ---------- | --------------- |
| Python     | 3.8+            |
| PyTorch    | 2.0+            |
| OpenCV     | 4.8+            |
| CUDA       | 11.8+           |
| GPU        | RTX 3060+ (8GB) |

---

## VII. KẾ HOẠCH THỰC HIỆN (5-6 TUẦN)

| Tuần | Công việc                           | Thành viên           | Output                   |
| ---- | ----------------------------------- | -------------------- | ------------------------ |
| 1-2  | Khảo sát tài liệu, thu thập dataset | Đặng Nguyễn Thái Đạt | Related Work (5-8 trang) |
| 3    | Xây dựng pipeline, tiền xử lý       | Lê Anh Đức           | Data pipeline            |
| 4    | Triển khai ResNet baseline          | Đặng Nguyễn Thái Đạt | Baseline model           |
| 5    | Huấn luyện, đánh giá                | Cả hai               | Kết quả                  |
| 6    | Cải tiến, viết báo cáo, build       | Lê Anh Đức           | Bài nộp hoàn chỉnh       |

### Deliverables

| Tuần | Deliverable       | Định dạng   |
| ---- | ----------------- | ----------- |
| 1-2  | Tài liệu khảo sát | Markdown    |
| 3-4  | Source code       | Python      |
| 5    | Model weights     | .pth        |
| 6    | Báo cáo IEEE      | DOCX/PDF    |
| 6    | Slide             | PPTX        |
| 6    | Release           | .exe + DLLs |

---

## VIII. DỰ KIẾN KẾT QUẢ

| Metric    | Baseline | Mục tiêu | Nâng cao |
| --------- | -------- | -------- | -------- |
| Accuracy  | 85%      | 90%      | 93%      |
| F1-score  | 0.85     | 0.90     | 0.92     |
| Precision | 83%      | 88%      | 91%      |
| Recall    | 87%      | 92%      | 93%      |
| Inference | 50ms     | 80ms     | 100ms    |

### Đóng góp dự kiến

- Mô hình hybrid CNN + Frequency cho watermark detection
- Dataset tổng hợp (visible + invisible)
- So sánh các phương pháp
- Chương trình có giao diện

---

## IX. TÀI LIỆU THAM KHẢO

[1] X. Ao et al., "AWPD: Frequency Shield Network for Agnostic Watermark Presence Detection," arXiv:2603.06723, 2026.

[2] R. Xu et al., "InvisMark: Invisible and Robust Watermarking for AI-Generated Image Provenance," arXiv:2411.07795, WACV 2025.

[3] T. Sander et al., "Watermark Anything with Localized Messages," arXiv:2411.07231, ICLR 2025.

[4] S. K. Padhi et al., "Deep Learning-based Dual Watermarking for Image Copyright Protection and Authentication," arXiv:2502.18501, IEEE TAI 2025.

[5] B. Meng et al., "DFCL: Dual-Pathway Fusion Contrastive Learning for Blind Single-Image Visible Watermark Removal," Neural Networks 184, 2025. DOI: 10.1016/j.neunet.2024.107077.

---

## X. GHI CHÚ

1. **Cấu trúc nộp bài:**
   
   ```
   23120227_23120236_Lab03/
   ├── Source/     # Mã nguồn
   ├── Release/    # .exe + DLLs
   └── Docs/       # Báo cáo + Slide
   ```

2. **Báo cáo IEEE:** Abstract → Introduction → Related Work → Proposed Method → Experiments → Conclusion → References

3. **Code:** Chú thích đầy đủ, đặt tên biến/hàm có ý nghĩa

4. **Deadline:** Theo lịch của giảng viên