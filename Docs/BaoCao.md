# BÁO CÁO ĐỒ ÁN CUỐI KỲ

## PHÁT HIỆN WATERMARK TRONG ẢNH SỐ

**Môn:** Thị giác Máy tính

**Người thực hiện:**

- Đặng Nguyễn Thái Đạt - MSSV: 23120227 - Email: 23120227@student.hcmus.edu.vn
- Lê Anh Đức - MSSV: 23120236 - Email: 23120236@student.hcmus.edu.vn

---

## I. TÓM TẮT (ABSTRACT)

Nghiên cứu này đề xuất một mô hình lai (hybrid) kết hợp mạng nơ-ron tích chập (CNN) với phân tích miền tần số để phát hiện watermark trong ảnh số. Bài toán phát hiện watermark đóng vai trò quan trọng trong việc bảo vệ bản quyền và xác thực nguồn gốc nội dung số. Mô hình đề xuất sử dụng kiến trúc hai nhánh: nhánh RGB sử dụng ResNet18 để trích xuất đặc trưng không gian, và nhánh tần số sử dụng FFT để nắm bắt các thay đổi vi mô trong miền tần số. Kết quả thực nghiệm trên tập dữ liệu kết hợp CLWD và COCO cho thấy mô hình đạt F1-score 84.76% (Ours v2) và 85.39% khi áp dụng Test-Time Augmentation, với thời gian xử lý dưới 10ms/ảnh trên GPU.

---

## II. GIỚI THIỆU (INTRODUCTION)

### 1.1. Bối cảnh nghiên cứu

Trong kỷ nguyên số hóa hiện nay, việc bảo vệ bản quyền và xác thực nguồn gốc nội dung ảnh trở nên cấp thiết hơn bao giờ hết. Với sự phát triển mạnh mẽ của mạng xã hội và các công cụ tạo nội dung AI (AIGC), hình ảnh có thể được sao chép, chỉnh sửa và phân phối một cách dễ dàng. Watermark (đánh dấu nước) là kỹ thuật nhúng thông tin ẩn vào ảnh số nhằm bảo vệ bản quyền tác phẩm, xác thực nội dung và nguồn gốc, theo dõi phân phối nội dung trên internet, cũng như hỗ trợ các vấn đề pháp lý về gian lận hình ảnh.

Watermark được chia thành hai loại chính: visible watermark (watermark nhìn thấy được như logo, chữ in trên ảnh) và invisible watermark (watermark ẩn, thay đổi pixel ở mức vi mô không nhìn thấy bằng mắt thường nhưng có thể giải mã được). Trong khi các nghiên cứu về watermark embedding (nhúng watermark) đã phát triển mạnh mẽ, bài toán watermark detection (phát hiện watermark) vẫn còn nhiều thách thức cần giải quyết.

### 1.2. Vấn đề nghiên cứu

Bài toán phát hiện watermark trong ảnh số đối mặt với nhiều thách thức:

1. **Watermark nhìn thấy (visible):** Mặc dù tương đối dễ phát hiện bằng mắt thường, việc xác định vị trí chính xác và phân biệt với các đối tượng tương tự (như logo, nhãn hiệu hợp pháp) vẫn là bài toán khó.

2. **Watermark ẩn (invisible):** Phụ thuộc nhiều vào thuật toán nhúng cụ thể, rất khó phát hiện đối với watermark "unknown" (không biết thuật toán gốc).

3. **Watermark yếu:** Các tác động như nén JPEG, resize, blur làm giảm tín hiệu watermark đáng kể, gây khó khăn cho việc phát hiện.

4. **Tổng hợp:** Cần mô hình có khả năng phát hiện cả hai loại watermark với độ chính xác cao trong nhiều điều kiện môi trường khác nhau.

### 1.3. Ý nghĩa thực tiễn

Nghiên cứu về phát hiện watermark có ý nghĩa thực tiễn quan trọng:

- Hỗ trợ kiểm tra bản quyền hình ảnh tự động trên các nền tảng số
- Phát hiện ảnh bị đánh cắp hoặc sử dụng trái phép
- Xác định nguồn gốc ảnh trong điều tra số
- Ứng dụng trong an ninh mạng và truyền thông

### 1.4. Đóng góp của bài báo

Trong bài báo này, chúng tôi đề xuất một mô hình hybrid kết hợp CNN với phân tích miền tần số để giải quyết bài toán phát hiện watermark. Các đóng góp chính bao gồm:

1. Đề xuất kiến trúc hai nhánh kết hợp đặc trưng không gian (spatial) từ ResNet18 và đặc trưng tần số (frequency) từ FFT.
2. Áp dụng cơ chế SE-Net attention để tăng cường khả năng trích xuất đặc trưng.
3. Xây dựng pipeline huấn luyện với các kỹ thuật augmentation đa dạng.
4. Thực nghiệm và đánh giá trên các tập dữ liệu chuẩn.

### 1.5. Cấu trúc bài báo

Phần còn lại của bài báo được tổ chức như sau: Phần III trình bày các nghiên cứu liên quan. Phần IV mô tả chi tiết mô hình đề xuất. Phần V trình bày thực nghiệm và kết quả đánh giá. Phần VI là kết luận và hướng phát triển tương lai.

---

## III. CÁC NGHIÊN CỨU LIÊN QUAN (RELATED WORK)

### 2.1. Tổng quan về các phương pháp phát hiện watermark

Trong literature, các phương pháp phát hiện watermark có thể được chia thành hai nhóm chính: phương pháp truyền thống dựa trên xử lý tín hiệu và phương pháp hiện đại dựa trên học sâu (deep learning). Bảng 1 dưới đây tổng hợp so sánh các phương pháp này.

**Bảng 1. So sánh các phương pháp phát hiện watermark**

| Tiêu chí                 | DCT/DWT    | CNN cơ bản | Frequency-aware CNN | Transformer |
| ------------------------ | ---------- | ---------- | ------------------- | ----------- |
| Tốc độ                   | Cao        | Trung bình | Trung bình          | Trung bình  |
| Độ chính xác (visible)   | Thấp       | Cao        | Cao                 | Cao         |
| Độ chính xác (invisible) | Trung bình | Thấp       | Cao                 | Trung bình  |
| Khả năng tổng quát hóa   | Kém        | Tốt        | Rất tốt             | Tốt         |

### 2.2. Phương pháp truyền thống (Traditional Methods)

Các phương pháp truyền thống dựa trên việc phân tích các đặc tính toán học của ảnh trong miền tần số hoặc không gian.

#### 2.2.1. Phương pháp DCT-based (Discrete Cosine Transform)

Phương pháp DCT phân tích hệ số cosine rời rạc để phát hiện watermark, đặc biệt hiệu quả với các ảnh nén JPEG. Công thức DCT 2D được định nghĩa:

$F(u,v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x,y) \cos\frac{(2x+1)u\pi}{2M} \cos\frac{(2y+1)v\pi}{2N}$

Ưu điểm: Hiệu quả cao với JPEG compression, tính toán nhanh. 

Nhược điểm: Phụ thuộc thuật toán nhúng watermark, khó phát hiện watermark unknown.

#### 2.2.2. Phương pháp DWT-based (Discrete Wavelet Transform)

DWT phân tích wavelet rời rạc với khả năng đa độ phân giải, cho phép phân tích watermark ở nhiều mức tần số khác nhau. Công thức DWT:

$W_\psi(a,b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} f(t) \psi^*\left(\frac{t-b}{a}\right) dt$

Ưu điểm: Đa độ phân giải, phân tích đa tầng. 

Nhược điểm: Khó phát hiện watermark yếu, tốn tính toán.

#### 2.2.3. Phương pháp SVD-based (Singular Value Decomposition)

SVD phân tích giá trị riêng của ma trận ảnh, có khả năng chống nhiễu tốt. 

Ưu điểm: Tolerant với nhiễu. 

Nhược điểm: Tốn tính toán, khó xử lý ảnh lớn.

#### 2.2.4. Phương pháp Statistical

Sử dụng thống kê histogram và phân tích Fourier để phát hiện watermark. 

Ưu điểm: Đơn giản, nhanh. 

Nhược điểm: Accuracy thấp, không hiệu quả với watermark phức tạp.

### 2.3. Phương pháp học sâu (Deep Learning Methods)

Với sự phát triển của deep learning, các phương pháp dựa trên CNN và transformer đã đạt được kết quả vượt trội so với phương pháp truyền thống.

#### 2.3.1. CNN classifiers

Các mô hình CNN như ResNet, VGG, MobileNet được sử dụng cho bài toán phân loại nhị phân (có watermark / không có watermark). ResNet18 và MobileNetV3 là các lựa chọn phổ biến do cân bằng giữa accuracy và inference speed.

Ưu điểm: Accuracy cao với visible watermark, generalization tốt. 

Nhược điểm: Khó phát hiện invisible watermark, cần nhiều dữ liệu training.

#### 2.3.2. YOLO-based detection

YOLO được sử dụng cho bài toán localization - định vị vùng chứa watermark trong ảnh. Phù hợp khi cần xác định vị trí cụ thể của watermark.

#### 2.3.3. Vision Transformer (ViT)

Transformer được áp dụng cho image understanding, có khả năng nắm bắt global context tốt. Tuy nhiên, cần nhiều dữ liệu và tính toán.

#### 2.3.4. Frequency-aware CNN

Các mô hình kết hợp FFT/DCT với CNN để cải thiện khả năng phát hiện invisible watermark. FSNet (Frequency Shield Network) là ví dụ điển hình với ASPM module.

### 2.4. Các nghiên cứu gần đây (2024-2026)

#### 2.4.1. FSNet / AWPD - Frequency Shield Network (arXiv 2603.06723, 2026)

FSNet (còn gọi là AWPD - Agnostic Watermark Presence Detection) đề xuất Adaptive Spectral Perception Module (ASPM) để phát hiện invisible watermark trong miền tần số. ASPM sử dụng learnable frequency gating để amplify high-frequency watermark signals và suppress low-frequency semantics. FSNet đạt zero-shot detection vượt trội trên tập dữ liệu UniFreq-100K.

Công thức ASPM:
$F_{stem} = \Phi_{ASPM}(X) = G_\theta(X) \cdot X$

$G_\theta(X) = \sigma(W_g(\text{AvgPool}(X)))$

#### 2.4.2. InvisMark (arXiv 2411.07795, WACV 2025)

InvisMark giới thiệu encoder-decoder cho AI-generated images, tập trung vào invisible watermark robustness.

#### 2.4.3. WAM - Watermark Anything Model (arXiv 2411.07231, ICLR 2025)

WAM phát hiện và định vị watermark trong vùng cục bộ (localized), cho phép granular detection.

#### 2.4.5. DFCL (Neural Networks 184, 2025)

Dual-Pathway Fusion Contrastive Learning cho visible watermark removal, sử dụng contrastive learning để cải thiện quality.

### 2.5. Nhận xét và lựa chọn phương pháp

Qua khảo sát literature, chúng tôi nhận thấy:

- Phương pháp truyền thống (DCT, DWT, SVD) có tốc độ nhanh nhưng accuracy thấp, đặc biệt với invisible watermark.
- CNN cơ bản đạt accuracy cao với visible watermark nhưng yếu với invisible watermark.
- Frequency-aware CNN là hướng tiềm năng nhất, kết hợp ưu điểm của cả hai phương pháp.

**Lựa chọn của chúng tôi:** Hybrid CNN + Frequency Domain vì:

1. Kế thừa ưu điểm của cả phương pháp truyền thống và học sâu
2. Cải thiện generalization cho invisible watermark
3. Phù hợp với khung thời gian 5-6 tuần
4. Đáp ứng yêu cầu cải tiến nâng cao

---

## IV. MÔ HÌNH ĐỀ XUẤT (PROPOSED METHOD)

### 3.1. Kiến trúc tổng quát

Chúng tôi đề xuất mô hình Hybrid CNN-Frequency với kiến trúc hai nhánh (dual-branch) như Hình 1. Mô hình bao gồm các thành phần chính:

1. **Preprocessing:** Tiền xử lý ảnh đầu vào với hai nhánh
2. **Feature Extraction:** Trích xuất đặc trưng từ cả hai nhánh
3. **Fusion & Attention:** Kết hợp đặc trưng và áp dụng attention
4. **Classification:** Phân loại nhị phân

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

**Hình 1. Kiến trúc tổng quát của mô hình đề xuất**

### 3.2. Tiền xử lý (Preprocessing)

#### 3.2.1. Nhánh RGB

- **Resize:** 224×224
- **Normalization:** Sử dụng ImageNet mean và std
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

#### 3.2.2. Nhánh tần số (Frequency Branch)

Ảnh đầu vào được chuyển đổi sang miền tần số sử dụng FFT:

$F_{freq} = \log(|\text{FFT}(I)| + 1)$

Trong đó:

- FFT(I): Fast Fourier Transform của ảnh đầu vào
- |.|: Lấy magnitude
- log: Log transformation để compress dynamic range

Quy trình tạo frequency representation:

1. Chuyển ảnh sang grayscale (nếu ảnh RGB)
2. Resize về 224×224
3. Áp dụng FFT 2D
4. Shift để đưa DC component ra giữa
5. Lấy magnitude spectrum
6. Log transform: log(magnitude + 1) để compress dynamic range
7. Normalize về [0, 1] bằng min-max scaling



### 3.3. Trích xuất đặc trưng (Feature Extraction)

#### 3.3.1. RGB Backbone

Sử dụng ResNet18 pretrained trên ImageNet làm backbone cho nhánh RGB. ResNet18 có 11.7M parameters, cung cấp sự cân bằng tốt giữa performance và efficiency.

Chi tiết kiến trúc ResNet18:

- Conv1: 7×7, 64, stride 2
- MaxPool: 3×3, stride 2
- Layer1: BasicBlock × 2 (64 channels)
- Layer2: BasicBlock × 2 (128 channels)
- Layer3: BasicBlock × 2 (256 channels)
- Layer4: BasicBlock × 2 (512 channels)
- Global Average Pooling

Output: 512-dimensional feature vector

#### 3.3.2. Frequency Branch

Nhánh tần số sử dụng 1D CNN với 3 layers để trích xuất đặc trưng từ frequency representation (input shape: 224 length × 3 channels, sau khi repeat từ grayscale FFT 1 channel):

| Layer         | Kernel Size | Stride | Channels | Output (length × channels) |
| ------------- | ----------- | ------ | -------- | -------------------------- |
| Conv1d_1      | 3           | 2      | 32       | 112 × 32                   |
| Conv1d_2      | 3           | 2      | 64       | 56 × 64                    |
| Conv1d_3      | 3           | 2      | 128      | 28 × 128                   |
| GlobalAvgPool | -           | -      | -        | 128                        |

Output: 128-dimensional feature vector

#### 3.3.3. Fusion

Hai nhánh được kết hợp thông qua concatenation:

$F_{fused} = \text{Concat}(F_{RGB}, F_{freq})$

$F_{fused} \in \mathbb{R}^{640} (512 + 128)$

Sau đó, một FC layer giảm chiều: FC(640 → 512)

### 3.4. Channel Attention (SE-Net)

Áp dụng Squeeze-and-Excitation (SE) attention để tăng cường đặc trưng:

**Bước 1 - Squeeze (Global Average Pooling):**
$z = F_{sq}(u) = \frac{1}{H \times W}\sum_{i=1}^{H}\sum_{j=1}^{W}u_{ij}$

**Bước 2 - Excitation:**
$s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot z))$

**Bước 3 - Scale:**
$v = F_{scale}(u, s) = s \cdot u$

Trong đó:

- $W_1 \in \mathbb{R}^{C/r \times C}$ (r = 16 là reduction ratio)
- $W_2 \mathbb{R}^{C \times C/r}$
- $\sigma$: Sigmoid activation

### 3.5. Phân loại (Classification)

Lớp classification bao gồm:

| Layer   | Configuration                   |
| ------- | ------------------------------- |
| FC1     | 512 → 256, ReLU                 |
| Dropout | 0.5                             |
| FC2     | 256 → 2 (Binary classification) |
| Output  | Softmax                         |

### 3.6. Hàm mất mát (Loss Function)

Sử dụng Cross-entropy loss:

$\mathcal{L}_{CE} = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$

Trong đó:

- $y_i$: ground truth label
- $\hat{y}_i$: predicted probability
- K = 2 (binary classification)

### 3.7. Tối ưu hóa (Optimizer)

- **Optimizer:** AdamW
- **Learning rate:** 1e-4
- **Weight decay:** 1e-2
- **Scheduler:** Cosine annealing

$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\frac{t\pi}{T_{max}}\right)$

Trong đó $\eta_t$ là learning rate tại step t, $T_{max}$ là tổng số epochs.

### 3.8. Data Augmentation

Để tăng cường khả năng tổng quát hóa, chúng tôi áp dụng các kỹ thuật augmentation. Pipeline được triển khai trong `Source/dual_dataset.py` gồm hai giai đoạn:

**Giai đoạn 1** (numpy/cv2, `_augment()`): áp dụng flip, JPEG, noise trước khi convert sang PIL.

**Giai đoạn 2** (torchvision, `_get_transform()`): áp dụng flip, rotation, color jitter, ToTensor, Normalize.

| Kỹ thuật               | Tham số                      | Xác suất      | Giai đoạn |
| ---------------------- | ---------------------------- | ------------- | --------- |
| Random Horizontal Flip | -                            | 0.5           | 1 + 2     |
| Random Rotation        | ±15°                         | 0.5           | 2         |
| Color Jitter           | brightness=0.2, contrast=0.2 | luôn áp dụng* | 2         |
| JPEG Compression       | quality=70-90                | 0.3           | 1         |
| Gaussian Noise         | σ=5-15                       | 0.2           | 1         |

*Ghi chú: `transforms.ColorJitter` không hỗ trợ tham số `p`, nên mỗi ảnh training đều được áp dụng brightness/contrast jitter (không có bước skip với xác suất 0.3 như các augmentation khác).

### 3.9. Cải tiến nâng cao (Advanced Improvements)

Ngoài kiến trúc hybrid chính, nhóm triển khai hai cải tiến inference (TTA, Multi-scale) và đề xuất hướng phát triển tương lai Semi-supervised Learning, nhằm cải thiện độ robust của hệ thống khi triển khai thực tế. Kết quả thực nghiệm ablation được trình bày tại mục 3.9.4.

#### 3.9.1. Test-Time Augmentation (TTA)

**Mục đích:** Giảm phương sai dự đoán bằng cách áp dụng các phép biến đổi hình học nhẹ lên ảnh đầu vào tại thời điểm kiểm thử, rồi lấy trung bình xác suất dự đoán từ nhiều phiên bản augmented. Kỹ thuật này tận dụng tính bất biến của mô hình đối với các biến đổi mà mô hình đã học được trong quá trình training.

**Các phép biến đổi sử dụng:**

| Phép biến đổi   | Mô tả           | Mục đích                                            |
| --------------- | --------------- | --------------------------------------------------- |
| Original        | Ảnh gốc 224×224 | Dự đoán chính                                       |
| Horizontal Flip | Lật ngang ảnh   | Mô hình đã học invariance theo flip từ augmentation |
| Rotation +10°   | Xoay ảnh +10 độ | Mô hình đã học invariance theo rotation ±15°        |
| Rotation −10°   | Xoay ảnh −10 độ | Bù đối xứng với +10° để bảo toàn trung bình         |

**Cơ chế ensemble:** Softmax probabilities của 4 phiên bản được tính trung bình, sau đó lấy argmax. Đây là soft-voting ở mức xác suất, hiệu quả hơn hard-voting khi có sự khác biệt nhỏ giữa các dự đoán.

**Cài đặt:** Triển khai trong `Source/detect.py` với cờ `--tta`. Tất cả phép biến đổi sử dụng `torchvision.transforms.functional` để chạy trên GPU theo batch, đảm bảo tốc độ inference không bị giảm đáng kể (hệ số x10 lần thời gian, chấp nhận được cho bài toán batch offline).

#### 3.9.2. Multi-scale Inference

**Mục đích:** Mô hình được huấn luyện ở độ phân giải cố định 224×224. Tuy nhiên watermark có thể xuất hiện ở nhiều kích thước khác nhau, từ watermark nhỏ chỉ chiếm vài pixel đến watermark lớn phủ toàn bộ ảnh. Multi-scale inference giúp mô hình "nhìn" watermark ở các mức chi tiết khác nhau.

**Chiến lược:** Inference ở hai scale 224 và 448, lấy trung bình xác suất dự đoán. Lưu ý: do mô hình được train ở 224, scale 448 chỉ mang tính chất thăm dò (exploratory) và được kỳ vọng cải thiện recall nhưng có thể giảm precision vì mô hình chưa thấy kích thước này trong training. Kết quả thực nghiệm chi tiết được trình bày tại mục 3.9.4 (Bảng 2).

**Cài đặt:** Triển khai trong `Source/detect.py` với cờ `--multi_scale`. Resize 224→448 sử dụng bilinear interpolation.

#### 3.9.3. Semi-supervised Learning (Hướng phát triển tương lai)

**Mục đích:** Mở rộng tập training bằng cách sử dụng pseudo-labeling trên các ảnh không có nhãn (unlabeled), tận dụng lượng lớn ảnh watermark trên Internet chưa được gán nhãn.

**Cơ chế dự kiến:**

1. Huấn luyện mô hình teacher trên tập labeled (đã thực hiện: mô hình Ours v2 đạt F1 84.76%)
2. Sử dụng teacher gán nhãn cho tập unlabeled, lấy mẫu có confidence cao (>0.95)
3. Huấn luyện mô hình student trên tập hợp (labeled + pseudo-labeled)
4. Lặp lại quá trình với student làm teacher mới (self-training)

**Tình trạng triển khai:** Do giới hạn thời gian và nguồn lực tính toán, nhóm chưa thực nghiệm phương pháp này trong đồ án này. Hướng phát triển này sẽ được đề cập trong phần Kết luận như một hướng nghiên cứu tiếp theo có tiềm năng cải thiện đáng kể độ robust của mô hình.

#### 3.9.4. Kết quả Ablation Study thực nghiệm

Nhóm thực hiện ablation study trên tập test (21.750 mẫu) với mô hình Ours v2 (RGB+Freq+SE) để đánh giá tác động thực sự của từng cải tiến inference. Kết quả được trình bày tại Bảng 2.

**Bảng 2. Tác động của TTA và Multi-scale (đo trên 21.750 mẫu test)**

| Cấu hình                  | Accuracy   | F1         | Precision | Recall | Thời gian / ảnh |
| ------------------------- | ---------- | ---------- | --------- | ------ | --------------- |
| Baseline (không cải tiến) | 84.99%     | 84.76%     | 89.29%    | 80.66% | 0.3 ms          |
| TTA (flip + ±10° rotate)  | **85.94%** | **85.39%** | 92.33%    | 79.41% | 3.0 ms          |
| Multi-scale (224+448)     | 84.77%     | 83.39%     | 95.66%    | 73.90% | 0.4 ms          |
| TTA + Multi-scale         | 83.37%     | 81.24%     | 97.52%    | 69.62% | 13.5 ms         |

**Phân tích kết quả:**

- **TTA đơn lẻ cải thiện F1 +0.63% và Accuracy +0.95%**, đồng thời tăng Precision lên 92.33% (do ensemble giúp giảm false positive). Đây là cải tiến có hiệu quả rõ ràng và được khuyến nghị sử dụng trong triển khai thực tế, với chi phí inference tăng gấp 10 lần (chấp nhận được cho batch processing).

- **Multi-scale đơn lẻ giảm F1 −1.37%**, mặc dù Precision tăng lên 95.66% (do ensemble từ 2 scale giúp mô hình "chắc chắn" hơn khi dự đoán positive). Tuy nhiên Recall giảm mạnh −6.76% là nguyên nhân chính. Lý do: mô hình được train ở 224×224, khi inference ở 448×448, các feature extractor đã học pattern ở scale 224 sẽ không tổng quát hóa tốt, dẫn đến bỏ sót nhiều watermark thật (false negative). Để multi-scale có hiệu quả, cần fine-tune mô hình trên nhiều scale trong training (multi-scale training), nằm ngoài phạm vi đồ án.

- **TTA + Multi-scale kết hợp giảm F1 −3.52%**, bị ảnh hưởng kép từ cả hai yếu tố. Tổng số inferences lên tới 8 lần (4 TTA × 2 scale), thời gian tăng gấp 44 lần (13.5 ms/ảnh) nhưng hiệu quả suy giảm.

**Kết luận ablation:** Chỉ TTA đơn lẻ mang lại cải thiện thực sự (+0.63% F1) và được tích hợp vào hệ thống detect.py làm cờ mặc định `--tta`. Multi-scale cần fine-tune ở multi-scale training mới phát huy hiệu quả, do đó tài liệu hóa như hướng phát triển tương lai.

---

## V. THỰC NGHIỆM VÀ KẾT QUẢ (EXPERIMENTS)

### 4.1. Datasets

#### 4.1.1. Tập dữ liệu sử dụng

| Dataset              | Nguồn            | Số lượng (Watermark) | Số lượng (No-WM) | Tổng        | Loại WM   |
| -------------------- | ---------------- | -------------------- | ---------------- | ----------- | --------- |
| CLWD                 | arXiv 2012.07616 | 70,000               | 70,000           | 140,000     | Visible   |
| COCO val + synthetic | COCO 2017        | 5,000                | 0                | 5,000       | Invisible |
| **Combined**         | -                | **75,000**           | **70,000**       | **145,000** | Cả hai    |

#### 4.1.2. Chia train/val/test

- Training: 70%
- Validation: 15%
- Testing: 15%

### 4.2. Metrics đánh giá

| Metric         | Công thức     | Ý nghĩa              |
| -------------- | ------------- | -------------------- |
| Accuracy       | (TP+TN)/Total | Độ chính xác tổng    |
| Precision      | TP/(TP+FP)    | Độ chính xác dự đoán |
| Recall         | TP/(TP+FN)    | Độ nhạy              |
| F1-score       | 2PR/(P+R)     | Cân bằng P/R         |
| Inference time | ms/ảnh        | Tốc độ               |

### 4.3. Môi trường thực nghiệm

| Thành phần | Phiên bản       |
| ---------- | --------------- |
| Python     | 3.8+            |
| PyTorch    | 2.0+            |
| OpenCV     | 4.8+            |
| CUDA       | 11.8+           |
| GPU        | RTX 3060+ (8GB) |

### 4.4. Kết quả

| Metric                    | Baseline 1 (ResNet18) | Baseline 2 (MobileNet) | Ours v1 | Ours v2 | **Ours v2 + TTA** |
| ------------------------- | --------------------- | ---------------------- | ------- | ------- | ----------------- |
| Accuracy                  | 84.15%                | 78.06%                 | 85.12%  | 84.99%  | **85.94%**        |
| Precision                 | 87.62%                | 91.08%                 | 93.64%  | 89.29%  | 92.33%            |
| Recall                    | 80.76%                | 63.83%                 | 76.42%  | 80.66%  | 79.41%            |
| F1-score                  | 84.05%                | 75.06%                 | 84.16%  | 84.76%  | **85.39%**        |
| Inference (full pipeline) | 4.9ms                 | 7.8ms                  | 6.1ms   | 7.3ms   | ~29.2ms*          |

*Ghi chú về inference time:*

- *4 cột đầu (4.9/7.8/6.1/7.3ms): đo bằng full pipeline (data loading + forward pass) trên GPU RTX 3060, batch size 32.*
- *Cột **Ours v2 + TTA = ~29.2ms**: ước lượng = 7.3ms × 4 forward passes (TTA ensemble: original + flip + rotate±10°). Đây là ước lượng dựa trên việc TTA thực hiện 4 lần forward pass cho cùng một ảnh. Kết quả đo chi tiết forward-pass-only bằng `evaluate_advanced.py` được trình bày tại mục 3.9.4 (Bảng 2: 3.0ms per-image ở batch 32, thấp hơn nhiều so với ước lượng tuần tự nhờ song song hóa batch).*

**Nhận xét:** Trong số các mô hình không áp dụng TTA, **Ours v1 đạt accuracy cao nhất (85.12%)** với precision 93.64% (cũng cao nhất bảng). Khi áp dụng TTA, cấu hình **Ours v2 + TTA** đạt accuracy tổng thể cao nhất 85.94% và F1-score tốt nhất 85.39%. Ours v2 cân bằng hơn với F1-score 84.76% và recall 80.66%, cho thấy SE attention giúp cải thiện khả năng phát hiện watermark thực sự (recall tăng từ 76.42% ở v1 lên 80.66% ở v2). Mô hình hybrid CNN-Frequency cải thiện đáng kể so với baseline ResNet18 (+0.97% accuracy với Ours v1, +0.84% với Ours v2). So với Ours v2 không TTA, TTA cải thiện thêm +0.95% accuracy và +0.63% F1-score. Chi tiết ablation study được trình bày tại mục 3.9.4.

---

### 4.5. So sánh với các nghiên cứu liên quan

### 4.5.1. Tổng quan về các nghiên cứu liên quan

Các nghiên cứu liên quan gần đây trong lĩnh vực phát hiện và nhúng watermark bao gồm FSNet (arXiv 2603.06723, 2026), WAM (ICLR 2025), và InvisMark (WACV 2025). FSNet là phương pháp detection với zero-shot capability trên UniFreq-100K, trong khi WAM và InvisMark là các phương pháp watermarking (nhúng và trích xuất thông điệp).

### 4.5.2. So sánh các phương pháp Detection

Bảng 3 dưới đây so sánh các phương pháp detection (phân loại nhị phân: có watermark / không có watermark).

| Phương pháp  | Task             | Dataset      | F1-score                  | Ghi chú                                        |
| ------------ | ---------------- | ------------ | ------------------------- | ---------------------------------------------- |
| FSNet (ASPM) | Zero-shot AWPD   | UniFreq-100K | 0.26-0.99 (per-algorithm) | No aggregate score; leave-one-algorithm-out CV |
| **Ours v1**  | Binary detection | CLWD + COCO  | **0.842**                 | Visible + Invisible, supervised                |
| **Ours v2**  | Binary detection | CLWD + COCO  | **0.848**                 | + SE Attention                                 |

**Ghi chú:** FSNet được đánh giá trên UniFreq-100K với leave-one-algorithm-out cross-validation (zero-shot). Kết quả F1 biến thiên từ 0.26 (LSB/Patchwork) đến 0.99 (HiDDeN) tùy thuật toán nhúng. Mô hình của nhóm được huấn luyện và đánh giá trên dataset kết hợp visible (CLWD) + invisible (COCO), với F1-score trung bình 0.84-0.85. Do sự khác biệt về dataset và task formulation (zero-shot vs supervised), không thể so sánh trực tiếp các chỉ số.

### 4.5.3. So sánh các phương pháp Watermarking (tham khảo)

Bảng 4 dưới đây trình bày các phương pháp watermarking (nhúng và trích xuất thông điệp) để tham khảo. Các phương pháp này không phải task detection nên không thể so sánh F1-score trực tiếp.

| Phương pháp | Task                   | Metric       | Giá trị             | Ghi chú                             |
| ----------- | ---------------------- | ------------ | ------------------- | ----------------------------------- |
| WAM         | Localized watermarking | Bit error    | <1 bit (32-bit msg) | ICLR 2025; có khả năng localization |
| InvisMark   | AI-image watermarking  | Bit accuracy | 97%                 | WACV 2025; 256-bit payload, PSNR~51 |

### 4.5.4. Phân tích định tính

**Điểm mạnh của mô hình đề xuất:**

1. **Dual-branch architecture:** Kết hợp đặc trưng không gian (ResNet18) và tần số (FFT + 1D CNN) trong một pipeline thống nhất, cho phép phát hiện cả visible và invisible watermark.

2. **Nhẹ và nhanh:** Với ~12M parameters, inference time chỉ 6.1-7.3ms trên GPU RTX 3060, phù hợp cho ứng dụng thực tế.

3. **SE Attention đơn giản:** Ours v2 sử dụng SE-Net (reduction=16) để recalibrate channel features, cải thiện recall từ 76.42% (v1) lên 80.66% (v2).

**Điểm yếu so với SOTA:**

1. **Chưa có adaptive gating:** FSNet sử dụng ASPM (Adaptive Spectral Perception Module) với learnable frequency gating để amplify high-frequency watermark signals. Mô hình của nhóm dùng FFT cố định, chưa có cơ chế adaptive.

2. **Chưa hỗ trợ localization:** WAM có khả năng định vị vùng chứa watermark trong ảnh (granular detection). Mô hình của nhóm chỉ thực hiện binary classification.

3. **Dataset nhỏ hơn:** UniFreq-100K có ~100K samples cho invisible watermark, trong khi invisible dataset của nhóm chỉ có 5,000 samples.

### 4.5.5. Hướng cải tiến tương lai

Dựa trên phân tích trên, các hướng phát triển tiếp theo bao gồm:

1. **Tích hợp ASPM module:** Thêm learnable frequency gating vào Frequency Branch để adaptive amplify watermark signals.

2. **Thêm localization head:** Kết hợp detection head (YOLO-style) để định vị vùng chứa watermark.

3. **Mở rộng dataset:** Thu thập thêm invisible watermark samples hoặc test trên UniFreq-100K để so sánh trực tiếp với FSNet.

4. **Multi-scale Training:** Huấn luyện ở nhiều resolution (224/256/320) để mô hình generalize tốt hơn, giúp multi-scale inference đạt hiệu quả (hiện tại multi-scale inference bị giảm F1 do mô hình chỉ thấy 224 trong training).

## VI. KẾT LUẬN (CONCLUSION)

Trong bài báo này, chúng tôi đã đề xuất một mô hình hybrid kết hợp CNN với phân tích miền tần số để phát hiện watermark trong ảnh số. Mô hình sử dụng kiến trúc hai nhánh với ResNet18 cho đặc trưng không gian và 1D CNN cho đặc trưng tần số, kết hợp với cơ chế SE-Net attention.

Các đóng góp chính:

1. Kiến trúc dual-branch kết hợp spatial và frequency features, đạt F1-score 84.76% (Ours v2) vượt baseline ResNet18 0.71%
2. Cơ chế SE attention giúp cân bằng precision/recall, nâng recall từ 76.42% (v1) lên 80.66% (v2)
3. Pipeline augmentation đa dạng với 5 phép biến đổi (flip, rotation, color jitter, JPEG, Gaussian noise) tăng độ robust
4. Tích hợp Test-Time Augmentation cải thiện F1 thêm +0.63% (lên 85.39%) trong khi giữ inference time chấp nhận được (3.0ms/ảnh)
5. Toàn bộ hệ thống được đóng gói thành CLI (`detect.py`) với các cờ `--tta` và `--multi_scale` cho triển khai thực tế

Hướng phát triển tương lai:

- **Semi-supervised Learning với pseudo-labeling:** Tận dụng lượng lớn ảnh không nhãn trên Internet bằng cơ chế self-training (teacher-student) để mở rộng tập huấn luyện. Kỳ vọng cải thiện recall đáng kể vì mô hình sẽ được "thấy" nhiều biến thể watermark hơn ngoài dữ liệu có nhãn hiện tại (140.000 mẫu).
- **Multi-scale Training:** Hiện tại mô hình chỉ train ở resolution 224, dẫn đến multi-scale inference bị giảm F1 (-1.37%). Training ở nhiều scale (224/256/320) sẽ giúp mô hình generalize tốt hơn và phát huy hiệu quả của multi-scale inference.
- **Tích hợp ASPM module từ FSNet** để cải thiện zero-shot detection trên các thuật toán watermark chưa thấy trong training.
- **Thử nghiệm với backbone lớn hơn** (ResNet50, EfficientNet-B3, ConvNeXt-Tiny) để đánh đổi giữa độ phức tạp và độ chính xác.
- **Bổ sung bài toán localization** (xác định vùng ảnh chứa watermark) thay vì chỉ phân loại nhị phân.
- **Tối ưu hóa cho edge devices** bằng quantize/prune, hướng đến triển khai mobile (iOS/Android) cho ứng dụng bảo vệ bản quyền tự động.

---

## TÀI LIỆU THAM KHẢO

[1] X. Ao et al., "AWPD: Frequency Shield Network for Agnostic Watermark Presence Detection," arXiv:2603.06723, 2026.

[2] R. Xu et al., "InvisMark: Invisible and Robust Watermarking for AI-Generated Image Provenance," arXiv:2411.07795, WACV 2025.

[3] T. Sander et al., "Watermark Anything with Localized Messages," arXiv:2411.07231, ICLR 2025.

[4] S. K. Padhi et al., "Deep Learning-based Dual Watermarking for Image Copyright Protection and Authentication," arXiv:2502.18501, IEEE TAI 2025.

[5] B. Meng et al., "DFCL: Dual-Pathway Fusion Contrastive Learning for Blind Single-Image Visible Watermark Removal," Neural Networks 184, 2025. https://doi.org/10.1016/j.neunet.2024.107077

[6] F. Xie et al., "SpecGuard: Spectral Projection-based Advanced Invisible Watermarking," arXiv:2510.07302, ICCV 2025.