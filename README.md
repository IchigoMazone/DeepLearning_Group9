# DeepLearning Group 9 - NumPy CNN Fruit Classification

Dự án xây dựng mô hình Convolutional Neural Network (CNN) bằng NumPy để phân loại 5 loại trái cây. Mục tiêu chính là hiểu cơ chế Deep Learning từ đầu: forward propagation, loss, backward propagation và cập nhật tham số bằng Gradient Descent.

## Thành viên nhóm

| STT | Mã sinh viên | Thành viên | Vai trò | Đóng góp |
| --- | --- | --- | --- | --- |
| 1 | 23010600 | Trịnh Như Nhất | Thành viên | 33% |
| 2 | 23010141 | Nguyễn Trần Quang | Thành viên | 33% |
| 3 | 23010625 | Trần Văn Nhật | Thành viên | 33% |

## Bài toán

Mô hình hiện đang tập trung vào bài toán phân loại 5 class:

| Label | Class |
| --- | --- |
| 0 | cucumber |
| 1 | grapes |
| 2 | kiwi |
| 3 | orange |
| 4 | pomegranate |

Dataset gốc tham khảo: [Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

## Kiến trúc mô hình

Mô hình nằm tại `midterm/models/CNN.py`.

Input hiện tại có kích thước:

```text
128 x 128 x 5
```

5 kênh gồm:

```text
RGB + grayscale + edge
```

Các layer chính:

```text
Input
-> Conv2D 3x3, 32 filters
-> ReLU
-> MaxPool 2x2

-> Conv2D 3x3, 64 filters
-> ReLU
-> MaxPool 2x2

-> Conv2D 3x3, 128 filters
-> ReLU

-> Global Average Pooling
-> Global Max Pooling
-> Concatenate

-> Dense 128
-> ReLU
-> Dropout

-> Dense 5
-> Softmax
```

Pipeline học:

```text
forward -> loss -> backward -> Gradient Descent -> update weights
```

Dự án không dùng PyTorch/TensorFlow cho phần huấn luyện chính. Các layer, backward và optimizer được cài bằng NumPy.

## Cấu trúc thư mục chính

```text
DeepLearning_Group9/
├── CollectData/
│   ├── main.py              # Chạy webcam
│   ├── webcam.py            # Thu thập ảnh và predict bằng webcam
│   └── new_dataset/         # Ảnh thu thập thêm theo label 0-4
│
├── datasets/
│   ├── raw/                 # File zip dataset gốc
│   └── processed/           # Dataset đã giải nén/xử lý
│
├── midterm/
│   ├── code/
│   │   ├── layers.py        # Conv, pooling, ReLU, softmax, dense, dropout
│   │   ├── backward.py      # Backpropagation
│   │   ├── optimizers.py    # Gradient Descent
│   │   ├── data.py          # Load ảnh, augment, batch, split CSV
│   │   ├── engine.py        # Train/eval/predict pipeline
│   │   ├── metrics.py       # Accuracy, F1, confusion matrix
│   │   └── checkpoint.py    # Save/load model
│   │
│   ├── datasets/
│   │   ├── dataset.csv      # CSV dataset 5 class
│   │   └── cf/              # train/val/test split
│   │
│   ├── models/
│   │   └── CNN.py           # Kiến trúc CNN
│   │
│   ├── outputs/
│   │   ├── best.pkl         # Checkpoint tốt nhất
│   │   └── latest.pkl       # Checkpoint mới nhất
│   │
│   └── main.py              # CLI chính
│
└── requirements.txt
```

## Cài đặt môi trường

Tạo và kích hoạt môi trường Python, sau đó cài thư viện:

```powershell
cd D:\Git\DeepLearning_Group9
pip install -r requirements.txt
```

Nếu dùng virtual environment có sẵn trong project:

```powershell
.\Sun65\Scripts\Activate.ps1
```

## Chuẩn bị dữ liệu

Nếu đã có `midterm/datasets/dataset.csv` và thư mục split `midterm/datasets/cf/`, có thể train ngay.

Tạo lại train/val/test split:

```powershell
python -m midterm.main --mode split --resplit
```

## Huấn luyện mô hình

Train mô hình 5 class:

```powershell
python -m midterm.main --mode train
```

Train lại từ đầu sau khi tạo lại split:

```powershell
python -m midterm.main --mode train --resplit
```

Một số tham số có thể chỉnh nhanh:

```powershell
python -m midterm.main --mode train --epochs 50 --batch-size 16 --lr 0.001
```

Checkpoint được lưu tại:

```text
midterm/outputs/best.pkl
midterm/outputs/latest.pkl
```

## Đánh giá mô hình

Chạy evaluation trên test set:

```powershell
python -m midterm.main --mode eval
```

Kết quả sẽ in ra:

```text
Test Loss
Accuracy
Macro F1
Weighted F1
Per-class metrics
Confusion matrix
```

## Predict một ảnh

```powershell
python -m midterm.main --mode predict --image path\to\image.jpg
```

Ví dụ:

```powershell
python -m midterm.main --mode predict --image midterm\qua_cam.jpg
```

## Webcam: thu thập dữ liệu và nhận diện trái cây

Chạy webcam:

```powershell
python CollectData\main.py
```

Phím điều khiển:

| Phím | Chức năng |
| --- | --- |
| 0 | Lưu ảnh vào `CollectData/new_dataset/0` |
| 1 | Lưu ảnh vào `CollectData/new_dataset/1` |
| 2 | Lưu ảnh vào `CollectData/new_dataset/2` |
| 3 | Lưu ảnh vào `CollectData/new_dataset/3` |
| 4 | Lưu ảnh vào `CollectData/new_dataset/4` |
| q | Thoát webcam |

Webcam có 2 chức năng:

1. Thu thập thêm ảnh cho từng label bằng phím 0-4.
2. Predict trái cây đang đưa lên camera bằng checkpoint `midterm/outputs/best.pkl`.

## Ghi chú kỹ thuật

- Optimizer hiện tại là Gradient Descent thường, không dùng Adam.
- Model dùng NumPy thuần để minh họa cơ chế Deep Learning.
- Input ảnh được mở rộng thành 5 kênh để giảm phụ thuộc màu sắc:
  - RGB
  - grayscale
  - edge magnitude
- Nếu đổi số class, cần train checkpoint mới vì shape lớp cuối thay đổi.

## Lệnh Git thường dùng

```powershell
git status
git add .
git commit -m "Update NumPy CNN project"
git push
```