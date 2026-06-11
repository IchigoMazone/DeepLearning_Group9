# DeepLearning Group 9 - NumPy CNN Fruit Classification

Du an xay dung mo hinh Convolutional Neural Network (CNN) bang NumPy de phan loai 5 loai trai cay. Muc tieu cua project la hieu ro co che Deep Learning tu dau: forward propagation, loss, backward propagation va cap nhat trong so bang Gradient Descent.

Project khong dung PyTorch/TensorFlow cho phan huan luyen CNN chinh. Cac layer, backward, optimizer va metrics duoc cai dat bang NumPy.

## Thanh vien nhom

| STT | Ma sinh vien | Thanh vien | Vai tro | Dong gop |
| --- | --- | --- | --- | --- |
| 1 | 23010600 | Trinh Nhu Nhat | Thanh vien | 33% |
| 2 | 23010141 | Nguyen Tran Quang | Thanh vien | 33% |
| 3 | 23010625 | Tran Van Nhat | Thanh vien | 33% |

## Bai toan

Mo hinh hien tai phan loai 5 class trai cay:

| Label | Class |
| --- | --- |
| 0 | Fruits_Cucumber |
| 1 | Fruits_Grapes |
| 2 | Fruits_Kiwi |
| 3 | Fruits_Orange |
| 4 | Fruits_Pomegranate |

Dataset goc tham khao: [Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

## Kien truc mo hinh

File kien truc:

```text
midterm/models/CNN.py
```

Kien truc CNN hien tai:

```text
Input 64x64x3
-> Conv 5x5, 16 filters -> ReLU -> MaxPool 2x2
-> Conv 3x3, 32 filters -> ReLU -> MaxPool 2x2
-> Conv 3x3, 64 filters -> ReLU -> MaxPool 2x2
-> Flatten
-> Dense 128 -> ReLU -> Dropout
-> Dense 5 -> Softmax
```

Input la anh RGB:

```text
64 x 64 x 3
```

Sau 3 lan MaxPool:

```text
64x64 -> 32x32 -> 16x16 -> 8x8
```

Sau Conv3 co 64 filters, Flatten tao vector:

```text
8 x 8 x 64 = 4096
```

Pipeline hoc:

```text
forward -> loss -> backward -> Gradient Descent -> update W,b
```

## Cau truc thu muc

```text
DeepLearning_Group9/
|-- CollectData/
|   |-- main.py
|   |-- webcam.py
|   `-- new_dataset/
|
|-- datasets/
|   |-- raw/
|   `-- processed/
|       `-- fruits5/
|
|-- midterm/
|   |-- code/
|   |   |-- data.py          # Load anh, split CSV, batch, augmentation
|   |   |-- layers.py        # Forward: Conv, ReLU, Pooling, Flatten, Dense, Dropout, Softmax
|   |   |-- backward.py      # Backward propagation
|   |   |-- im2col.py        # im2col/col2im cho convolution
|   |   |-- optimizers.py    # Gradient Descent
|   |   |-- metrics.py       # Loss, accuracy, F1, confusion matrix
|   |   |-- checkpoint.py    # Save/load model .pkl
|   |   `-- engine.py        # Train/eval/predict pipeline
|   |
|   |-- datasets/
|   |   |-- dataset_5.csv
|   |   `-- cf_5/
|   |       |-- train.csv
|   |       |-- val.csv
|   |       `-- test.csv
|   |
|   |-- models/
|   |   `-- CNN.py
|   |
|   |-- outputs/
|   |   |-- best_cf5_numpy.pkl
|   |   |-- latest_cf5_numpy.pkl
|   |   `-- reports/
|   |
|   |-- train/
|   |   `-- train.py
|   |
|   `-- main.py
|
|-- requirements.txt
|-- requirements-cpu.txt
|-- requirements-cu121.txt
`-- README.md
```

## Package code

### data.py

Xu ly du lieu:

- Doc anh tu duong dan trong CSV.
- Resize anh ve `64x64`.
- Chuan hoa pixel ve `[0,1]`.
- Chia dataset thanh train/val/test.
- Tao batch khi train.
- Augmentation anh khi train.

Mac dinh hien tai:

```text
add_structure = False
```

Nghia la mo hinh dung anh RGB 3 channel, khong them grayscale/edge channel.

### layers.py

Chua cac ham forward:

- `conv_forward`
- `relu`
- `max_pool_forward`
- `flatten_forward`
- `dense_forward`
- `dropout_forward`
- `softmax`

### backward.py

Chua cac ham backward:

- `conv_backward`
- `relu_backward`
- `max_pool_backward`
- `flatten_backward`
- `dense_backward`
- `dropout_backward`
- `model_backward`

`model_backward()` tra ve gradient cua cac tham so:

```text
dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5
```

### im2col.py

Toi uu convolution bang cach bien cac cua so truot cua anh thanh ma tran:

```text
X_col @ W_col
```

### optimizers.py

Chua optimizer `GradientDescent`.

Cong thuc cap nhat:

```text
W = W - learning_rate * dW
b = b - learning_rate * db
```

### metrics.py

Tinh loss va cac chi so danh gia:

- Cross entropy loss
- Accuracy
- Precision
- Recall
- F1 score
- Macro F1
- Weighted F1
- Top-3 accuracy
- Confusion matrix

### checkpoint.py

Luu va load checkpoint `.pkl`, gom:

```text
parameters W,b
epoch
val_loss
val_acc
class_names
image_size
num_classes
metadata
history
```

### engine.py

Dieu phoi train/eval/predict:

```text
load data
-> init CNN
-> create batches
-> augment
-> forward
-> compute loss
-> backward
-> optimizer step
-> evaluate validation
-> save checkpoint
-> export reports
```

## Cai dat moi truong

Di vao thu muc project:

```powershell
cd D:\Git\DeepLearning_Group9
```

Neu dung virtual environment co san:

```powershell
.\Sun65\Scripts\Activate.ps1
```

Cai thu vien:

```powershell
pip install -r requirements.txt
```

Neu chi chay CPU:

```powershell
pip install -r requirements-cpu.txt
```

## Du lieu

Dataset 5 class hien tai:

```text
midterm/datasets/dataset_5.csv
```

Split train/val/test:

```text
midterm/datasets/cf_5/train.csv
midterm/datasets/cf_5/val.csv
midterm/datasets/cf_5/test.csv
```

Thu muc anh dang duoc CSV tro toi:

```text
datasets/processed/fruits5/Fruits/
```

Tao lai split:

```powershell
python -m midterm.main --mode split
```

Train va tao lai split truoc khi train:

```powershell
python -m midterm.main --mode train --resplit
```

## Huan luyen

Train voi tham so mac dinh trong `midterm/main.py`:

```powershell
python -m midterm.main --mode train
```

Train nhanh voi tham so tu command line:

```powershell
python -m midterm.main --mode train --epochs 50 --batch-size 16 --lr 0.005
```

Tat augmentation khi train:

```powershell
python -m midterm.main --mode train --no-augment
```

Tham so mac dinh quan trong:

```text
target_size = (64, 64)
input_channels = 3
num_classes = 5
epochs = 50
batch_size = 16
lr = 0.005
dropout_keep_prob = 0.85
label_smoothing = 0.0
add_structure = False
```

Checkpoint:

```text
midterm/outputs/best_cf5_numpy.pkl
midterm/outputs/latest_cf5_numpy.pkl
```

Bao cao train/eval:

```text
midterm/outputs/reports/
```

## Danh gia

Chay evaluate tren test set:

```powershell
python -m midterm.main --mode eval
```

Ket qua gom:

```text
Test Loss
Accuracy
Macro F1
Weighted F1
Top-3 Accuracy
Per-class metrics
Confusion matrix
```

## Predict mot anh

```powershell
python -m midterm.main --mode predict --image "path\to\image.jpg"
```

Vi du:

```powershell
python -m midterm.main --mode predict --image "D:\Git\DeepLearning_Group9\midterm\qua_cam.jpg"
```

Chuong trinh se:

```text
load best_cf5_numpy.pkl
-> load anh
-> resize ve 64x64x3
-> forward qua CNN
-> in class du doan va confidence
```

## Webcam

Chay webcam:

```powershell
python CollectData\main.py
```

Chuc nang:

1. Collect data: dua trai cay len camera, bam phim label `0-4` de luu anh.
2. Predict: dua trai cay len camera, load checkpoint va hien class du doan.

Bang label:

| Phim | Label | Class |
| --- | --- | --- |
| 0 | 0 | Fruits_Cucumber |
| 1 | 1 | Fruits_Grapes |
| 2 | 2 | Fruits_Kiwi |
| 3 | 3 | Fruits_Orange |
| 4 | 4 | Fruits_Pomegranate |
| q | - | Thoat webcam |

Anh collect them chua tu dong lam mo hinh tot hon. Sau khi collect them anh can cap nhat dataset va train lai.

## Ghi chu ky thuat

- Mo hinh hien tai dung input `64x64x3`.
- CNN dung 3 block `Conv -> ReLU -> MaxPool`.
- Mo hinh dung `Flatten`, khong dung Global Average Pool / Global Max Pool.
- Optimizer la Gradient Descent thuong.
- `W,b` duoc khoi tao mot lan dau qua trinh train, sau moi batch se duoc cap nhat va dung tiep cho batch/epoch sau.
- Neu doi kien truc CNN hoac doi so class, can train lai checkpoint moi.

## Lenh Git thuong dung

```powershell
git status
git add .
git commit -m "Update NumPy CNN project"
git push origin feature/Sun65
```
