**Intensity Transformation** (Biến đổi cường độ ảnh) là kỹ thuật trong xử lý ảnh số dùng để thay đổi độ sáng của các pixel trong ảnh nhằm cải thiện chất lượng ảnh hoặc phục vụ các việc phân tích ảnh.

**Công thức tổng quát**:

$$
s = T(r)
$$

* $r$: indensity đầu vào
* $s$: indensity đầu ra
* $T$: hàm biến đổi

**1. Contract Stretching** (Kéo giãn độ tương phản)

Là kỹ thuật kéo giãn dải indensity của ảnh để tăng cường tương phản và làm rõ ảnh hơn.

<div align="center" 
    style="padding-bottom: 20px; padding-top: 10px;">
    <img src="images/constact_stretching.png" width="500">
    <img src="images/constact_stretching.png" width="500">
</div>

**Đồ thị dạng sigmoid:**    

$$
s = \frac{1}{1 + e^{-z}}
$$

**Nhận xét:**

**1. $\; 0 \leq r < k:$** Vùng tối
<br>
* Hàm tăng chậm, gần như phẳng
* Các pixel vẫn tối, bị nén lại
* Độ tương phản thấp trong vùng này

**2. $\;r \approx k:$** Điểm uốn
<br>
* Độ dốc lớn nhất
* Đây là vùng chuyển tiếp hàm tăng nhanh nhất
* Contract được kéo giãn mạnh nhất tại đây

**3. $k < r \leq r_0:$** Vùng sáng tăng nhanh
<br>
* Hàm tiếp tục tăng nhưng bắt đầu chậm lại
* Các pixel trung bình được kéo lên giá trị cao

**4. $r > r_0:$** Vùng bão hòa
* Hàm phẳng như giai đoạn 1
* Mọi pixel đều được kéo lên một giá trị cao
* Mất chi tiết vùng highlight

**2. Thresholding** 

Là kỹ thuật phân ngưỡng dùng để chuyển ảnh grayscale thành ảnh nhị phân nhằm tách vật thể  ra khỏi ảnh.

<div align="center" 
    style="padding-bottom: 20px; padding-top: 10px;">
    <img src="images/thresholding.png" width="500">
</div>

**Nhận xét:**

**1. $\; 0 \leq r < k: $** Vùng tối
<br>
* Hàm bằng 0 hoàng toàn, phẳng tuyệt đối
* Mội pixel tối đều bị map về 0 (đen)
* Mất toàn bộ chi tiết vùng tối

**2. $\; r=k: $** Điểm ngưỡng
<br>
* Hàm nhảy bậc tức từ 0 lên giá trị tối đa
* Không có vùng chuyển tiếp, độ dốc bằng vô hạn

**3. $\; r>k: $** Vùng sáng
<br>
* Hàm bằng giá trị tối đa, phẳng tuyệt đối
* Mọi pixel sáng đều map về giá trị tối đa
* Mất toàn bộ chi tiết vùng sáng

**3. Log Transformation**

Là phép biến đổi dùng hàm logarithm đề kéo giãn vùng tối và nén vùng sáng

<div align="center" 
    style="padding-bottom: 20px; padding-top: 10px;">
    <img src="images/log_transformation.png" width="500">
</div>

**Công thức toán học:**

$$
s = c \cdot log(1 + r), \; c = \frac{255}{log(1 + r_{max})}
$$

<br>

$$
s = 255 \cdot \frac{log(1 + r)}{log(1 + r_{max})}
$$

<br>

$$
mà \; r \in [0, 255] \rightarrow \frac{log(1 + r)}{log(1 + r_{max})} = \frac{log(1 + r)}{log(256)} \in [0, 1] \rightarrow s = 255 \cdot \frac{log(1 + r)}{log(1 + r_{max})} \in [0, 255]
$$

**Nhận xét:**

**1. $\; 0 \leq r < L/4:$** Vùng tối
<br>
* Hàm tăng rất nhanh, độ dốc lớn nhất
* Đầu ra vọt lên gần $L/2$ trong khi đầu vào chỉ đến $L/4$
* Kéo giãn mạnh nhất ở đây

**2. $\; L/4 \leq r < L/2:$** Điểm uốn
<br>
* Độ dốc giảm dần
* Đầu ra tiến gần $3L/4$
* Vẫn nằm trong idensity, vẫn kéo giãn nhưng mà yếu hơn

**3. $\; L/2 \leq r < 3L/4:$** Vùng sáng tăng nhanh
<br>
* Độ dốc nhỏ hơn 1 (cắt dưới indensity)
* Đầu ra tăng chậm, bắt đầu nén lại

**4. $\; 3L/4 \leq r \leq L-1 :$** Vùng bão hòa
* Hàm tăng rất chậm, độ dốc gần bằng 0
* Nhiều giá trị sáng bị dồn vào dải hẹp gần $L-1$
* Nén mạnh nhất ở đây

<div align="center" style="padding-top: 50px; padding-bottom: 50px;">

| Vùng | r | Độ dốc | Hành vi |
|------|---|--------|---------|
| Tối | $[0, L/4)$ | Lớn nhất | Kéo giãn mạnh |
| Trung bình thấp | $[L/4, L/2)$ | Giảm dần | Kéo giãn nhẹ |
| Trung bình cao | $[L/2, 3L/4)$ | Nhỏ hơn 1 | Bắt đầu nén |
| Sáng | $[3L/4, L-1]$ | Gần 0 | Nén mạnh |

</div>

**4. Negative**

Là phép biến đổi âm bản ánh xạ mỗi cường độ vào sang mức đối xứng trong dải $[0, L-1]$, tạo ra ảnh giống như phim âm bản trong nhiếp ảnh truyền thống.

**Công thức toán học:**

$$
s = (L - 1) - r 
$$