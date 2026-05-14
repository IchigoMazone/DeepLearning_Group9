# Kế hoạch xây dựng Dataset phân loại Xoài 5 Cấp độ

Tài liệu này hệ thống hóa các tiêu chuẩn phân loại xoài trên băng chuyền công nghiệp phục vụ cho việc huấn luyện mô hình YOLO (Object Detection/Classification).

## 1. Bảng tiêu chuẩn phân loại chi tiết

| Cấp độ   | Tên gọi        | Đặc điểm nhận dạng (YOLO Labeling)                        | Số lượng ảnh yêu cầu   | Mục đích sử dụng       |
|:---------|:---------------|:----------------------------------------------------------|:-----------------------|:-----------------------|
| G1       | Xanh cứng      | Vỏ xanh đậm hoàn toàn, bề mặt căng bóng, không vết thâm.  | 1000 ảnh               | Xuất khẩu xa           |
| G2       | Ươm (Hơi chín) | Vỏ xanh nhạt, xuất hiện đốm vàng nhỏ (< 20% diện tích).   | 1000 ảnh               | Siêu thị nội địa       |
| G3       | Chín vàng      | Vỏ vàng đều (70-90%), có thể còn một ít màu xanh ở cuống. | 1000 ảnh               | Ăn ngay/Đóng hộp       |
| G4       | Chín mọng      | Vỏ vàng đậm hoặc cam, có đốm tàn nhang tự nhiên nhỏ.      | 1000 ảnh               | Chế biến sâu (mứt/sấy) |
| G5       | Hàng lỗi/Hỏng  | Vết thâm lớn, dập nát, nấm mốc hoặc biến dạng rõ rệt.     | 1000 ảnh               | Loại bỏ/Tái chế        |

---

## 2. Chiến lược thu thập dữ liệu (Dataset Strategy)

Để đạt được độ chính xác cao cho bài tập lớn, tổng quy mô Dataset mục tiêu là **5,000 ảnh** (1,000 ảnh cho mỗi cấp độ).