# XÂY DỰNG HỆ THỐNG NHẬN DIỆN CỬ CHỈ TAY HỖ TRỢ NGƯỜI KHUYẾT TẬT GIAO TIẾP

<p align="center">
  <img src="images/logoDaiNam.png" alt="DaiNam University Logo" width="200"/>
  <img src="LogoAIoTLab.png" alt="AIoTLab Logo" width="170"/>
</p>

[![Made by AIoTLab](https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Fit DNU](https://img.shields.io/badge/Fit%20DNU-green?style=for-the-badge)](https://fitdnu.net/)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge)](https://dainam.edu.vn)

## 🌟 Giới thiệu

Người khuyết tật về nghe và nói gặp nhiều khó khăn trong giao tiếp với cộng đồng. Ngôn ngữ ký hiệu là phương tiện giao tiếp chính của họ, nhưng không phải ai cũng hiểu được. Vì vậy, việc xây dựng một hệ thống nhận diện cử chỉ tay giúp chuyển đổi ngôn ngữ ký hiệu thành văn bản hoặc giọng nói là rất cần thiết.

Với sự phát triển của trí tuệ nhân tạo (AI) và thị giác máy tính, đề tài này tập trung nghiên cứu và ứng dụng các mô hình học sâu để nhận diện cử chỉ tay, giúp người khuyết tật giao tiếp thuận tiện hơn. Hệ thống không chỉ hỗ trợ cá nhân mà còn góp phần tạo ra một môi trường giao tiếp thân thiện, hòa nhập hơn cho cộng đồng.

## 🛠️ Chức năng chính

- **Chuẩn bị dữ liệu:** Tải bộ dữ liệu ASL Alphabet từ Kaggle, giải nén và sắp xếp dữ liệu theo từng lớp ký hiệu.
- **Huấn luyện mô hình:** Chạy chương trình để huấn luyện mô hình nhận diện cử chỉ tay. Theo dõi quá trình học qua biểu đồ và ma trận nhầm lẫn.
- **Nhận diện từ ảnh:** Cung cấp một ảnh chứa cử chỉ tay. Hệ thống hiển thị ảnh và dự đoán ký hiệu.
- **Nhận diện từ video:** Tải lên video có chứa cử chỉ tay. Hệ thống tách khung hình và dự đoán từng cử chỉ.
- **Chuyển đổi giọng nói:** Sau khi nhận diện, hệ thống phát âm thanh tương ứng với ký hiệu.
- **Lưu và sử dụng lại mô hình:** Mô hình được lưu lại để sử dụng sau mà không cần huấn luyện lại.

## 📦 Các thư viện Python cần thiết

Cài đặt các thư viện bằng lệnh sau:
```sh
pip install opencv-python tensorflow keras numpy pandas scikit-learn matplotlib seaborn plotly gtts
