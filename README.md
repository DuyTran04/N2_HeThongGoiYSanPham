# NGHIÊN CỨU VÀ XÂY DỰNG HỆ THỐNG GỢI Ý SẢN PHẨM SỬ DỤNG THUẬT TOÁN DEEP MATRIX FACTORIZATION

## Thành Viên Nhóm
| STT | Họ và tên | MSSV | Lớp | Vai trò |
|-----|-----------|------|------|----------|
| 1 | Đặng Hoàng Phương Uyên | 2254810214 | 22ĐHTT05 | Nhóm Trưởng |
| 2 | Trần Đình Anh Duy | 2254810246 | 22ĐHTT05 | Thành Viên |
| 3 | Văn Hồng Quân | 2254810255 | 22ĐHTT06 | Thành Viên |

## Tính Năng Chính
- Phân tích và đề xuất sản phẩm dựa trên Deep Matrix Factorization
- Lọc sản phẩm theo danh mục
- Hiển thị chi tiết sản phẩm và đánh giá
- Gợi ý các sản phẩm tương tự
- Giao diện web trực quan, dễ sử dụng

## Công Nghệ Sử Dụng
- **Python**: Ngôn ngữ lập trình chính
- **PyTorch**: Framework deep learning
- **Streamlit**: Framework web interface
- **Pandas & NumPy**: Xử lý dữ liệu
- **Scikit-learn**: Tiền xử lý dữ liệu và đánh giá mô hình

## Cài Đặt và Chạy Dự Án
1. Tải xuống:
- Chọn Download Zip ở nút xanh lục có tên là <> Code

2. Cài đặt các thư viện cần thiết:
```bash
pip install torch tensorflow pandas numpy streamlit scikit-learn tqdm seaborn matplotlib jupyter
```

3. Train model:
- Di chuyển đến thư mục project (Ví dụ: D:\VAA 8\Final Project\N2_HeThongGoiYSanPham)
- Mở Command Prompt (cmd)
- Nhập lệnh để mở Jupyter Notebook:
```bash
jupyter notebook
```
- Mở và chọn file `Deep MF.ipynb` để train model

4. Chạy web application:
- Mở project trong Visual Studio Code
- Chọn Terminal -> New Terminal
- Nhập lệnh để chạy Streamlit:
```bash
streamlit run app.py
```

## Hướng Dẫn Sử Dụng
1. Truy cập web interface (mặc định tại `http://localhost:8501`)
2. Chọn danh mục sản phẩm từ dropdown menu
3. Chọn sản phẩm cụ thể để xem chi tiết
4. Nhấn nút "Get Recommendations" để nhận các đề xuất sản phẩm tương tự
