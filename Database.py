import pandas as pd
import sqlite3
import re
# Bước 1: Đọc dữ liệu từ file CSV
df = pd.read_csv(r"C:\Users\DELL\Downloads\N2_HeThongGoiYSanPham-main\N2_HeThongGoiYSanPham-main\clean_amazon.csv")

# Bước 2: Khởi tạo danh sách chứa dữ liệu review
review_data = []

# Bước 3: Tách dữ liệu vào bảng products và reviews
for idx, row in df.iterrows():
    product_id = row["product_id"]
    
    # Chuyển đổi các giá trị thành chuỗi và tách theo dấu phẩy
    review_ids = str(row["review_id"]).split(",") if pd.notna(row["review_id"]) else []
    user_ids = str(row["user_id"]).split(",") if pd.notna(row["user_id"]) else []
    user_names = str(row["user_name"]).split(",") if pd.notna(row["user_name"]) else []
    review_titles = str(row["review_title"]).split(",") if pd.notna(row["review_title"]) else []
    review_contents = str(row["review_content"]).split(",") if pd.notna(row["review_content"]) else []
    
    # Xác định số lượng review (dựa vào số lượng nhỏ nhất)
    n = min(len(review_ids), len(user_ids), len(user_names), len(review_titles), len(review_contents))
    
    for i in range(n):
        review_data.append({
            "review_id": review_ids[i].strip(),
            "product_id": product_id,
            "user_id": user_ids[i].strip(),
            "user_name": user_names[i].strip(),
            "review_title": review_titles[i].strip(),
            "review_content": review_contents[i].strip()
        })

# Chuyển review_data thành DataFrame
df_reviews = pd.DataFrame(review_data)

# Lọc ra bảng sản phẩm (loại bỏ các cột liên quan đến review)
df_products = df.drop(columns=["user_id", "user_name", "review_id", "review_title", "review_content"])
df_products = df_products.drop_duplicates()
df_reviews = df_reviews.drop_duplicates()

print("Tổng số bản ghi sau khi loại bỏ trùng:")
print(f"Products: {len(df_products)}")
print(f"Reviews: {len(df_reviews)}")

# Bước 4: Tạo database và các bảng trong SQLite
conn = sqlite3.connect("database.db")
cursor = conn.cursor()
def generate_search_keywords(row):
    """Hàm tạo từ khóa tìm kiếm từ tên sản phẩm, danh mục, mô tả."""
    text = f"{row['product_name']} {row['category']} {row['about_product']}"
    text = text.lower()  # Chuyển về chữ thường
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Loại bỏ ký tự đặc biệt
    text = " ".join(set(text.split()))  # Loại bỏ từ trùng lặp
    return text

df_products["search_product"] = df_products.apply(generate_search_keywords, axis=1)
# Tạo bảng products
cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        product_id TEXT PRIMARY KEY,
        product_name TEXT,
        category TEXT,
        discounted_price REAL,
        actual_price REAL,
        discount_percentage REAL,
        rating REAL,
        rating_count INTEGER,
        about_product TEXT,
        img_link TEXT,
        product_link TEXT,
        category_encoded INTEGER,
        discount_percentage_calculated REAL,
        purchase_count_estimated INTEGER
        search_product TEXT
    )
''')

# Tạo bảng reviews
cursor.execute('''
    CREATE TABLE IF NOT EXISTS reviews (
        review_id TEXT PRIMARY KEY,
        product_id TEXT,
        user_id TEXT,
        user_name TEXT,
        review_title TEXT,
        review_content TEXT,
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    )
''')

conn.commit()

# Bước 5: Lưu dữ liệu vào SQLite
df_products.to_sql("products", conn, if_exists="replace", index=False)
df_reviews.to_sql("reviews", conn, if_exists="replace", index=False)

conn.commit()
conn.close()
print("✅ Dữ liệu đã được tách thành 2 bảng: 'products' và 'reviews' trong database.db thành công!")
