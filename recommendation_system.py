import psycopg2
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import joblib
import os

# Kết nối PostgreSQL
conn = psycopg2.connect(
    dbname="product-suggestions",
    user="postgres",
    password="20082024",
    host="localhost",
    port="5432"
)

# Truy vấn dữ liệu từ purchase_history để xác định sở thích khách hàng
query_purchase = """
    SELECT ph.customer_id, ph.product_id, ph.quantity, ph.total_price, ph.purchased_at, p.category_id
    FROM purchase_history ph
    JOIN products p ON ph.product_id = p.product_id
"""
purchase_df = pd.read_sql(query_purchase, conn)

# Kiểm tra dữ liệu trống
if purchase_df.empty:
    print("Không có dữ liệu trong purchase_history. Vui lòng kiểm tra cơ sở dữ liệu.")
    conn.close()
    exit()

# Tính danh mục yêu thích dựa trên tổng quantity
favorite_categories = purchase_df.groupby(['customer_id', 'category_id'])['quantity'].sum().reset_index()
favorite_categories = favorite_categories.sort_values(['customer_id', 'quantity'], ascending=[True, False])
favorite_categories = favorite_categories.drop_duplicates(subset=['customer_id'], keep='first')

# Truy vấn dữ liệu khuyến mãi và sản phẩm giảm giá
query_promotions = """
    SELECT pc.promotion_id, pc.category_id, p.discount_type, p.discount_value, p.start_date, p.end_date
    FROM promotion_categories pc
    JOIN promotions p ON pc.promotion_id = p.promotion_id
    WHERE p.start_date <= NOW() AND (p.end_date IS NULL OR p.end_date >= NOW())
"""
promo_df = pd.read_sql(query_promotions, conn)

# Kiểm tra dữ liệu trống
if promo_df.empty:
    print("Không có chương trình khuyến mãi nào đang hoạt động.")
    conn.close()
    exit()

# Lấy danh sách sản phẩm trong các danh mục có khuyến mãi
query_products = """
    SELECT p.product_id, p.name, p.category_id, p.price, p.brand
    FROM products p
    JOIN promotion_categories pc ON p.category_id = pc.category_id
    JOIN promotions promo ON pc.promotion_id = promo.promotion_id
    WHERE promo.start_date <= NOW() AND (promo.end_date IS NULL OR promo.end_date >= NOW())
"""
products_df = pd.read_sql(query_products, conn)

# Kiểm tra dữ liệu trống
if products_df.empty:
    print("Không có sản phẩm nào trong danh mục được khuyến mãi.")
    conn.close()
    exit()

# Thêm cột giá sau giảm
products_df = products_df.merge(promo_df[['category_id', 'discount_type', 'discount_value']], on='category_id')
products_df['discounted_price'] = products_df.apply(
    lambda row: row['price'] * (1 - row['discount_value'] / 100) if row['discount_type'] == 'percentage' else row['price'] - row['discount_value'],
    axis=1
)
products_df['discount'] = products_df['price'] - products_df['discounted_price']

# Kiểm tra phân bố dữ liệu
print("Số lượng customer_id duy nhất:", purchase_df['customer_id'].nunique())
print("Số lượng product_id duy nhất:", purchase_df['product_id'].nunique())

# Chuẩn hóa rating về thang 0-5
max_rating = (purchase_df['total_price'] / purchase_df['quantity']).max()
purchase_df['rating'] = (purchase_df['total_price'] / purchase_df['quantity']) / max_rating * 5
purchase_df['rating'] = purchase_df['rating'].clip(0, 5)  # Giới hạn từ 0 đến 5

# In dữ liệu để kiểm tra
print("Kiểm tra dữ liệu rating trong purchase_df:")
print(purchase_df[['customer_id', 'product_id', 'total_price', 'quantity', 'rating']].head())
print("Thống kê rating:", purchase_df['rating'].describe())

# Chuẩn bị dữ liệu cho Surprise
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(purchase_df[['customer_id', 'product_id', 'rating']], reader)

# Chia tập train và test
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Huấn luyện hoặc tải mô hình SVD
model_file = "svd_model.pkl"
if os.path.exists(model_file):
    print("Tải mô hình đã huấn luyện từ file...")
    model = joblib.load(model_file)
else:
    print("Huấn luyện mô hình mới...")
    model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    joblib.dump(model, model_file)

# Đánh giá mô hình (RMSE - Root Mean Squared Error)
predictions = model.test(testset)
print("RMSE:", accuracy.rmse(predictions))

# Đóng kết nối
conn.close()