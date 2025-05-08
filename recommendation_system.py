import psycopg2
import pandas as pd

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

# Lấy danh sách sản phẩm trong các danh mục có khuyến mãi
query_products = """
    SELECT p.product_id, p.name, p.category_id, p.price, p.brand
    FROM products p
    JOIN promotion_categories pc ON p.category_id = pc.category_id
    JOIN promotions promo ON pc.promotion_id = promo.promotion_id
    WHERE promo.start_date <= NOW() AND (promo.end_date IS NULL OR promo.end_date >= NOW())
"""
products_df = pd.read_sql(query_products, conn)

# Thêm cột giá sau giảm
products_df = products_df.merge(promo_df[['category_id', 'discount_type', 'discount_value']], on='category_id')
products_df['discounted_price'] = products_df.apply(
    lambda row: row['price'] * (1 - row['discount_value'] / 100) if row['discount_type'] == 'percentage' else row['price'] - row['discount_value'],
    axis=1
)
products_df['discount'] = products_df['price'] - products_df['discounted_price']

# Kiểm tra dữ liệu
print("Dữ liệu từ purchase_history:")
print(purchase_df.head())
print("\nDanh mục yêu thích của khách hàng:")
print(favorite_categories.head())
print("\nSản phẩm giảm giá:")
print(products_df.head())