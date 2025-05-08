import psycopg2
from faker import Faker
import random
from datetime import datetime, timedelta

# Kết nối PostgreSQL với CSDL product-suggestions
conn = psycopg2.connect(
    dbname="product-suggestions",
    user="postgres",
    password="20082024",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# Khởi tạo Faker để tạo dữ liệu giả
fake = Faker()

# Hàm tạo ngày ngẫu nhiên với xử lý lỗi
def random_date(start, end):
    if end is None or start is None:
        return start if start else datetime.now()
    return start + timedelta(days=random.randint(0, int((end - start).days)))

start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 12, 31)  # Mở rộng đến cuối năm 2025

# 1. Thêm 10.000 khách hàng
customers = []
for i in range(1, 10001):  # Từ 1 đến 10.000
    name = fake.name()
    email = f"{name.lower().replace(' ', '.')}{i}@example.com"
    phone_number = fake.phone_number()[:20]
    gender = random.choice(['male', 'female', 'other'])
    birth_date = fake.date_of_birth(minimum_age=18, maximum_age=80)
    city = fake.city()
    country = 'Vietnam'
    customers.append((name, email, phone_number, gender, birth_date, city, country))
cur.executemany("""
    INSERT INTO customers (name, email, phone_number, gender, birth_date, city, country)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
""", customers)

# Lấy danh sách customer_id thực tế từ bảng customers
cur.execute("SELECT customer_id FROM customers")
customer_ids = [row[0] for row in cur.fetchall()]

# 2. Thêm 100 danh mục sản phẩm
categories = [(f"Category {i}", f"Description for Category {i}") for i in range(1, 101)]
cur.executemany("""
    INSERT INTO product_categories (name, description)
    VALUES (%s, %s)
""", categories)

# 3. Thêm 50.000 sản phẩm
cur.execute("SELECT category_id FROM product_categories")
category_ids = [row[0] for row in cur.fetchall()]
products = []
for i in range(1, 50001):  # Từ 1 đến 50.000
    name = f"{fake.word().capitalize()} {random.choice(['Pro', 'Plus', 'Lite', 'Max'])}"
    description = fake.sentence()
    sku = f"SKU{i}"
    category_id = random.choice(category_ids)
    brand = fake.company()
    price = round(random.uniform(5.99, 1999.99), 2)
    stock_quantity = random.randint(10, 500)
    products.append((name, description, sku, category_id, brand, price, stock_quantity))
cur.executemany("""
    INSERT INTO products (name, description, sku, category_id, brand, price, stock_quantity)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
""", products)

# Lấy danh sách product_id thực tế từ bảng products
cur.execute("SELECT product_id FROM products")
product_ids = [row[0] for row in cur.fetchall()]

# 4. Thêm 100 khu vực cửa hàng
locations = [(f"Aisle {i}", f"Section for Aisle {i}") for i in range(1, 101)]
cur.executemany("""
    INSERT INTO store_locations (name, description)
    VALUES (%s, %s)
""", locations)

# Lấy danh sách store_location_id thực tế từ bảng store_locations
cur.execute("SELECT location_id FROM store_locations")
store_location_ids = [row[0] for row in cur.fetchall()]

# 5. Thêm 200 chương trình khuyến mãi
promotions = []
for i in range(1, 201):
    name = f"Promotion {i}"
    description = f"Description for Promotion {i}"
    discount_type = random.choice(['percentage', 'fixed_amount'])
    discount_value = random.uniform(5.00, 20.00) if discount_type == 'percentage' else random.uniform(1.00, 10.00)
    promo_start_date = random_date(start_date, end_date)
    promo_end_date = random_date(promo_start_date, end_date) if random.choice([True, False]) else None
    promotions.append((name, description, discount_type, discount_value, promo_start_date, promo_end_date))
cur.executemany("""
    INSERT INTO promotions (name, description, discount_type, discount_value, start_date, end_date)
    VALUES (%s, %s, %s, %s, %s, %s)
""", promotions)

# Lấy danh sách promotion_id thực tế từ bảng promotions
cur.execute("SELECT promotion_id FROM promotions")
promotion_ids = [row[0] for row in cur.fetchall()]

# 6. Thêm dữ liệu cho promotion_products (20.000 bản ghi)
promotion_products = [(random.choice(promotion_ids), random.choice(product_ids)) for _ in range(20000)]
cur.executemany("""
    INSERT INTO promotion_products (promotion_id, product_id)
    VALUES (%s, %s)
    ON CONFLICT (promotion_id, product_id) DO NOTHING
""", promotion_products)

# 7. Thêm dữ liệu cho promotion_categories (5.000 bản ghi)
cur.execute("SELECT category_id FROM product_categories")
category_ids = [row[0] for row in cur.fetchall()]
promotion_categories = [(random.choice(promotion_ids), random.choice(category_ids)) for _ in range(5000)]
cur.executemany("""
    INSERT INTO promotion_categories (promotion_id, category_id)
    VALUES (%s, %s)
    ON CONFLICT (promotion_id, category_id) DO NOTHING
""", promotion_categories)

# 8. Thêm dữ liệu cho promotion_customers (10.000 bản ghi)
promotion_customers = [(random.choice(promotion_ids), random.choice(customer_ids)) for _ in range(10000)]
cur.executemany("""
    INSERT INTO promotion_customers (promotion_id, customer_id)
    VALUES (%s, %s)
    ON CONFLICT (promotion_id, customer_id) DO NOTHING
""", promotion_customers)

# 9. Thêm 500.000 đơn hàng
orders = []
for i in range(1, 500001):  # Từ 1 đến 500.000
    customer_id = random.choice(customer_ids)
    order_date = random_date(start_date, end_date)
    total_amount = round(random.uniform(10.00, 5000.00), 2)
    order_status = random.choice(['pending', 'completed', 'shipped', 'processing'])
    shipping_address = f"{fake.building_number()} {fake.street_name()}, {fake.city()}, Vietnam"
    billing_address = f"{fake.building_number()} {fake.street_name()}, {fake.city()}, Vietnam"
    payment_method = random.choice(['credit_card', 'paypal', 'cash', 'debit_card'])
    orders.append((customer_id, order_date, total_amount, order_status, shipping_address, billing_address, payment_method))
cur.executemany("""
    INSERT INTO orders (customer_id, order_date, total_amount, order_status, shipping_address, billing_address, payment_method)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
""", orders)

# Lấy danh sách order_id thực tế từ bảng orders
cur.execute("SELECT order_id FROM orders")
order_ids = [row[0] for row in cur.fetchall()]

# 10. Thêm 1.000.000 mục đơn hàng
order_items = []
for _ in range(1000000):
    order_id = random.choice(order_ids)
    product_id = random.choice(product_ids)
    quantity = random.randint(1, 10)
    price = random.uniform(5.99, 1999.99)
    discount_amount = random.uniform(0.00, 200.00) if random.choice([True, False]) else 0.00
    promotion_id = random.choice(promotion_ids + [None] * 5)
    order_items.append((order_id, product_id, quantity, price, discount_amount, promotion_id))
cur.executemany("""
    INSERT INTO order_items (order_id, product_id, quantity, price, discount_amount, promotion_id)
    VALUES (%s, %s, %s, %s, %s, %s)
""", order_items)

# 11. Thêm 500.000 bản ghi lịch sử mua sắm
purchase_history = []
for _ in range(500000):
    customer_id = random.choice(customer_ids)
    product_id = random.choice(product_ids)
    quantity = random.randint(1, 5)
    total_price = round(quantity * random.uniform(5.99, 1999.99), 2)
    purchased_at = random_date(start_date, end_date)
    purchase_history.append((customer_id, product_id, quantity, total_price, purchased_at))
cur.executemany("""
    INSERT INTO purchase_history (customer_id, product_id, quantity, total_price, purchased_at)
    VALUES (%s, %s, %s, %s, %s)
""", purchase_history)

# 12. Thêm 200.000 mục giỏ hàng
shopping_cart_items = []
for _ in range(200000):
    customer_id = random.choice(customer_ids)
    product_id = random.choice(product_ids)
    quantity = random.randint(1, 5)
    reserved_quantity = quantity
    promotion_id = random.choice(promotion_ids + [None] * 5)
    store_location_id = random.choice(store_location_ids)
    shopping_cart_items.append((customer_id, product_id, quantity, reserved_quantity, promotion_id, store_location_id))
cur.executemany("""
    INSERT INTO shopping_cart_items (customer_id, product_id, quantity, reserved_quantity, promotion_id, store_location_id)
    VALUES (%s, %s, %s, %s, %s, %s)
""", shopping_cart_items)

# 13. Thêm 1.000.000 bản ghi lịch sử giá
product_price_history = []
for _ in range(1000000):
    product_id = random.choice(product_ids)
    price = random.uniform(5.99, 1999.99)
    price_start_date = random_date(start_date, end_date)
    price_end_date = random_date(price_start_date, end_date) if random.choice([True, False]) else None
    product_price_history.append((product_id, price, price_start_date, price_end_date))
cur.executemany("""
    INSERT INTO product_price_history (product_id, price, start_date, end_date)
    VALUES (%s, %s, %s, %s)
""", product_price_history)

# Commit và đóng kết nối
conn.commit()
cur.close()
conn.close()
print("Dữ liệu đã được chèn thành công!")