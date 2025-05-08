import psycopg2

# Kết nối PostgreSQL
try:
    conn = psycopg2.connect(
        dbname="product-suggestions",
        user="postgres",
        password="20082024",
        host="localhost",
        port="5432"
    )
    print("Kết nối cơ sở dữ liệu thành công!")
    conn.close()
except Exception as e:
    print(f"Lỗi kết nối: {e}")

