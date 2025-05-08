import psycopg2
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import joblib
import os
from flask import Flask, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Cấu hình
use_subsampling = True
subsample_size = 100000  # Kích thước mẫu cho subsampling

# Hàm tải dữ liệu
def load_data(conn):
    query_purchase = """
        SELECT ph.customer_id, ph.product_id, ph.quantity, ph.total_price, ph.purchased_at, p.category_id
        FROM purchase_history ph
        JOIN products p ON ph.product_id = p.product_id
    """
    purchase_df = pd.read_sql(query_purchase, conn)

    query_loyalty = """
        SELECT customer_id, loyalty_score
        FROM customers
    """
    loyalty_df = pd.read_sql(query_loyalty, conn)
    purchase_df = purchase_df.merge(loyalty_df, on='customer_id', how='left')
    purchase_df['loyalty_score'] = purchase_df['loyalty_score'].fillna(0)

    query_promotions = """
        SELECT pc.promotion_id, pc.category_id, p.discount_type, p.discount_value, p.start_date, p.end_date
        FROM promotion_categories pc
        JOIN promotions p ON pc.promotion_id = p.promotion_id
        WHERE p.start_date <= NOW() AND (p.end_date IS NULL OR p.end_date >= NOW())
    """
    promo_df = pd.read_sql(query_promotions, conn)

    query_products = """
        SELECT p.product_id, p.name, p.category_id, p.price, p.brand
        FROM products p
        JOIN promotion_categories pc ON p.category_id = pc.category_id
        JOIN promotions promo ON pc.promotion_id = promo.promotion_id
        WHERE promo.start_date <= NOW() AND (promo.end_date IS NULL OR promo.end_date >= NOW())
    """
    products_df = pd.read_sql(query_products, conn)

    return purchase_df, promo_df, products_df

# Hàm làm sạch dữ liệu
def clean_data(products_df):
    popular_brands = products_df['brand'].value_counts()[products_df['brand'].value_counts() > 1000].index
    products_df = products_df[products_df['brand'].isin(popular_brands)]
    return products_df

# Hàm chuẩn bị đặc trưng
def prepare_features(products_df, use_subsampling, subsample_size):
    if use_subsampling:
        products_sample = products_df.sample(n=subsample_size, random_state=42)
        print("Số lượng hàng trong products_sample:", len(products_sample))
        brand_encoded = pd.get_dummies(products_sample['brand'], prefix='brand')
        features = pd.concat([products_sample[['price', 'discount']], brand_encoded], axis=1).fillna(0)
    else:
        brand_encoded = pd.get_dummies(products_df['brand'], prefix='brand')
        features = pd.concat([products_df[['price', 'discount']], brand_encoded], axis=1).fillna(0)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    similarity_matrix = cosine_similarity(features_scaled)
    product_id_to_index = {pid: idx for idx, pid in enumerate(products_sample['product_id'] if use_subsampling else products_df['product_id'])}

    return features_scaled, similarity_matrix, product_id_to_index

# Hàm huấn luyện mô hình
def train_model(data):
    model_file = "svd_model.pkl"
    if os.path.exists(model_file):
        print("Tải mô hình đã huấn luyện từ file...")
        model = joblib.load(model_file)
    else:
        print("Huấn luyện mô hình mới...")
        model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        model.fit(trainset)
        joblib.dump(model, model_file)
        predictions = model.test(testset)
        print("RMSE:", accuracy.rmse(predictions))
    return model

# Hàm gợi ý sản phẩm
def get_discounted_recommendations(customer_id, products_df, favorite_categories, model, product_id_to_index, use_subsampling, cf_weight=0.7, cb_weight=0.3):
    fav_category = favorite_categories[favorite_categories['customer_id'] == customer_id]['category_id'].values
    if len(fav_category) == 0:
        print(f"Không tìm thấy danh mục yêu thích cho khách hàng {customer_id}.")
        return pd.DataFrame()
    fav_category = fav_category[0]
    
    purchased_products = purchase_df[purchase_df['customer_id'] == customer_id]['product_id'].unique()
    
    discounted_products = products_df[
        (products_df['category_id'] == fav_category) & 
        (~products_df['product_id'].isin(purchased_products))
    ].copy()
    
    if discounted_products.empty:
        print(f"Không có sản phẩm giảm giá trong danh mục {fav_category} cho khách hàng {customer_id}. Mở rộng sang danh mục khác.")
        related_categories = purchase_df[purchase_df['customer_id'] == customer_id]['category_id'].unique()
        discounted_products = products_df[
            (products_df['category_id'].isin(related_categories)) & 
            (~products_df['product_id'].isin(purchased_products))
        ].copy()
    
    loyalty_score = purchase_df[purchase_df['customer_id'] == customer_id]['loyalty_score'].values[0]
    
    recommendations = []
    for _, row in discounted_products.iterrows():
        product_id = row['product_id']
        pred = model.predict(customer_id, product_id)
        cf_score = pred.est * (1 + loyalty_score / 1000)
        
        cb_score = 0
        if len(purchased_products) > 0 and use_subsampling:
            product_idx = product_id_to_index.get(product_id)
            if product_idx is not None:
                similarities = []
                for purchased_id in purchased_products:
                    purchased_idx = product_id_to_index.get(purchased_id)
                    if purchased_idx is not None:
                        sim = similarity_matrix[product_idx, purchased_idx]
                        similarities.append(sim)
                cb_score = max(similarities) if similarities else 0
        
        combined_score = cf_weight * cf_score + cb_weight * cb_score
        recommendations.append((product_id, combined_score, row['discounted_price'], row['discount']))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recs = recommendations[:5]
    
    result = []
    for product_id, combined_score, discounted_price, discount in top_recs:
        product_info = products_df[products_df['product_id'] == product_id].iloc[0]
        result.append({
            'product_id': product_id,
            'name': product_info['name'],
            'category_id': product_info['category_id'],
            'brand': product_info['brand'],
            'price': product_info['price'],
            'discounted_price': discounted_price,
            'discount': discount,
            'combined_score': combined_score
        })
    return pd.DataFrame(result)

# Hàm chính
def main():
    # Kết nối và tải dữ liệu
    conn = psycopg2.connect(
        dbname="product-suggestions",
        user="postgres",
        password="20082024",
        host="localhost",
        port="5432"
    )
    purchase_df, promo_df, products_df = load_data(conn)

    # Làm sạch dữ liệu
    products_df = clean_data(products_df)
    products_df = products_df.merge(promo_df[['category_id', 'discount_type', 'discount_value']], on='category_id')
    products_df['discounted_price'] = products_df.apply(
        lambda row: row['price'] * (1 - row['discount_value'] / 100) if row['discount_type'] == 'percentage' else row['price'] - row['discount_value'],
        axis=1
    )
    products_df['discount'] = products_df['price'] - products_df['discounted_price']

    # Chuẩn bị đặc trưng
    features_scaled, similarity_matrix, product_id_to_index = prepare_features(products_df, use_subsampling, subsample_size)

    # Tính danh mục yêu thích
    favorite_categories = purchase_df.groupby(['customer_id', 'category_id'])['quantity'].sum().reset_index()
    favorite_categories = favorite_categories.sort_values(['customer_id', 'quantity'], ascending=[True, False])
    favorite_categories = favorite_categories.drop_duplicates(subset=['customer_id'], keep='first')

    # Chuẩn hóa rating
    max_rating = (purchase_df['total_price'] / purchase_df['quantity']).max()
    purchase_df['rating'] = (purchase_df['total_price'] / purchase_df['quantity']) / max_rating * 5
    purchase_df['rating'] = purchase_df['rating'].clip(0, 5)

    # In kiểm tra
    print("Số lượng giá trị duy nhất trong cột 'brand' sau khi làm sạch:", products_df['brand'].nunique())
    print("Số lượng hàng trong products_df sau khi làm sạch:", len(products_df))
    print("Số lượng customer_id duy nhất:", purchase_df['customer_id'].nunique())
    print("Số lượng product_id duy nhất:", purchase_df['product_id'].nunique())
    print("Kiểm tra loyalty_score trong purchase_df:")
    print(purchase_df[['customer_id', 'loyalty_score']].drop_duplicates().head())
    print("Kiểm tra dữ liệu rating trong purchase_df:")
    print(purchase_df[['customer_id', 'product_id', 'total_price', 'quantity', 'rating']].head())
    print("Thống kê rating:", purchase_df['rating'].describe())

    # Chuẩn bị dữ liệu cho Surprise
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(purchase_df[['customer_id', 'product_id', 'rating']], reader)

    # Huấn luyện mô hình
    model = train_model(data)

    # Đánh giá mô hình
    print("Đánh giá mô hình với cross-validation...")
    cross_validate(model, data, measures=['RMSE'], cv=5, verbose=True)

    # Khởi tạo Flask
    app = Flask(__name__)

    @app.route('/recommend/<int:customer_id>', methods=['GET'])
    def recommend(customer_id):
        try:
            recommendations = get_discounted_recommendations(customer_id, products_df, favorite_categories, model, product_id_to_index, use_subsampling)
            if recommendations.empty:
                return jsonify({'error': 'Không tìm thấy gợi ý cho khách hàng này'}), 404
            return jsonify(recommendations.to_dict(orient='records'))
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Chạy ứng dụng
    app.run(debug=True, host='0.0.0.0', port=5000)
    conn.close()

if __name__ == '__main__':
    main()