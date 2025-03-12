import random
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from utils.data_processing import process_and_save_data

app = Flask(__name__)
app.secret_key = 'amazon_recommendation_secret_key'  # For session management

# Create app_data directory if it doesn't exist
if not os.path.exists('app_data'):
    os.makedirs('app_data')

# Kết nối MongoDB
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['ecommerce_db']
    products_collection = db['products']
    print("Successfully connected to MongoDB")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    # Fallback to not using MongoDB
    products_collection = None

# Global variables
model = None
metadata = None
df = None
product_encoder = None
idx_to_product = None
product_to_name = None
rating_scaler = None
product_categories = None
user_embeddings = None
product_embeddings = None

def load_model_and_data():
    """Load the trained model, metadata, and product data"""
    global model, metadata, df, product_encoder, idx_to_product, product_to_name, rating_scaler, product_categories
    global user_embeddings, product_embeddings
    
    result = {}
    print("Loading model and data...")
    
    # Load the model
    try:
        model = tf.keras.models.load_model('amazon_recommender_model.keras')
        print("Model loaded successfully from amazon_recommender_model.keras")
        result['model'] = model
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        result['model'] = None
    
    # Load metadata
    try:
        with open('amazon_recommender_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        print("Metadata loaded successfully from amazon_recommender_metadata.pkl")
        
        result['metadata'] = metadata
        
        if metadata is not None:
            # Extract necessary components from metadata
            product_encoder = metadata.get('product_encoder')
            user_encoder = metadata.get('user_encoder', None)
            idx_to_product = metadata.get('idx_to_product')
            product_to_name = metadata.get('product_to_name')
            rating_scaler = metadata.get('rating_scaler')
            
            # Lấy embeddings từ metadata nếu có
            user_embeddings = metadata.get('user_embeddings')
            product_embeddings = metadata.get('product_embeddings')
            
            result['product_encoder'] = product_encoder
            result['user_encoder'] = user_encoder 
            result['idx_to_product'] = idx_to_product
            result['product_to_name'] = product_to_name
            result['rating_scaler'] = rating_scaler
            result['user_embeddings'] = user_embeddings
            result['product_embeddings'] = product_embeddings
    except Exception as e:
        print(f"Error loading metadata: {e}")
        metadata = None
        result['metadata'] = None
    
    # Load product data
    try:
        # Tải từ file CSV
        if os.path.exists('amazon.csv'):
            df = pd.read_csv('amazon.csv')
            print(f"Data loaded successfully from amazon.csv: {len(df)} products")
        elif os.path.exists('amazon_cleaned.csv'):
            df = pd.read_csv('amazon_cleaned.csv')
            print(f"Data loaded successfully from amazon_cleaned.csv: {len(df)} products")
        else:
            print("CSV data file not found")
            df = None
            
        result['df'] = df
        
        # Lưu dữ liệu vào MongoDB nếu có kết nối
        if df is not None and products_collection is not None:
            try:
                # Chỉ lưu vào MongoDB nếu collection rỗng
                if products_collection.count_documents({}) == 0:
                    print("Saving data to MongoDB...")
                    process_and_save_data('amazon.csv')
                    print("Data saved to MongoDB successfully")
            except Exception as e:
                print(f"Error saving data to MongoDB: {e}")
        
        # Create product categories for navigation
        if df is not None and 'category' in df.columns:
            product_categories = df['category'].dropna().unique().tolist()  # Convert to list
            result['product_categories'] = product_categories
        else:
            product_categories = []
            result['product_categories'] = []
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        df = None
        result['df'] = None
        result['product_categories'] = []
    
    return result

# Hàm lấy Top Products từ MongoDB
def get_top_products(limit=5):
    """
    Lấy top sản phẩm dựa trên popularity từ MongoDB.
    
    Args:
        limit (int): Số lượng sản phẩm tối đa cần lấy (mặc định là 5).
    
    Returns:
        list: Danh sách sản phẩm đã được sắp xếp theo popularity.
    """
    # Kiểm tra nếu không có kết nối MongoDB, sử dụng DataFrame
    if products_collection is None:
        return get_popular_products(limit)
    
    try:
        top_products = products_collection.find().sort('popularity', -1).limit(limit)
        return [
            {
                'product_id': product['product_id'],
                'name': product['product_name'],
                'image_url': product.get('img_link', '/static/img/placeholder.jpg'),
                'discounted_price': product.get('discounted_price', 'N/A'),
                'price': product.get('actual_price', 'N/A'),
                'rating': product.get('rating', 'N/A'),
                'popularity': product.get('popularity', 0)
            }
            for product in top_products
        ]
    except Exception as e:
        print(f"Error getting top products from MongoDB: {e}")
        return get_popular_products(limit)

# Get product details by product_id
def get_product_details(product_id):
    """Get product details from the dataframe or MongoDB"""
    # Thử lấy sản phẩm từ MongoDB trước
    if products_collection is not None:
        try:
            product = products_collection.find_one({'product_id': product_id})
            if product:
                # Chuyển đổi từ MongoDB document sang dictionary
                details = {
                    'product_id': product['product_id'],
                    'name': product['product_name'],
                    'category': product.get('category', 'Uncategorized'),
                    'price': product.get('actual_price', 'N/A'),
                    'discounted_price': product.get('discounted_price', 'N/A'),
                    'discount_percentage': product.get('discount_percentage', '0%'),
                    'rating': product.get('rating', 'N/A'),
                    'rating_count': product.get('rating_count', '0'),
                    'about': product.get('about_product', 'No description available'),
                    'image_url': product.get('img_link', '/static/img/placeholder.jpg')
                }
                return details
        except Exception as e:
            print(f"Error getting product from MongoDB: {e}")
    
    # Nếu không có MongoDB hoặc không tìm thấy sản phẩm, thử lấy từ DataFrame
    if df is None:
        return None
    
    product_data = df[df['product_id'] == product_id]
    if product_data.empty:
        return None
    
    # Get the first matching product
    product = product_data.iloc[0]
    
    # Create a dictionary with product details
    details = {
        'product_id': product['product_id'],
        'name': product['product_name'],
        'category': product.get('category', 'Uncategorized'),
        'price': product.get('actual_price', 'N/A'),
        'discounted_price': product.get('discounted_price', 'N/A'),
        'discount_percentage': product.get('discount_percentage', '0%'),
        'rating': product.get('rating', 'N/A'),
        'rating_count': product.get('rating_count', '0'),
        'about': product.get('about_product', 'No description available'),
        'image_url': product.get('img_link', '/static/img/placeholder.jpg')
    }
    
    return details

# Get popular products (highest rated)
def get_popular_products(limit=10):
    """Get popular products based on ratings"""
    if df is None:
        return []
    
    # Sort by rating and rating_count
    try:
        # Convert rating to numeric
        df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Filter products with at least some ratings
        popular = df[pd.to_numeric(df['rating_count'], errors='coerce') > 10]
        
        # Sort by rating
        popular = popular.sort_values(by='rating_numeric', ascending=False)
        
        # Get top products
        top_products = []
        for _, product in popular.head(limit).iterrows():
            top_products.append({
                'product_id': product['product_id'],
                'name': product['product_name'],
                'price': product.get('actual_price', 'N/A'),
                'discounted_price': product.get('discounted_price', 'N/A'),
                'rating': product.get('rating', 'N/A'),
                'image_url': product.get('img_link', '/static/img/placeholder.jpg')
            })
        
        return top_products
    except Exception as e:
        print(f"Error getting popular products: {e}")
        return []

# Get products by category
def get_products_by_category(category, limit=20):
    """Get products by category"""
    # Thử lấy từ MongoDB trước
    if products_collection is not None:
        try:
            category_products_cursor = products_collection.find({'category': category}).limit(limit)
            products = []
            for product in category_products_cursor:
                products.append({
                    'product_id': product['product_id'],
                    'name': product['product_name'],
                    'price': product.get('actual_price', 'N/A'),
                    'discounted_price': product.get('discounted_price', 'N/A'),
                    'rating': product.get('rating', 'N/A'),
                    'image_url': product.get('img_link', '/static/img/placeholder.jpg')
                })
            if products:
                return products
        except Exception as e:
            print(f"Error getting products by category from MongoDB: {e}")
    
    # Nếu không có MongoDB hoặc không tìm thấy sản phẩm, thử lấy từ DataFrame
    if df is None:
        return []
    
    # Filter by category
    category_products = df[df['category'] == category]
    
    # Get product details
    products = []
    for _, product in category_products.head(limit).iterrows():
        products.append({
            'product_id': product['product_id'],
            'name': product['product_name'],
            'price': product.get('actual_price', 'N/A'),
            'discounted_price': product.get('discounted_price', 'N/A'),
            'rating': product.get('rating', 'N/A'),
            'image_url': product.get('img_link', '/static/img/placeholder.jpg')
        })
    
    return products

# Get recommended products based on shopping behavior
def get_recommended_products(cart_items, limit=5):
    """Get product recommendations based on items in cart"""
    if model is None or metadata is None or df is None:
        # Fallback to popular products if model is not available
        return get_popular_products(limit)
    
    try:
        # Kiểm tra xem có sẵn embeddings trong metadata không
        if product_embeddings is not None:
            # Sử dụng embeddings có sẵn
            embeddings_to_use = product_embeddings
        else:
            # Nếu không có embeddings sẵn, tính từ model
            product_embedding_layer = model.get_layer('product_embedding')
            embeddings_to_use = product_embedding_layer.get_weights()[0]
        
        # Get product indices for cart items
        cart_product_indices = []
        for product_id in cart_items:
            try:
                # First, check if we can encode directly
                product_idx = product_encoder.transform([product_id])[0]
                if product_idx < len(embeddings_to_use):
                    cart_product_indices.append(product_idx)
            except:
                # If not, find it in the dataframe
                product_data = df[df['product_id'] == product_id]
                if not product_data.empty:
                    # Try to use product_idx if available in DataFrame
                    if 'product_idx' in product_data.columns:
                        product_idx = int(product_data.iloc[0].get('product_idx', -1))
                        if product_idx >= 0 and product_idx < len(embeddings_to_use):
                            cart_product_indices.append(product_idx)
        
        if not cart_product_indices:
            return get_popular_products(limit)
        
        # Create virtual user embedding as average of cart product embeddings
        virtual_user_embedding = np.mean([embeddings_to_use[idx] for idx in cart_product_indices], axis=0)
        
        # Compute similarity with all products
        similarities = []
        for idx, product_embedding in enumerate(embeddings_to_use):
            # Skip products already in cart
            if idx in cart_product_indices:
                continue
                
            # Compute cosine similarity
            similarity = np.dot(virtual_user_embedding, product_embedding) / (
                np.linalg.norm(virtual_user_embedding) * np.linalg.norm(product_embedding) + 1e-8
            )
            
            similarities.append((idx, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        recommendations = []
        for idx, _ in similarities[:limit]:
            # Convert index to product_id
            try:
                product_id = idx_to_product.get(idx) or product_encoder.inverse_transform([idx])[0]
                product_details = get_product_details(product_id)
                if product_details:
                    recommendations.append(product_details)
            except Exception as e:
                print(f"Error getting product details for index {idx}: {e}")
                continue
        
        return recommendations
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return get_popular_products(limit)

# Search products
def search_products(query, limit=20):
    """Search products by name"""
    # Thử tìm kiếm trong MongoDB trước
    if products_collection is not None:
        try:
            # Tạo truy vấn tìm kiếm với MongoDB
            search_query = {'product_name': {'$regex': query, '$options': 'i'}}
            search_results = products_collection.find(search_query).limit(limit)
            
            results = []
            for product in search_results:
                results.append({
                    'product_id': product['product_id'],
                    'name': product['product_name'],
                    'price': product.get('actual_price', 'N/A'),
                    'discounted_price': product.get('discounted_price', 'N/A'),
                    'rating': product.get('rating', 'N/A'),
                    'image_url': product.get('img_link', '/static/img/placeholder.jpg')
                })
            
            if results:
                return results
        except Exception as e:
            print(f"Error searching products in MongoDB: {e}")
    
    # Nếu không có MongoDB hoặc không tìm thấy kết quả, thử tìm trong DataFrame
    if df is None:
        return []
    
    # Convert query to lowercase for case-insensitive search
    query = query.lower()
    
    # Filter products by name
    matching_products = df[df['product_name'].str.lower().str.contains(query, na=False)]
    
    # Get product details
    results = []
    for _, product in matching_products.head(limit).iterrows():
        results.append({
            'product_id': product['product_id'],
            'name': product['product_name'],
            'price': product.get('actual_price', 'N/A'),
            'discounted_price': product.get('discounted_price', 'N/A'),
            'rating': product.get('rating', 'N/A'),
            'image_url': product.get('img_link', '/static/img/placeholder.jpg')
        })
    
    return results

# Xác thực người dùng (đơn giản hóa, nên thay thế bằng mô-đun từ data.database)
def signup_user(username, email, password):
    """Simple user registration function"""
    # Trong thực tế, bạn nên kết nối với cơ sở dữ liệu người dùng
    try:
        # Kiểm tra email đã tồn tại
        if db.users.find_one({'email': email}):
            return False, "Email already exists"
        
        # Mã hóa mật khẩu
        hashed_password = generate_password_hash(password)
        
        # Tạo người dùng mới
        user = {
            'username': username,
            'email': email,
            'password': hashed_password
        }
        
        # Lưu vào cơ sở dữ liệu
        db.users.insert_one(user)
        
        return True, "Sign up successful"
    except Exception as e:
        print(f"Error during signup: {e}")
        return False, "An error occurred during signup"

def login_user(email, password):
    """Simple user login function"""
    try:
        # Tìm người dùng theo email
        user = db.users.find_one({'email': email})
        
        if not user:
            return False, "Invalid email or password"
        
        # Kiểm tra mật khẩu
        if check_password_hash(user['password'], password):
            return True, "Login successful"
        
        return False, "Invalid email or password"
    except Exception as e:
        print(f"Error during login: {e}")
        return False, "An error occurred during login"

# Routes
@app.route('/')
def home():
    """Home page with personalized recommendations"""
    # Lấy top 5 sản phẩm phổ biến
    top_products = get_top_products(limit=5)
    
    # Lấy sản phẩm phổ biến
    popular_products = get_popular_products(8)
    
    # Lấy danh mục nổi bật
    if product_categories is not None and len(product_categories) > 0:
        featured_category = random.choice(product_categories)
        featured_products = get_products_by_category(featured_category, 4)
    else:
        featured_category = "Featured Products"
        featured_products = popular_products[:4]
    
    # Lấy sản phẩm đã xem gần đây
    recently_viewed = get_recently_viewed_products(4)
    
    # Lấy đề xuất dựa trên lịch sử xem
    history_recommendations = []
    if recently_viewed:  # Chỉ hiển thị nếu có sản phẩm đã xem
        history_recommendations = get_recommendations_from_history(4)
    
    # Lấy sản phẩm từ danh mục ưa thích nếu người dùng đã đăng nhập
    preferred_category, preferred_products = get_preferred_category_products(4)
    
    # Get cart for recommendations
    cart = session.get('cart', {})
    cart_recommendations = []
    if cart:  # Chỉ hiển thị nếu giỏ hàng có sản phẩm
        cart_recommendations = get_recommended_products(list(cart.keys()), 4)
    
    # Get current user (if logged in)
    current_user = session.get('user')
    
    return render_template('home.html', 
                          top_products=top_products,
                          popular_products=popular_products,
                          featured_category=featured_category,
                          featured_products=featured_products,
                          recently_viewed=recently_viewed,
                          history_recommendations=history_recommendations,
                          preferred_category=preferred_category,
                          preferred_products=preferred_products,
                          cart_recommendations=cart_recommendations,
                          categories=product_categories,
                          current_user=current_user)
# Hàm đồng bộ lịch sử xem từ session vào MongoDB khi đăng nhập
def sync_viewed_history(user_email):
    """Đồng bộ lịch sử xem từ session vào MongoDB khi đăng nhập"""
    viewed_history = session.get('viewed_history', [])
    
    if not viewed_history or db is None:
        return
    
    try:
        # Lấy lịch sử hiện có từ MongoDB
        user_behavior = db.user_behaviors.find_one({'email': user_email})
        
        # Khởi tạo người dùng mới trong collection user_behaviors nếu chưa tồn tại
        if not user_behavior:
            db.user_behaviors.insert_one({
                'email': user_email,
                'viewed_products': [],
                'category_views': {},
                'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Lấy ID của các sản phẩm đã xem trong MongoDB
        existing_product_ids = set()
        if user_behavior and 'viewed_products' in user_behavior:
            existing_product_ids = {item['product_id'] for item in user_behavior['viewed_products']}
        
        # Cập nhật thông tin category_views
        category_updates = {}
        
        # Đồng bộ từng sản phẩm từ session
        for item in viewed_history:
            product_id = item['product_id']
            product = get_product_details(product_id)
            
            if not product:
                continue
                
            # Cập nhật đếm xem danh mục
            category = product.get('category', 'Uncategorized')
            if category not in category_updates:
                category_updates[category] = 0
            category_updates[category] += 1
            
            # Thêm vào viewed_products nếu chưa tồn tại
            if product_id not in existing_product_ids:
                db.user_behaviors.update_one(
                    {'email': user_email},
                    {
                        '$push': {
                            'viewed_products': {
                                'product_id': product_id,
                                'product_name': product['name'],
                                'category': category,
                                'timestamp': item['timestamp']
                            }
                        }
                    }
                )
        
        # Cập nhật category_views
        for category, count in category_updates.items():
            db.user_behaviors.update_one(
                {'email': user_email},
                {'$inc': {f'category_views.{category}': count}}
            )
    except Exception as e:
        print(f"Error syncing viewed history: {e}")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup page"""
    if request.method == 'GET':
        return render_template('auth/signup.html', categories=product_categories)
    elif request.method == 'POST':
        # Kiểm tra xem request là AJAX hay form submit
        if request.is_json:
            data = request.get_json()
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
        else:
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')

        success, message = signup_user(username, email, password)
        
        if success:
            flash('Sign up successful! Please login.', 'success')
            
            # Nếu là AJAX, trả về JSON
            if request.is_json:
                return jsonify({'message': message, 'success': True}), 200
            # Nếu là form submit thông thường, chuyển hướng
            return redirect(url_for('login'))
        
        flash(message, 'danger')
        
        # Nếu là AJAX, trả về JSON
        if request.is_json:
            return jsonify({'message': message, 'success': False}), 400
        # Nếu là form submit thông thường, hiển thị lại form
        return render_template('auth/signup.html', categories=product_categories, error=message)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'GET':
        return render_template('auth/login.html', categories=product_categories)
    elif request.method == 'POST':
        # Kiểm tra xem request là AJAX hay form submit
        if request.is_json:
            data = request.get_json()
            email = data.get('email')
            password = data.get('password')
        else:
            email = request.form.get('email')
            password = request.form.get('password')

        success, message = login_user(email, password)
        
        if success:
            # Lưu thông tin người dùng vào session
            user = {'email': email}
            session['user'] = user
            
            # Đồng bộ lịch sử xem từ session vào MongoDB
            sync_viewed_history(email)
            
            # Lấy giỏ hàng đã lưu (nếu có) từ MongoDB
            try:
                user_data = db.users.find_one({'email': email})
                if user_data and 'saved_cart' in user_data and user_data['saved_cart']:
                    # Khôi phục giỏ hàng đã lưu, nhưng không ghi đè đồ đã có trong giỏ hiện tại
                    current_cart = session.get('cart', {})
                    saved_cart = user_data['saved_cart']
                    
                    # Kết hợp giỏ hàng hiện tại với giỏ hàng đã lưu
                    for product_id, quantity in saved_cart.items():
                        if product_id not in current_cart:
                            current_cart[product_id] = quantity
                    
                    session['cart'] = current_cart
                    print(f"Restored saved cart for user {email}")
            except Exception as e:
                print(f"Error restoring saved cart: {e}")
            
            flash(message, 'success')
            
            # Nếu là AJAX, trả về JSON
            if request.is_json:
                return jsonify({'message': message, 'success': True}), 200
            # Nếu là form submit thông thường, chuyển hướng
            return redirect(url_for('home'))
        
        flash(message, 'danger')
        
        # Nếu là AJAX, trả về JSON
        if request.is_json:
            return jsonify({'message': message, 'success': False}), 400
        # Nếu là form submit thông thường, hiển thị lại form
        return render_template('auth/login.html', categories=product_categories, error=message)

@app.route('/logout')
def logout():
    """Logout route to clear session"""
    current_user = session.get('user')
    
    # Lưu giỏ hàng hiện tại vào MongoDB trước khi xóa session
    if current_user and 'email' in current_user and db is not None:
        cart = session.get('cart', {})
        
        if cart:
            try:
                # Cập nhật giỏ hàng đã lưu trong cơ sở dữ liệu
                db.users.update_one(
                    {'email': current_user['email']},
                    {'$set': {'saved_cart': cart}}
                )
                print(f"Saved cart for user {current_user['email']}")
            except Exception as e:
                print(f"Error saving cart: {e}")
    
    # Xóa session
    session.pop('user', None)
    # Không xóa session['cart'] và session['viewed_history'] để duy trì trải nghiệm cho người dùng không đăng nhập
    
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

# Hàm xử lý lịch sử xem
def track_view_history(product_id):
    """Theo dõi lịch sử xem sản phẩm"""
    product = get_product_details(product_id)
    if not product:
        return
        
    current_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Lưu vào session cho tất cả người dùng (đăng nhập hoặc không)
    viewed_history = session.get('viewed_history', [])
    
    # Loại bỏ sản phẩm nếu đã xem trước đó
    viewed_history = [item for item in viewed_history if item['product_id'] != product_id]
    
    # Thêm vào đầu danh sách
    viewed_history.insert(0, {
        'product_id': product_id,
        'timestamp': current_time
    })
    
    # Giới hạn số lượng sản phẩm lưu trữ
    viewed_history = viewed_history[:20]
    session['viewed_history'] = viewed_history
    
    # Lưu vào MongoDB nếu người dùng đã đăng nhập
    current_user = session.get('user')
    if current_user and 'email' in current_user and db is not None:
        try:
            # Cập nhật lịch sử xem của người dùng
            db.user_behaviors.update_one(
                {'email': current_user['email']},
                {
                    '$push': {
                        'viewed_products': {
                            '$each': [{
                                'product_id': product_id,
                                'product_name': product['name'],
                                'category': product.get('category', 'Uncategorized'),
                                'timestamp': current_time
                            }],
                            '$position': 0,
                            '$slice': 50  # Giữ tối đa 50 sản phẩm đã xem
                        }
                    },
                    '$inc': {
                        f'category_views.{product.get("category", "Uncategorized")}': 1
                    }
                },
                upsert=True  # Tạo document mới nếu chưa tồn tại
            )
        except Exception as e:
            print(f"Error saving view history to MongoDB: {e}")

@app.route('/product/<product_id>')
def product_detail(product_id):
    """Product detail page"""
    product = get_product_details(product_id)
    
    if product is None:
        flash('Product not found', 'error')
        return redirect(url_for('home'))
    
    # Theo dõi lịch sử xem sản phẩm
    track_view_history(product_id)
    
    # Lấy sản phẩm tương tự theo danh mục
    similar_products = []
    if df is not None:
        # Get products from same category
        same_category = df[df['category'] == product['category']]
        sample_size = min(4, len(same_category))
        if sample_size > 0:
            for _, prod in same_category.sample(sample_size).iterrows():
                if prod['product_id'] != product_id:
                    similar_products.append({
                        'product_id': prod['product_id'],
                        'name': prod['product_name'],
                        'price': prod.get('actual_price', 'N/A'),
                        'image_url': prod.get('img_link', '/static/img/placeholder.jpg')
                    })
    
    # Get current user (if logged in)
    current_user = session.get('user')
    
    return render_template('product_detail.html', 
                          product=product, 
                          similar_products=similar_products,
                          categories=product_categories,
                          current_user=current_user)

# Hàm lấy sản phẩm đã xem gần đây
def get_recently_viewed_products(limit=4):
    """Lấy danh sách sản phẩm đã xem gần đây"""
    current_user = session.get('user')
    viewed_products = []
    
    # Nếu người dùng đã đăng nhập, ưu tiên lấy từ MongoDB
    if current_user and 'email' in current_user and db is not None:
        try:
            user_behavior = db.user_behaviors.find_one({'email': current_user['email']})
            if user_behavior and 'viewed_products' in user_behavior:
                # Lấy sản phẩm đã xem từ MongoDB
                for item in user_behavior['viewed_products'][:limit]:
                    product = get_product_details(item['product_id'])
                    if product:
                        viewed_products.append(product)
        except Exception as e:
            print(f"Error getting viewed products from MongoDB: {e}")
    
    # Nếu không có dữ liệu từ MongoDB hoặc chưa đăng nhập, lấy từ session
    if not viewed_products:
        viewed_history = session.get('viewed_history', [])
        for item in viewed_history[:limit]:
            product = get_product_details(item['product_id'])
            if product:
                viewed_products.append(product)
    
    return viewed_products

# Hàm lấy đề xuất dựa trên lịch sử xem
def get_recommendations_from_history(limit=4):
    """Lấy đề xuất dựa trên lịch sử xem sản phẩm"""
    current_user = session.get('user')
    
    # Lấy lịch sử xem từ MongoDB (nếu đã đăng nhập)
    viewed_product_ids = []
    if current_user and 'email' in current_user and db is not None:
        try:
            user_behavior = db.user_behaviors.find_one({'email': current_user['email']})
            if user_behavior and 'viewed_products' in user_behavior:
                viewed_product_ids = [item['product_id'] for item in user_behavior['viewed_products'][:10]]
        except Exception as e:
            print(f"Error getting viewed products from MongoDB: {e}")
    
    # Nếu không có dữ liệu từ MongoDB hoặc chưa đăng nhập, lấy từ session
    if not viewed_product_ids:
        viewed_history = session.get('viewed_history', [])
        viewed_product_ids = [item['product_id'] for item in viewed_history[:10]]
    
    # Nếu vẫn không có lịch sử xem, trả về sản phẩm phổ biến (cold start)
    if not viewed_product_ids:
        return get_popular_products(limit)
    
    # Sử dụng hàm get_recommended_products hiện có để lấy đề xuất
    recommended_products = get_recommended_products(viewed_product_ids, limit)
    
    # Loại bỏ các sản phẩm đã xem khỏi danh sách đề xuất
    filtered_recommendations = []
    for product in recommended_products:
        if product['product_id'] not in viewed_product_ids:
            filtered_recommendations.append(product)
    
    # Nếu không đủ đề xuất, bổ sung thêm từ sản phẩm phổ biến
    if len(filtered_recommendations) < limit:
        popular_products = get_popular_products(limit * 2)
        for product in popular_products:
            if product['product_id'] not in viewed_product_ids and len(filtered_recommendations) < limit:
                # Kiểm tra xem sản phẩm này đã có trong danh sách đề xuất chưa
                if not any(rec['product_id'] == product['product_id'] for rec in filtered_recommendations):
                    filtered_recommendations.append(product)
    
    return filtered_recommendations[:limit]

# Hàm lấy sản phẩm từ danh mục ưa thích
def get_preferred_category_products(limit=4):
    """Lấy sản phẩm từ danh mục ưa thích của người dùng"""
    current_user = session.get('user')
    
    # Chỉ xử lý cho người dùng đã đăng nhập
    if not current_user or 'email' not in current_user or db is None:
        return None, []
    
    try:
        user_behavior = db.user_behaviors.find_one({'email': current_user['email']})
        if not user_behavior or 'category_views' not in user_behavior or not user_behavior['category_views']:
            return None, []
        
        # Lấy danh mục được xem nhiều nhất
        category_views = user_behavior['category_views']
        preferred_category = max(category_views.items(), key=lambda x: x[1])[0]
        
        # Lấy sản phẩm từ danh mục đó
        preferred_products = get_products_by_category(preferred_category, limit)
        
        return preferred_category, preferred_products
    except Exception as e:
        print(f"Error getting preferred category products: {e}")
        return None, []

@app.route('/category/<category>')
def category(category):
    """Category page"""
    products = get_products_by_category(category)
    
    # Get current user (if logged in)
    current_user = session.get('user')
    
    return render_template('category.html', 
                          category=category,
                          products=products,
                          categories=product_categories,
                          current_user=current_user)

@app.route('/search')
def search():
    """Search results page"""
    query = request.args.get('q', '')
    
    if not query:
        return redirect(url_for('home'))
    
    results = search_products(query)
    
    # Get current user (if logged in)
    current_user = session.get('user')
    
    return render_template('search_results.html',
                          query=query,
                          results=results,
                          categories=product_categories,
                          current_user=current_user)

@app.route('/cart')
def cart():
    """Cart page"""
    cart_items = session.get('cart', {})
    
    # Get product details for cart items
    cart_products = []
    total = 0
    
    for product_id, quantity in cart_items.items():
        product = get_product_details(product_id)
        if product:
            # Calculate price (use discounted price if available)
            price = product['discounted_price']
            if price == 'N/A':
                price = product['price']
            
            # Try to convert price to number
            try:
                if isinstance(price, str):
                    # Remove currency symbol and commas
                    price = price.replace('₹', '').replace(',', '').strip()
                price_num = float(price)
            except:
                price_num = 0
            
            # Calculate item total
            item_total = price_num * quantity
            total += item_total
            
            cart_products.append({
                'product_id': product_id,
                'name': product['name'],
                'price': price,
                'quantity': quantity,
                'item_total': item_total,
                'image_url': product['image_url']
            })
    
    # Get recommendations based on cart
    recommended_products = get_recommended_products(list(cart_items.keys()), 4)
    
    # Get current user (if logged in)
    current_user = session.get('user')
    
    return render_template('cart.html',
                          cart_products=cart_products,
                          total=total,
                          recommended_products=recommended_products,
                          categories=product_categories,
                          current_user=current_user)

@app.route('/add_to_cart/<product_id>', methods=['POST'])
def add_to_cart(product_id):
    """Add product to cart"""
    quantity = int(request.form.get('quantity', 1))
    
    # Check if product exists
    product = get_product_details(product_id)
    if product is None:
        flash('Product not found', 'error')
        return redirect(url_for('home'))
    
    # Add to cart in session
    cart = session.get('cart', {})
    
    if product_id in cart:
        cart[product_id] += quantity
    else:
        cart[product_id] = quantity
    
    session['cart'] = cart
    flash(f"Added {product['name']} to cart", 'success')
    
    # Redirect back to previous page or product page
    next_page = request.form.get('next') or url_for('product_detail', product_id=product_id)
    return redirect(next_page)

@app.route('/update_cart', methods=['POST'])
def update_cart():
    """Update cart quantities"""
    cart = session.get('cart', {})
    
    for key, value in request.form.items():
        if key.startswith('quantity_'):
            product_id = key.replace('quantity_', '')
            try:
                quantity = int(value)
                if quantity > 0:
                    cart[product_id] = quantity
                else:
                    # Remove item if quantity is 0
                    if product_id in cart:
                        del cart[product_id]
            except:
                pass
    
    session['cart'] = cart
    flash('Cart updated', 'success')
    return redirect(url_for('cart'))

@app.route('/remove_from_cart/<product_id>')
def remove_from_cart(product_id):
    """Remove item from cart"""
    cart = session.get('cart', {})
    
    if product_id in cart:
        del cart[product_id]
        session['cart'] = cart
        flash('Item removed from cart', 'success')
    
    return redirect(url_for('cart'))

@app.route('/checkout')
def checkout():
    """Checkout page"""
    cart_items = session.get('cart', {})
    
    if not cart_items:
        flash('Your cart is empty', 'info')
        return redirect(url_for('home'))
    
    # Get product details for cart items
    cart_products = []
    total = 0
    
    for product_id, quantity in cart_items.items():
        product = get_product_details(product_id)
        if product:
            # Calculate price (use discounted price if available)
            price = product['discounted_price']
            if price == 'N/A':
                price = product['price']
            
            # Try to convert price to number
            try:
                if isinstance(price, str):
                    # Remove currency symbol and commas
                    price = price.replace('₹', '').replace(',', '').strip()
                price_num = float(price)
            except:
                price_num = 0
            
            # Calculate item total
            item_total = price_num * quantity
            total += item_total
            
            cart_products.append({
                'product_id': product_id,
                'name': product['name'],
                'price': price,
                'quantity': quantity,
                'item_total': item_total
            })
    
    # Get current user (if logged in)
    current_user = session.get('user')
    
    return render_template('checkout.html',
                          cart_products=cart_products,
                          total=total,
                          categories=product_categories,
                          current_user=current_user)

@app.route('/complete_order', methods=['POST'])
def complete_order():
    """Process the order"""
    # In a real application, you would process payment, save order to database, etc.
    
    # Clear the cart
    session['cart'] = {}
    
    flash('Order completed successfully!', 'success')
    return redirect(url_for('home'))

# Initialize the application
with app.app_context():
    # Load model and data
    globals_dict = load_model_and_data()
    model = globals_dict.get('model')
    metadata = globals_dict.get('metadata')
    df = globals_dict.get('df')
    product_encoder = globals_dict.get('product_encoder')
    idx_to_product = globals_dict.get('idx_to_product')
    product_to_name = globals_dict.get('product_to_name')
    rating_scaler = globals_dict.get('rating_scaler')
    product_categories = globals_dict.get('product_categories')
    user_embeddings = globals_dict.get('user_embeddings')
    product_embeddings = globals_dict.get('product_embeddings')
    
    # Lưu dữ liệu vào MongoDB nếu có kết nối và collection rỗng
    if products_collection is not None and df is not None:
        try:
            if products_collection.count_documents({}) == 0:
                print("Saving data to MongoDB...")
                process_and_save_data('amazon.csv')
                print("Data saved to MongoDB successfully")
        except Exception as e:
            print(f"Error saving data to MongoDB: {e}")

if __name__ == '__main__':
    app.run(debug=True)
