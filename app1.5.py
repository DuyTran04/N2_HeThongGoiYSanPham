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
        if os.path.exists('amazon_cleaned.csv'):
            df = pd.read_csv('amazon_cleaned.csv')
            print(f"Data loaded successfully from amazon_cleaned.csv: {len(df)} products")
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
                    process_and_save_data('amazon_cleaned.csv')
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
    """Home page"""
    # Lấy top 5 sản phẩm dựa trên popularity
    top_products = get_top_products(limit=5)
    
    popular_products = get_popular_products(8)
    
    # Get random featured category
    if product_categories is not None and len(product_categories) > 0:
        featured_category = random.choice(product_categories)
        featured_products = get_products_by_category(featured_category, 4)
    else:
        featured_category = "Featured Products"
        featured_products = popular_products[:4]
    
    # Get cart for recommendations
    cart = session.get('cart', {})
    recommended_products = get_recommended_products(list(cart.keys()), 4)
    
    # Get current user (if logged in)
    current_user = session.get('user')
    
    return render_template('home.html', 
                          top_products=top_products,
                          popular_products=popular_products,
                          featured_category=featured_category,
                          featured_products=featured_products,
                          recommended_products=recommended_products,
                          categories=product_categories,
                          current_user=current_user)

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
            user = {'email': email}  # Có thể mở rộng để lưu thêm thông tin khác
            session['user'] = user
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
    session.pop('user', None)
    session.pop('cart', None)  # Optional: Clear cart on logout
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/product/<product_id>')
def product_detail(product_id):
    """Product detail page"""
    product = get_product_details(product_id)
    
    if product is None:
        flash('Product not found', 'error')
        return redirect(url_for('home'))
    
    # Get similar products
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
                process_and_save_data('amazon_cleaned.csv')
                print("Data saved to MongoDB successfully")
        except Exception as e:
            print(f"Error saving data to MongoDB: {e}")

if __name__ == '__main__':
    app.run(debug=True)
