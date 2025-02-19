// server.js
const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const multer = require('multer');
const cors = require('cors');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const path = require('path');

const app = express();
const PORT = 3000;
const JWT_SECRET = 'your-secret-key'; // Trong thực tế nên dùng biến môi trường

// Middleware
app.use(cors());
app.use(express.json());
app.use('/uploads', express.static('uploads'));

// Cấu hình multer để upload ảnh
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/')
    },
    filename: function (req, file, cb) {
        cb(null, Date.now() + path.extname(file.originalname))
    }
});

const upload = multer({ storage: storage });

// Kết nối database
const db = new sqlite3.Database('./database/shop.db', (err) => {
    if (err) {
        console.error('Error connecting to database:', err);
    } else {
        console.log('Connected to SQLite database');
        initDatabase();
    }
});

// Khởi tạo database
function initDatabase() {
    db.serialize(() => {
        // Tạo bảng users
        db.run(`CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )`);

        // Tạo bảng products
        db.run(`CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            description TEXT,
            price REAL,
            image_url TEXT,
            user_id INTEGER,
            category TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )`);
    });
}

// Routes cho user
app.post('/api/register', async (req, res) => {
    const { username, email, password } = req.body;

    try {
        // Hash password
        const hashedPassword = await bcrypt.hash(password, 10);

        // Thêm user vào database
        db.run(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            [username, email, hashedPassword],
            function(err) {
                if (err) {
                    if (err.message.includes('UNIQUE constraint failed')) {
                        return res.status(400).json({ error: 'Username hoặc email đã tồn tại' });
                    }
                    return res.status(500).json({ error: 'Lỗi đăng ký' });
                }
                res.json({ message: 'Đăng ký thành công', userId: this.lastID });
            }
        );
    } catch (error) {
        res.status(500).json({ error: 'Lỗi server' });
    }
});

app.post('/api/login', (req, res) => {
    const { username, password } = req.body;

    db.get(
        'SELECT * FROM users WHERE username = ? OR email = ?',
        [username, username],
        async (err, user) => {
            if (err) return res.status(500).json({ error: 'Lỗi server' });
            if (!user) return res.status(401).json({ error: 'Thông tin đăng nhập không đúng' });

            // Kiểm tra password
            const validPassword = await bcrypt.compare(password, user.password);
            if (!validPassword) return res.status(401).json({ error: 'Thông tin đăng nhập không đúng' });

            // Tạo token
            const token = jwt.sign({ userId: user.id }, JWT_SECRET, { expiresIn: '24h' });
            res.json({ token, username: user.username });
        }
    );
});

// Middleware xác thực token
function authenticateToken(req, res, next) {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) return res.status(401).json({ error: 'Chưa đăng nhập' });

    jwt.verify(token, JWT_SECRET, (err, user) => {
        if (err) return res.status(403).json({ error: 'Token không hợp lệ' });
        req.user = user;
        next();
    });
}

// Routes cho products
app.post('/api/products', authenticateToken, upload.single('image'), (req, res) => {
    const { name, description, price, category } = req.body;
    const imageUrl = req.file ? `/uploads/${req.file.filename}` : null;

    db.run(
        'INSERT INTO products (name, description, price, image_url, user_id, category) VALUES (?, ?, ?, ?, ?, ?)',
        [name, description, price, imageUrl, req.user.userId, category],
        function(err) {
            if (err) return res.status(500).json({ error: 'Lỗi khi thêm sản phẩm' });
            res.json({ message: 'Thêm sản phẩm thành công', productId: this.lastID });
        }
    );
});

app.get('/api/products', (req, res) => {
    db.all('SELECT * FROM products ORDER BY created_at DESC', [], (err, products) => {
        if (err) return res.status(500).json({ error: 'Lỗi khi lấy danh sách sản phẩm' });
        res.json(products);
    });
});

app.get('/api/products/top', (req, res) => {
    // Trong thực tế, bạn cần thêm bảng orders để tính sản phẩm bán chạy
    db.all('SELECT * FROM products ORDER BY created_at DESC LIMIT 5', [], (err, products) => {
        if (err) return res.status(500).json({ error: 'Lỗi khi lấy top sản phẩm' });
        res.json(products);
    });
});

// Khởi động server
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});