// routes/users.js
const express = require('express');
const router = express.Router();
const bcrypt = require('bcrypt');
const db = require('../database/db');

// API đăng ký user mới
router.post('/register', async (req, res) => {
    try {
        const { fullname, email, phone, password } = req.body;

        // Kiểm tra email đã tồn tại
        db.get('SELECT email FROM users WHERE email = ?', [email], async (err, row) => {
            if (err) {
                return res.status(500).json({ error: 'Database error' });
            }
            if (row) {
                return res.status(400).json({ error: 'Email đã được sử dụng' });
            }

            // Kiểm tra số điện thoại đã tồn tại
            db.get('SELECT phone FROM users WHERE phone = ?', [phone], async (err, row) => {
                if (err) {
                    return res.status(500).json({ error: 'Database error' });
                }
                if (row) {
                    return res.status(400).json({ error: 'Số điện thoại đã được sử dụng' });
                }

                try {
                    // Mã hóa mật khẩu
                    const hashedPassword = await bcrypt.hash(password, 10);

                    // Thêm user mới vào database
                    const sql = `INSERT INTO users (fullname, email, phone, password) 
                               VALUES (?, ?, ?, ?)`;
                    
                    db.run(sql, [fullname, email, phone, hashedPassword], function(err) {
                        if (err) {
                            return res.status(500).json({ error: 'Không thể tạo tài khoản' });
                        }
                        
                        res.status(201).json({
                            message: 'Đăng ký thành công',
                            userId: this.lastID
                        });
                    });
                } catch (error) {
                    res.status(500).json({ error: 'Server error' });
                }
            });
        });
    } catch (error) {
        res.status(500).json({ error: 'Server error' });
    }
});

module.exports = router;