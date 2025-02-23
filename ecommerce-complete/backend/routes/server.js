const express = require('express');
const path = require('path');
const app = express();
const userRoutes = require('./routes/users');

// Middleware
app.use(express.json());
app.use(express.static(path.join(__dirname, '../')));

// Routes
app.use('/api/users', userRoutes);

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});