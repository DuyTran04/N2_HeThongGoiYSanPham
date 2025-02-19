// Mẫu dữ liệu sản phẩm nổi bật
const featuredProducts = [
    {
        id: 1,
        name: 'Sản phẩm 1',
        price: 199000,
        image: '/api/placeholder/200/200',
        description: 'Mô tả sản phẩm 1'
    },
    {
        id: 2,
        name: 'Sản phẩm 2',
        price: 299000,
        image: '/api/placeholder/200/200',
        description: 'Mô tả sản phẩm 2'
    },
    // Thêm sản phẩm khác...
];

// Mẫu dữ liệu top bán chạy
const bestSellers = [
    {
        id: 1,
        name: 'Top 1',
        price: 399000,
        sales: 150,
        image: '/api/placeholder/200/200'
    },
    // Thêm sản phẩm khác...
];

// Hiển thị sản phẩm nổi bật
function displayFeaturedProducts() {
    const productGrid = document.querySelector('.product-grid');
    featuredProducts.forEach(product => {
        const productElement = document.createElement('div');
        productElement.className = 'product-card';
        productElement.innerHTML = `
            <img src="${product.image}" alt="${product.name}">
            <h3>${product.name}</h3>
            <p>${product.description}</p>
            <span class="price">${product.price.toLocaleString('vi-VN')}đ</span>
            <button class="btn-add-cart">Thêm vào giỏ</button>
        `;
        productGrid.appendChild(productElement);
    });
}

// Hiển thị top bán chạy
function displayBestSellers() {
    const bestSellersGrid = document.querySelector('.best-sellers-grid');
    bestSellers.forEach(product => {
        const productElement = document.createElement('div');
        productElement.className = 'product-card';
        productElement.innerHTML = `
            <img src="${product.image}" alt="${product.name}">
            <h3>${product.name}</h3>
            <span class="price">${product.price.toLocaleString('vi-VN')}đ</span>
            <span class="sales">Đã bán: ${product.sales}</span>
        `;
        bestSellersGrid.appendChild(productElement);
    });
}

// Xử lý form đăng sản phẩm
document.getElementById('uploadForm')?.addEventListener('submit', function(e) {
    e.preventDefault();
    // Xử lý logic đăng sản phẩm ở đây
    alert('Chức năng đang được phát triển!');
});

// Khởi tạo trang
window.addEventListener('DOMContentLoaded', () => {
    displayFeaturedProducts();
    displayBestSellers();
});

// Xử lý đăng nhập/đăng xu