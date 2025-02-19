// Các hàm giao tiếp với API
async function registerUser(userData) {
    try {
        const response = await fetch('http://localhost:3000/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(userData)
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.error);
        return data;
    } catch (error) {
        throw error;
    }
}

async function loginUser(credentials) {
    try {
        const response = await fetch('http://localhost:3000/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(credentials)
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.error);
        return data;
    } catch (error) {
        throw error;
    }
}

// Lưu token
function saveUserToken(token, username) {
    localStorage.setItem('userToken', token);
    localStorage.setItem('username', username);
}

// Xử lý đăng ký
document.getElementById('registerForm')?.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const username = document.getElementById('username').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;

    if (password !== confirmPassword) {
        alert('Mật khẩu không khớp!');
        return;
    }

    try {
        await registerUser({ username, email, password });
        alert('Đăng ký thành công!');
        window.location.href = 'login.html';
    } catch (error) {
        alert(error.message);
    }
});

// Xử lý đăng nhập
document.getElementById('loginForm')?.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    try {
        const { token, username: userName } = await loginUser({ username, password });
        saveUserToken(token, userName);
        alert('Đăng nhập thành công!');
        window.location.href = '../index.html';
    } catch (error) {
        alert(error.message);
    }
});

// Xử lý đăng xuất
function logout() {
    localStorage.removeItem('userToken');
    localStorage.removeItem('username');
    window.location.href = 'login.html';
}

// Kiểm tra trạng thái đăng nhập
function checkLoginStatus() {
    const token = localStorage.getItem('userToken');
    const username = localStorage.getItem('username');
    const userActions = document.querySelector('.user-actions');
    const userMenu = document.querySelector('.user-menu');
    
    if (token && username) {
        if (userActions) {
            document.querySelector('.btn-login')?.classList.add('hidden');
            document.querySelector('.btn-register')?.classList.add('hidden');
            userMenu?.classList.remove('hidden');
            const usernameElement = userMenu?.querySelector('.username');
            if (usernameElement) {
                usernameElement.textContent = `Xin chào, ${username}`;
            }
        }
    } else {
        if (userActions) {
            document.querySelector('.btn-login')?.classList.remove('hidden');
            document.querySelector('.btn-register')?.classList.remove('hidden');
            userMenu?.classList.add('hidden');
        }
    }
}

// Thêm sự kiện đăng xuất
document.querySelector('.btn-logout')?.addEventListener('click', function(e) {
    e.preventDefault();
    logout();
});

// Kiểm tra trạng thái đăng nhập khi trang tải xong
document.addEventListener('DOMContentLoaded', checkLoginStatus);