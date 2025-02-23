// assets/js/auth.js
document.getElementById('registerForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Lấy dữ liệu từ form
    const formData = {
        fullname: document.getElementById('fullname').value,
        email: document.getElementById('email').value,
        phone: document.getElementById('phone').value,
        password: document.getElementById('password').value,
        confirmPassword: document.getElementById('confirmPassword').value
    };

    // Kiểm tra mật khẩu khớp nhau
    if (formData.password !== formData.confirmPassword) {
        alert('Mật khẩu không khớp');
        return;
    }

    try {
        const response = await fetch('/api/users/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (response.ok) {
            alert('Đăng ký thành công!');
            window.location.href = '/pages/login.html';
        } else {
            alert(data.error || 'Đăng ký thất bại');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Có lỗi xảy ra, vui lòng thử lại sau');
    }
});

// Toggle hiển thị/ẩn mật khẩu
document.querySelectorAll('.toggle-password').forEach(button => {
    button.addEventListener('click', function() {
        const input = this.parentElement.querySelector('input');
        const type = input.getAttribute('type') === 'password' ? 'text' : 'password';
        input.setAttribute('type', type);
        this.classList.toggle('fa-eye');
        this.classList.toggle('fa-eye-slash');
    });
});