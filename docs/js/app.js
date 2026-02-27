// ============================================================
// app.js  —  AgriLoan Credit Scorer
// Connects the loan application form to the Flask backend API.
// Backend must be running at http://127.0.0.1:5000
// ============================================================

const API_BASE = "https://your-backend-url.onrender.com";
document.addEventListener('DOMContentLoaded', function () {
    initializeFormValidation();
    initializeAadhaarFormatting();
    initializeFileUpload();
    handleFormSubmission();
});

// ── Form validation ──────────────────────────────────────────────────────────

function initializeFormValidation() {
    const form   = document.getElementById('loanApplicationForm');
    if (!form) return;
    const inputs = form.querySelectorAll('input[required], select[required]');
    inputs.forEach(input => {
        input.addEventListener('blur',  () => validateField(input));
        input.addEventListener('input', () => {
            if (input.classList.contains('border-red-500')) validateField(input);
        });
    });
}

function validateField(field) {
    const value = field.value.trim();
    if (value === '') {
        showFieldError(field, 'This field is required');
        return false;
    }
    if (field.id === 'phone' && !/^\d{10}$/.test(value)) {
        showFieldError(field, 'Please enter a valid 10-digit phone number');
        return false;
    }
    if (field.id === 'aadhaar' && !/^\d{4}-\d{4}-\d{4}$/.test(value)) {
        showFieldError(field, 'Please enter a valid Aadhaar number (XXXX-XXXX-XXXX)');
        return false;
    }
    if (field.id === 'landSize' && (parseFloat(value) <= 0 || parseFloat(value) > 1000)) {
        showFieldError(field, 'Please enter a valid land size between 0 and 1000 acres');
        return false;
    }
    clearFieldError(field);
    return true;
}

function showFieldError(field, message) {
    field.classList.add('border-red-500');
    field.classList.remove('border-gray-300');
    const existing = field.parentElement.querySelector('.error-message');
    if (existing) existing.remove();
    const div = document.createElement('div');
    div.className   = 'error-message text-red-500 text-sm mt-1';
    div.textContent = message;
    field.parentElement.appendChild(div);
}

function clearFieldError(field) {
    field.classList.remove('border-red-500');
    field.classList.add('border-gray-300');
    const existing = field.parentElement.querySelector('.error-message');
    if (existing) existing.remove();
}

// ── Aadhaar formatting ───────────────────────────────────────────────────────

function initializeAadhaarFormatting() {
    const input = document.getElementById('aadhaar');
    if (!input) return;
    input.addEventListener('input', function (e) {
        let digits = e.target.value.replace(/\D/g, '');
        let formatted = '';
        for (let i = 0; i < digits.length && i < 12; i++) {
            if (i === 4 || i === 8) formatted += '-';
            formatted += digits[i];
        }
        e.target.value = formatted;
    });
}

// ── File upload ──────────────────────────────────────────────────────────────

function initializeFileUpload() {
    const fileInput  = document.getElementById('landDocument');
    if (!fileInput) return;
    const uploadArea = fileInput.nextElementSibling;

    uploadArea.addEventListener('click',    () => fileInput.click());
    uploadArea.addEventListener('dragover', e  => { e.preventDefault(); uploadArea.classList.add('border-green-500'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('border-green-500'));
    uploadArea.addEventListener('drop', e => {
        e.preventDefault();
        uploadArea.classList.remove('border-green-500');
        if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', e => {
        if (e.target.files.length) handleFileSelect(e.target.files[0]);
    });
}

function handleFileSelect(file) {
    const allowed = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg'];
    if (!allowed.includes(file.type)) { showNotification('Please upload a PDF or image file', 'error'); return; }
    if (file.size > 10 * 1024 * 1024)  { showNotification('File size must be less than 10MB', 'error'); return; }
    const para = document.querySelector('#landDocument + div p');
    if (para) {
        para.textContent = `Selected: ${file.name} (${(file.size/1024/1024).toFixed(2)} MB)`;
        para.classList.add('text-green-600', 'font-semibold');
    }
    showNotification('File selected successfully', 'success');
}

// ── Form submission — calls Flask backend ────────────────────────────────────

function handleFormSubmission() {
    const form = document.getElementById('loanApplicationForm');
    if (!form) return;

    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        // Validate required fields
        let isValid = true;
        form.querySelectorAll('input[required], select[required]').forEach(field => {
            if (!validateField(field)) isValid = false;
        });

        const lat = document.getElementById('latitude').value;
        const lng = document.getElementById('longitude').value;
        if (!lat || !lng) {
            showNotification('Please mark your land location on the map', 'error');
            isValid = false;
        }

        if (!isValid) {
            showNotification('Please fill all required fields correctly', 'error');
            return;
        }

        showLoadingModal();

        // Build payload matching what the backend expects
        const payload = {
            fullName:      document.getElementById('fullName').value,
            aadhaar:       document.getElementById('aadhaar').value,
            phone:         document.getElementById('phone').value,
            district:      document.getElementById('district').value,
            village:       document.getElementById('village').value,
            surveyNumber:  document.getElementById('surveyNumber').value,
            landSize:      parseFloat(document.getElementById('landSize').value),
            irrigation:    document.getElementById('irrigation').value,
            ownership:     document.getElementById('ownership').value,
            primaryCrop:   document.getElementById('primaryCrop').value,
            cropDiversity: parseInt(document.getElementById('cropDiversity').value),
            yield1:        parseFloat(document.getElementById('yield1').value) || 0,
            yield2:        parseFloat(document.getElementById('yield2').value) || 0,
            yield3:        parseFloat(document.getElementById('yield3').value) || 0,
            latitude:      parseFloat(lat),
            longitude:     parseFloat(lng),
        };

        try {
            const response = await fetch(`${API_BASE}/evaluate`, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify(payload),
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || `Server error ${response.status}`);
            }

            // Save both the form data and the full backend result for result.html
            localStorage.setItem('applicationData',  JSON.stringify(payload));
            localStorage.setItem('evaluationResult', JSON.stringify(result));

            hideLoadingModal();
            window.location.href = 'result.html';

        } catch (err) {
            hideLoadingModal();
            console.error('Evaluation error:', err);
            showNotification(
                err.message.includes('fetch')
                    ? 'Cannot reach backend. Is the server running on port 5000?'
                    : 'Error: ' + err.message,
                'error'
            );
        }
    });
}

// ── Modal helpers ────────────────────────────────────────────────────────────

function showLoadingModal() {
    const modal = document.getElementById('loadingModal');
    if (modal) { modal.classList.remove('hidden'); modal.classList.add('flex'); }
}

function hideLoadingModal() {
    const modal = document.getElementById('loadingModal');
    if (modal) { modal.classList.add('hidden'); modal.classList.remove('flex'); }
}

// ── Toast notification ───────────────────────────────────────────────────────

function showNotification(message, type = 'info') {
    const colors = { error: 'bg-red-500', success: 'bg-green-500', info: 'bg-blue-500' };
    const el = document.createElement('div');
    el.className = `notification ${colors[type] || colors.info} text-white`;
    el.innerHTML = `<div class="flex items-center"><span>${message}</span></div>`;
    document.body.appendChild(el);
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transition = 'opacity 0.3s';
        setTimeout(() => el.remove(), 300);
    }, 3500);
}

// Slide-out animation
const _style = document.createElement('style');
_style.textContent = `
@keyframes slideOut {
  0%   { transform: translateX(0);    opacity: 1; }
  100% { transform: translateX(100%); opacity: 0; }
}`;

document.head.appendChild(_style);
