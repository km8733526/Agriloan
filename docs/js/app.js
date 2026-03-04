// ============================================================
// AgriLoan — Frontend prediction (app.js)
// - No backend required
// - Supports loading /model/weights.json (linear model)
// - Falls back to a rule-based predictor if weights are not present
// ============================================================

/* ====== CONFIG ====== */
// Put your OpenWeather API key here (create at https://openweathermap.org/)
const OPENWEATHER_API_KEY = "7b116c492e6c76fcb37eb72fe0573926";

/* ====== MODEL STORAGE ======
 If you want to use your trained model, place a file at:
   /model/weights.json
 with format:
 {
   "intercept": -1.23,
   "coefficients": {
     "landSize": 0.45,
     "irrigation_yes": 0.7,
     "cropDiversity": 0.12,
     "soilPH": 0.2,
     "rainfall": 0.15,
     "yield_avg": 0.3
   },
   "thresholds": { "approved": 0.7, "review": 0.55 }  // optional (0..1)
 }
 If not present, the script will use a sensible default rule-based estimator.
======================*/

let MODEL = null; // will hold weights if loaded

document.addEventListener('DOMContentLoaded', () => {
  initializeFormValidation();
  initializeAadhaarFormatting();
  initializeFileUpload();
  detectLocation();
  loadModelWeights();    // attempt to load /model/weights.json (optional)
  handleFormSubmission();
});

/* ---------------------------
   Load model weights (optional)
   --------------------------- */
async function loadModelWeights() {
  try {
    const res = await fetch('/model/weights.json', { cache: 'no-store' });
    if (!res.ok) throw new Error('no model file');
    const json = await res.json();
    // Basic validation
    if (json && typeof json.intercept === 'number' && json.coefficients) {
      MODEL = {
        intercept: json.intercept,
        coeffs: json.coefficients,
        thresholds: json.thresholds || { approved: 0.7, review: 0.55 }
      };
      console.log('Model weights loaded:', MODEL);
    } else {
      console.warn('Model file present but invalid. Using fallback predictor.');
      MODEL = null;
    }
  } catch (err) {
    console.log('No model weights found — using fallback predictor.');
    MODEL = null;
  }
}

/* ---------------------------
   GEOLOCATION (auto-fill lat/lon)
   --------------------------- */
function detectLocation() {
  if (!navigator.geolocation) return;
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const latEl = document.getElementById('latitude');
      const lonEl = document.getElementById('longitude');
      if (latEl && !latEl.value) latEl.value = pos.coords.latitude;
      if (lonEl && !lonEl.value) lonEl.value = pos.coords.longitude;
    },
    (err) => {
      console.log('Geolocation denied or unavailable:', err);
    },
    { timeout: 10000 }
  );
}

/* ---------------------------
   Soil data via SoilGrids (phh2o)
   --------------------------- */
async function getSoilData(lat, lon) {
  // returns soilPH (number) or default 6.5
  try {
    const url = `https://rest.isric.org/soilgrids/v2.0/properties/query?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&property=phh2o`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('soil API error');
    const data = await res.json();
    // Safe navigation
    const ph = data?.properties?.layers?.[0]?.depths?.[0]?.values?.mean;
    if (typeof ph === 'number' && !Number.isNaN(ph)) return ph;
    return 6.5;
  } catch (e) {
    console.warn('Soil API failed, using default pH:', e);
    return 6.5;
  }
}

/* ---------------------------
   Weather via OpenWeather (rainfall 1h, temperature, humidity)
   --------------------------- */
async function getWeather(lat, lon) {
  // returns { rainfall, temp, humidity }
  if (!OPENWEATHER_API_KEY || OPENWEATHER_API_KEY === 'YOUR_OPENWEATHER_KEY') {
    // If user didn't set key, return safe defaults
    return { rainfall: 0, temp: 30, humidity: 60 };
  }
  try {
    const url = `https://api.openweathermap.org/data/2.5/weather?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&appid=${OPENWEATHER_API_KEY}&units=metric`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('weather API error');
    const d = await res.json();
    return { rainfall: d?.rain?.['1h'] || 0, temp: d?.main?.temp ?? 30, humidity: d?.main?.humidity ?? 60 };
  } catch (e) {
    console.warn('Weather API failed, using defaults:', e);
    return { rainfall: 0, temp: 30, humidity: 60 };
  }
}

/* ---------------------------
   Feature engineering
   Build a numeric feature vector used by the model
   --------------------------- */
function buildFeatures(payload, soilPH, weather) {
  // numeric features and binary-encoded categories that match expected coefficients
  const yieldCount = (payload.yield1 ? 1 : 0) + (payload.yield2 ? 1 : 0) + (payload.yield3 ? 1 : 0);
  const yieldAvg = yieldCount ? ((payload.yield1 || 0) + (payload.yield2 || 0) + (payload.yield3 || 0)) / yieldCount : 0;
  return {
    landSize: Number.isFinite(payload.landSize) ? payload.landSize : 0,
    irrigation_yes: (payload.irrigation || '').toLowerCase() === 'yes' || (payload.irrigation || '').toLowerCase() === 'drip' ? 1 : 0,
    cropDiversity: Number.isFinite(payload.cropDiversity) ? payload.cropDiversity : 0,
    soilPH: Number.isFinite(soilPH) ? soilPH : 6.5,
    rainfall: Number.isFinite(weather?.rainfall) ? weather.rainfall : 0,
    yield_avg: Number.isFinite(yieldAvg) ? yieldAvg : 0
  };
}

/* ---------------------------
   Prediction using loaded linear model
   - linear combination -> sigmoid (0..1)
   - thresholds to map to Approved/Review/Rejected
   --------------------------- */
function predictWithWeights(features) {
  const intercept = MODEL.intercept || 0;
  const coeffs = MODEL.coeffs || {};
  let linear = intercept;
  for (const [k, v] of Object.entries(features)) {
    const coef = coeffs[k] ?? 0;
    linear += coef * v;
  }
  // Sigmoid transform to 0..1 probability
  const prob = 1 / (1 + Math.exp(-linear));
  const t = MODEL.thresholds || { approved: 0.7, review: 0.55 };
  let status = 'Rejected';
  if (prob >= t.approved) status = 'Approved';
  else if (prob >= t.review) status = 'Review';
  return { score: +(prob * 100).toFixed(1), status, prob, raw: linear };
}

/* ---------------------------
   Fallback predictor (rule-based)
   Used only when weights not available — keeps behavior reasonable
   --------------------------- */
function fallbackPredict(features) {
  // score out of 100
  let score = 40;
  score += Math.min(features.landSize, 100) * 0.4; // land contributes up to 40
  score += features.irrigation_yes ? 15 : 0;
  score += Math.min(features.cropDiversity, 10) * 1.5;
  // soilPH good range centered near 6.5-7.5
  if (features.soilPH >= 6 && features.soilPH <= 7.5) score += 10;
  // rainfall modest positive contribution
  if (features.rainfall > 2) score += 8;
  score += Math.min(features.yield_avg, 10) * 2; // yield contribution
  score = Math.max(0, Math.min(100, score));
  let status = 'Rejected';
  if (score >= 70) status = 'Approved';
  else if (score >= 55) status = 'Review';
  return { score: +score.toFixed(1), status };
}

/* ---------------------------
   Top-level prediction wrapper
   --------------------------- */
function predict(features) {
  if (MODEL) return predictWithWeights(features);
  return fallbackPredict(features);
}

/* ---------------------------
   Form validation helpers
   --------------------------- */
function initializeFormValidation() {
  const form = document.getElementById('loanApplicationForm');
  if (!form) return;
  const inputs = form.querySelectorAll('input[required], select[required]');
  inputs.forEach(input => {
    input.addEventListener('blur', () => validateField(input));
    input.addEventListener('input', () => {
      if (input.classList.contains('border-red-500')) validateField(input);
    });
  });
}

function validateField(field) {
  if (!field) return true;
  const value = (field.value || '').trim();
  if (value === '') {
    showFieldError(field, 'This field is required');
    return false;
  }
  if (field.id === 'phone' && !/^\d{10}$/.test(value)) {
    showFieldError(field, 'Please enter a valid 10-digit phone number');
    return false;
  }
  if (field.id === 'aadhaar' && !/^\d{4}-\d{4}-\d{4}$/.test(value)) {
    showFieldError(field, 'Please enter Aadhaar as XXXX-XXXX-XXXX');
    return false;
  }
  if (field.id === 'landSize') {
    const n = parseFloat(value);
    if (Number.isNaN(n) || n <= 0 || n > 10000) {
      showFieldError(field, 'Enter valid land size');
      return false;
    }
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
  div.className = 'error-message text-red-500 text-sm mt-1';
  div.textContent = message;
  field.parentElement.appendChild(div);
}

function clearFieldError(field) {
  field.classList.remove('border-red-500');
  field.classList.add('border-gray-300');
  const existing = field.parentElement.querySelector('.error-message');
  if (existing) existing.remove();
}

/* ---------------------------
   Aadhaar formatting (XXXX-XXXX-XXXX)
   --------------------------- */
function initializeAadhaarFormatting() {
  const input = document.getElementById('aadhaar');
  if (!input) return;
  input.addEventListener('input', (e) => {
    let digits = (e.target.value || '').replace(/\D/g, '');
    let formatted = '';
    for (let i = 0; i < digits.length && i < 12; i++) {
      if (i === 4 || i === 8) formatted += '-';
      formatted += digits[i];
    }
    e.target.value = formatted;
  });
}

/* ---------------------------
   File upload UI
   --------------------------- */
function initializeFileUpload() {
  const fileInput = document.getElementById('landDocument');
  if (!fileInput) return;
  const uploadArea = fileInput.nextElementSibling;
  uploadArea?.addEventListener('click', () => fileInput.click());
  uploadArea?.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('border-green-500'); });
  uploadArea?.addEventListener('dragleave', () => uploadArea.classList.remove('border-green-500'));
  uploadArea?.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('border-green-500');
    if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFileSelect(e.target.files[0]);
  });
}

function handleFileSelect(file) {
  const allowed = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg'];
  if (!allowed.includes(file.type)) { showNotification('Please upload PDF or image', 'error'); return; }
  if (file.size > 10 * 1024 * 1024) { showNotification('File must be <10MB', 'error'); return; }
  const para = document.querySelector('#landDocument + div p');
  if (para) {
    para.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    para.classList.add('text-green-600', 'font-semibold');
  }
  showNotification('File selected', 'success');
}

/* ---------------------------
   Loading modal helpers
   --------------------------- */
function showLoadingModal() {
  const modal = document.getElementById('loadingModal');
  if (modal) { modal.classList.remove('hidden'); modal.classList.add('flex'); }
}
function hideLoadingModal() {
  const modal = document.getElementById('loadingModal');
  if (modal) { modal.classList.add('hidden'); modal.classList.remove('flex'); }
}

/* ---------------------------
   Notifications
   --------------------------- */
function showNotification(message, type = 'info') {
  const colors = { error: 'bg-red-500', success: 'bg-green-500', info: 'bg-blue-500' };
  const el = document.createElement('div');
  el.className = `notification ${colors[type] || colors.info} text-white p-3 fixed right-4 bottom-4 rounded shadow`;
  el.innerHTML = `<div>${message}</div>`;
  document.body.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity 0.4s'; setTimeout(() => el.remove(), 400); }, 3000);
}

/* ---------------------------
   Main: handle form submit -> fetch APIs -> predict -> save -> redirect
   --------------------------- */
function handleFormSubmission() {
  const form = document.getElementById('loanApplicationForm');
  if (!form) return;
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    // Basic client-side validation
    let isValid = true;
    form.querySelectorAll('input[required], select[required]').forEach(f => { if (!validateField(f)) isValid = false; });

    const lat = document.getElementById('latitude')?.value;
    const lon = document.getElementById('longitude')?.value;
    if (!lat || !lon) {
      showNotification('Please mark your land location (latitude/longitude).', 'error');
      isValid = false;
    }
    if (!isValid) { return; }

    showLoadingModal();

    // Build payload (fields expected by your form)
    const payload = {
      fullName: document.getElementById('fullName')?.value || '',
      aadhaar: document.getElementById('aadhaar')?.value || '',
      phone: document.getElementById('phone')?.value || '',
      district: document.getElementById('district')?.value || '',
      village: document.getElementById('village')?.value || '',
      surveyNumber: document.getElementById('surveyNumber')?.value || '',
      landSize: parseFloat(document.getElementById('landSize')?.value) || 0,
      irrigation: document.getElementById('irrigation')?.value || 'no',
      ownership: document.getElementById('ownership')?.value || '',
      primaryCrop: document.getElementById('primaryCrop')?.value || '',
      cropDiversity: parseInt(document.getElementById('cropDiversity')?.value) || 0,
      yield1: parseFloat(document.getElementById('yield1')?.value) || 0,
      yield2: parseFloat(document.getElementById('yield2')?.value) || 0,
      yield3: parseFloat(document.getElementById('yield3')?.value) || 0,
      latitude: parseFloat(lat),
      longitude: parseFloat(lon)
    };

    try {
      // fetch external data
      const [soilPH, weather] = await Promise.all([
        getSoilData(payload.latitude, payload.longitude),
        getWeather(payload.latitude, payload.longitude)
      ]);

      // prepare features & predict
      const features = buildFeatures(payload, soilPH, weather);
      const prediction = predict(features); // returns {score, status, ...}

      // store and redirect to result page
      localStorage.setItem('applicationData', JSON.stringify(payload));
      localStorage.setItem('evaluationResult', JSON.stringify({
        ...prediction,
        features,
        soilPH,
        weather
      }));

      hideLoadingModal();
      window.location.href = 'result.html';
    } catch (err) {
      console.error('Prediction error', err);
      hideLoadingModal();
      showNotification('Prediction failed. Try again later.', 'error');
    }
  });
}
