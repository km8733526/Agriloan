// ============================================================
// dashboard.js  —  Lender Dashboard
// Fetches live application data from the Flask backend API.
// ============================================================

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000";
document.addEventListener('DOMContentLoaded', function () {
    if (document.getElementById('applicationsTable')) {
        initializeDashboard();
    }
});

// ── Bootstrap ────────────────────────────────────────────────────────────────

async function initializeDashboard() {
    await loadStats();
    await loadApplications();
    initializeFilters();
}

// ── Stats cards (total / approved / review / rejected) ───────────────────────

async function loadStats() {
    try {
        const res  = await fetch(`${API_BASE}/stats`);
        const data = await res.json();

        // Update the 4 stat cards if they have data-stat attributes
        // Falls back gracefully if the HTML doesn't have them
        safeSetText('stat-total',    data.total    ?? '—');
        safeSetText('stat-approved', data.approved ?? '—');
        safeSetText('stat-review',   data.review   ?? '—');
        safeSetText('stat-rejected', data.rejected ?? '—');
    } catch (err) {
        console.warn('Could not load stats:', err.message);
    }
}

function safeSetText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

// ── Applications table ────────────────────────────────────────────────────────

let _allApplications = [];   // cache for client-side filtering

async function loadApplications(filters = {}) {
    const tbody = document.getElementById('applicationsTable');
    if (!tbody) return;

    // Build query string from filter object
    const params = new URLSearchParams();
    if (filters.status)   params.set('status',   filters.status);
    if (filters.district) params.set('district', filters.district);
    if (filters.search)   params.set('search',   filters.search);
    params.set('per_page', '100');

    try {
        const res  = await fetch(`${API_BASE}/applications?${params}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        _allApplications = data.applications || [];
        renderTable(_allApplications);
        updateResultCount(_allApplications.length, data.total);

    } catch (err) {
        console.error('Failed to load applications:', err);
        tbody.innerHTML = `
            <tr>
              <td colspan="8" class="px-6 py-8 text-center text-red-500">
                ⚠️ Could not connect to backend. 
                Make sure <strong>python app.py</strong> is running on port 5000.
                <br><small>${err.message}</small>
              </td>
            </tr>`;
    }
}

function renderTable(applications) {
    const tbody = document.getElementById('applicationsTable');
    if (!tbody) return;

    if (applications.length === 0) {
        tbody.innerHTML = `
            <tr>
              <td colspan="8" class="px-6 py-8 text-center text-gray-400">
                No applications found.
              </td>
            </tr>`;
        return;
    }

    tbody.innerHTML = applications.map(app => `
        <tr class="hover:bg-gray-50 cursor-pointer">
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${app.application_id || ('#AL' + String(app.db_id).padStart(7,'0'))}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                ${escHtml(app.farmer_name || '—')}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 capitalize">
                ${escHtml(app.district || '—')}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${app.land_size ? app.land_size + ' acres' : '—'}
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
                <span class="text-2xl font-bold ${trustScoreColor(app.trust_score)}">
                    ${app.trust_score ?? '—'}
                </span>
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
                ${riskBadge(app.risk_level)}
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
                ${statusBadge(app.status)}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm">
                <button onclick="viewDetails(${app.db_id})"
                        class="text-blue-600 hover:text-blue-900 font-medium">
                    View Details
                </button>
            </td>
        </tr>`
    ).join('');
}

function updateResultCount(showing, total) {
    const el = document.querySelector('.showing-count');
    if (el) el.textContent = `Showing ${showing} of ${total} results`;
}

// ── Filters ───────────────────────────────────────────────────────────────────

function initializeFilters() {
    const search   = document.getElementById('searchFarmer');
    const district = document.getElementById('filterDistrict');
    const status   = document.getElementById('filterStatus');

    if (search)   search.addEventListener('input',   debounce(applyFilters, 300));
    if (district) district.addEventListener('change', applyFilters);
    if (status)   status.addEventListener('change',   applyFilters);
}

function applyFilters() {
    const search   = (document.getElementById('searchFarmer')?.value   || '').trim();
    const district = (document.getElementById('filterDistrict')?.value || '').trim();
    const status   = (document.getElementById('filterStatus')?.value   || '').trim();

    // Map frontend filter values → backend status names
    const statusMap = { approved: 'Approved', review: 'Review', rejected: 'Rejected' };

    loadApplications({
        search,
        district,
        status: statusMap[status] || '',
    });
}

// ── Navigate to result page for a specific application ────────────────────────

function viewDetails(dbId) {
    const app = _allApplications.find(a => a.db_id === dbId);
    if (app) {
        localStorage.setItem('viewApplication', JSON.stringify(app));
    }
    window.location.href = `result.html?id=${dbId}`;
}

// ── Badge helpers ─────────────────────────────────────────────────────────────

function trustScoreColor(score) {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
}

function riskBadge(level) {
    const map = {
        Low:    'bg-green-100 text-green-800',
        Medium: 'bg-yellow-100 text-yellow-800',
        High:   'bg-red-100 text-red-800',
    };
    const cls = map[level] || 'bg-gray-100 text-gray-800';
    return `<span class="px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${cls}">${level || '—'}</span>`;
}

function statusBadge(status) {
    const map = {
        Approved: 'bg-green-100 text-green-800',
        Review:   'bg-yellow-100 text-yellow-800',
        Rejected: 'bg-red-100 text-red-800',
    };
    const cls = map[status] || 'bg-gray-100 text-gray-800';
    return `<span class="px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${cls}">${status || '—'}</span>`;
}

// ── Export to CSV ─────────────────────────────────────────────────────────────

function exportToCSV() {
    const headers = ['Application ID','Farmer Name','District','Land Size','Trust Score','Risk Level','Status','Date'];
    const rows    = _allApplications.map(app => [
        app.application_id || app.db_id,
        app.farmer_name,
        app.district,
        app.land_size ? app.land_size + ' acres' : '',
        app.trust_score,
        app.risk_level,
        app.status,
        (app.created_at || '').split('T')[0],
    ]);

    let csv = headers.join(',') + '\n';
    rows.forEach(r => { csv += r.map(v => `"${v ?? ''}"`).join(',') + '\n'; });

    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    const a   = Object.assign(document.createElement('a'), { href: url, download: 'applications_export.csv' });
    a.click();
    URL.revokeObjectURL(url);
}

// Add Export CSV button to dashboard table header
document.addEventListener('DOMContentLoaded', () => {
    const headerDiv = document.querySelector('#applicationsTable')
        ?.closest('.bg-white')
        ?.querySelector('.px-6.py-4.border-b');
    if (headerDiv) {
        headerDiv.classList.add('flex', 'justify-between', 'items-center');
        const btn = document.createElement('button');
        btn.className   = 'bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 transition text-sm';
        btn.textContent = 'Export CSV';
        btn.onclick     = exportToCSV;
        headerDiv.appendChild(btn);
    }
});

// ── Utilities ─────────────────────────────────────────────────────────────────

function debounce(fn, ms) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}