// Complete JavaScript for authentication, API calls, and UI interactions

class LungScanApp {
    constructor() {
        this.token = localStorage.getItem('authToken');
        this.user = JSON.parse(localStorage.getItem('user') || '{}');
        this.currentReportId = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkAuthentication();
        this.loadUserProfile();
    }

    setupEventListeners() {
        // Authentication forms
        const loginForm = document.getElementById('loginForm');
        const registerForm = document.getElementById('registerForm');
        const analysisForm = document.getElementById('analysisForm');
        const logoutBtn = document.getElementById('logoutBtn');

        if (loginForm) loginForm.addEventListener('submit', (e) => this.handleLogin(e));
        if (registerForm) registerForm.addEventListener('submit', (e) => this.handleRegister(e));
        if (analysisForm) analysisForm.addEventListener('submit', (e) => this.handleAnalysis(e));
        if (logoutBtn) logoutBtn.addEventListener('click', () => this.handleLogout());

        // Tab navigation
        const tabLinks = document.querySelectorAll('[data-tab]');
        tabLinks.forEach(link => {
            link.addEventListener('click', (e) => this.switchTab(e));
        });

        // File upload handling
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }
    }

    async handleLogin(e) {
        e.preventDefault();
        const formData = new FormData(e.target);
        
        try {
            this.showLoading('Signing in...');
            const response = await fetch('/api/login', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Login failed');
            }

            const data = await response.json();
            this.token = data.access_token;
            this.user = data.user;

            localStorage.setItem('authToken', this.token);
            localStorage.setItem('user', JSON.stringify(this.user));

            window.location.href = 'dashboard.html';
        } catch (error) {
            this.showError('Login failed. Please check your credentials.');
        } finally {
            this.hideLoading();
        }
    }

    async handleAnalysis(e) {
        e.preventDefault();
        if (!this.token) {
            this.showError('Please log in to analyze images');
            return;
        }

        const formData = new FormData(e.target);
        const fileInput = document.getElementById('fileInput');
        
        if (!fileInput.files[0]) {
            this.showError('Please select an X-ray image');
            return;
        }

        formData.append('file', fileInput.files[0]);

        try {
            this.showLoading('Analyzing X-ray...');
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.token}`
                },
                body: formData
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const result = await response.json();
            this.displayResults(result);
            this.currentReportId = result.report_id;
        } catch (error) {
            this.showError('Analysis failed. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    displayResults(result) {
        // Update UI with results
        document.getElementById('diagnosisClass').textContent = result.class;
        document.getElementById('confidenceValue').textContent = `${result.percentage.toFixed(1)}%`;
        document.getElementById('diagnosisDescription').textContent = result.description;
        
        // Update risk level
        const riskElement = document.getElementById('riskLevel');
        riskElement.innerHTML = result.risk_level === 'LOW' 
            ? '<span class="risk-low">LOW RISK</span>'
            : '<span class="risk-high">HIGH RISK</span>';

        // Show results section
        document.getElementById('resultsSection').style.display = 'block';
    }

    async downloadPDFReport() {
        if (!this.currentReportId) return;
        
        try {
            const response = await fetch(`/api/reports/${this.currentReportId}/pdf`, {
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `Lung_Report_${this.currentReportId}.pdf`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }
        } catch (error) {
            this.showError('Failed to download report');
        }
    }

    // ... other methods for registration, logout, tab management, etc.
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new LungScanApp();
});