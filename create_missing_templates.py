"""
Create missing template files for PneumoScan Pro
"""

from pathlib import Path

def create_missing_templates():
    """Create all missing template files"""
    
    templates_dir = Path('templates')
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("Creating missing template files...")
    print("="*50)
    
    # 1. Create 404.html
    error_404 = templates_dir / '404.html'
    if not error_404.exists():
        error_404_content = '''{% extends "base.html" %}

{% block title %}Page Not Found - PneumoScan Pro{% endblock %}

{% block content %}
<section class="error-page">
    <div class="container">
        <div class="error-content" data-aos="fade-up">
            <div class="error-code">404</div>
            <h1 class="error-title">Page Not Found</h1>
            <p class="error-message">The page you are looking for doesn't exist or has been moved.</p>
            <div class="error-actions">
                <a href="/" class="btn btn-primary">
                    <i class="fas fa-home"></i> Back to Home
                </a>
                <a href="/dashboard" class="btn btn-outline">
                    <i class="fas fa-chart-line"></i> Go to Dashboard
                </a>
            </div>
        </div>
    </div>
</section>

<style>
    .error-page {
        min-height: 70vh;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 4rem 2rem;
    }
    .error-code {
        font-size: 6rem;
        font-weight: 800;
        color: var(--primary-color);
        margin-bottom: 1rem;
    }
    .error-title {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    .error-message {
        color: var(--text-secondary);
        margin-bottom: 2rem;
    }
    .error-actions {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
    }
</style>
{% endblock %}
'''
        with open(error_404, 'w', encoding='utf-8') as f:
            f.write(error_404_content)
        print(f"  Created: {error_404}")
    
    # 2. Create 500.html
    error_500 = templates_dir / '500.html'
    if not error_500.exists():
        error_500_content = '''{% extends "base.html" %}

{% block title %}Server Error - PneumoScan Pro{% endblock %}

{% block content %}
<section class="error-page">
    <div class="container">
        <div class="error-content" data-aos="fade-up">
            <div class="error-code">500</div>
            <h1 class="error-title">Internal Server Error</h1>
            <p class="error-message">Something went wrong on our end. Please try again later.</p>
            <div class="error-actions">
                <a href="/" class="btn btn-primary">
                    <i class="fas fa-home"></i> Back to Home
                </a>
                <button onclick="location.reload()" class="btn btn-outline">
                    <i class="fas fa-sync-alt"></i> Try Again
                </button>
            </div>
        </div>
    </div>
</section>

<style>
    .error-page {
        min-height: 70vh;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 4rem 2rem;
    }
    .error-code {
        font-size: 6rem;
        font-weight: 800;
        color: var(--danger-color);
        margin-bottom: 1rem;
    }
    .error-title {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    .error-message {
        color: var(--text-secondary);
        margin-bottom: 2rem;
    }
    .error-actions {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
    }
</style>
{% endblock %}
'''
        with open(error_500, 'w', encoding='utf-8') as f:
            f.write(error_500_content)
        print(f"  Created: {error_500}")
    
    # 3. Create about.html if missing
    about_html = templates_dir / 'about.html'
    if not about_html.exists():
        about_content = '''{% extends "base.html" %}

{% block title %}About Us - PneumoScan Pro{% endblock %}

{% block content %}
<section class="about-page">
    <div class="container">
        <div class="section-header" data-aos="fade-up">
            <h1 class="section-title">About PneumoScan Pro</h1>
            <p class="section-subtitle">Revolutionizing Medical Imaging with AI</p>
        </div>
        
        <div class="about-content" data-aos="fade-up">
            <div class="about-text">
                <h2>Our Mission</h2>
                <p>PneumoScan Pro is an advanced AI-powered medical imaging platform designed to assist healthcare professionals in detecting multiple respiratory diseases from chest X-rays with high accuracy.</p>
                
                <h2>Technology</h2>
                <p>Our platform uses state-of-the-art deep learning models including DenseNet121, ResNet50, EfficientNet, and Vision Transformers to analyze medical images and provide instant diagnostic insights.</p>
                
                <h2>Key Features</h2>
                <ul>
                    <li>Multi-disease detection (Pneumonia, COVID-19, Tuberculosis, Lung Opacity, Lung Cancer)</li>
                    <li>Ensemble learning for improved accuracy</li>
                    <li>Explainable AI with GradCAM visualizations</li>
                    <li>Automated PDF report generation</li>
                    <li>Patient history tracking</li>
                    <li>3D anatomical visualization</li>
                    <li>Telemedicine integration</li>
                </ul>
            </div>
        </div>
    </div>
</section>
{% endblock %}
'''
        with open(about_html, 'w', encoding='utf-8') as f:
            f.write(about_content)
        print(f"  Created: {about_html}")
    
    # 4. Create disease_predict.html if missing
    disease_predict = templates_dir / 'disease_predict.html'
    if not disease_predict.exists():
        disease_predict_content = '''{% extends "base.html" %}

{% block title %}{{ disease_name }} Detection - PneumoScan Pro{% endblock %}

{% block content %}
<section class="predict-page">
    <div class="container">
        <div class="section-header" data-aos="fade-up">
            <h1 class="section-title">{{ disease_name }} Detection</h1>
            <p class="section-subtitle">{{ disease_description }}</p>
        </div>
        
        <div class="predict-container" data-aos="fade-up">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon"><i class="fas fa-cloud-upload-alt"></i></div>
                <div class="upload-title">Click to upload or drag and drop</div>
                <div class="upload-subtitle">PNG, JPG, JPEG or DICOM (Max 16MB)</div>
                <input type="file" id="imageFile" name="image" accept=".png,.jpg,.jpeg,.dcm" style="display: none;">
            </div>
            
            <div class="file-info" id="fileInfo" style="display: none;">
                <div class="file-details">
                    <i class="fas fa-file-medical"></i>
                    <div>
                        <div class="file-name" id="fileName">No file selected</div>
                        <div class="file-size" id="fileSize"></div>
                    </div>
                    <button type="button" class="remove-file" id="removeFileBtn"><i class="fas fa-times"></i></button>
                </div>
            </div>
            
            <div class="preview-section" id="previewSection" style="display: none;">
                <img class="preview-image" id="previewImage" alt="Preview">
            </div>
            
            <button class="btn btn-primary btn-block" id="analyzeBtn" disabled>
                <i class="fas fa-microscope"></i> Analyze X-Ray
            </button>
        </div>
        
        <div id="resultsContainer"></div>
    </div>
</section>

<script>
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('imageFile');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const previewSection = document.getElementById('previewSection');
    const previewImage = document.getElementById('previewImage');
    const removeFileBtn = document.getElementById('removeFileBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    let selectedFile = null;
    
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) handleFile(e.target.files[0]);
    });
    
    function handleFile(file) {
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        const isValid = validTypes.includes(file.type) || file.name.endsWith('.dcm');
        
        if (!isValid) {
            alert('Please upload a valid image or DICOM file');
            return;
        }
        
        if (file.size > 16 * 1024 * 1024) {
            alert('File size must be less than 16MB');
            return;
        }
        
        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.style.display = 'block';
        
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewSection.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
        
        analyzeBtn.disabled = false;
    }
    
    removeFileBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        fileInfo.style.display = 'none';
        previewSection.style.display = 'none';
        analyzeBtn.disabled = true;
    });
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) {
            alert('Please select an X-ray image');
            return;
        }
        
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('disease', '{{ disease_id }}');
        formData.append('patient_name', 'Demo Patient');
        formData.append('patient_age', '45');
        formData.append('enable_xai', 'true');
        
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.success) {
                sessionStorage.setItem('pneumoscan_results', JSON.stringify(data.data));
                window.location.href = '/results/' + data.data.prediction_id;
            } else {
                alert('Error: ' + data.message);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to analyze image. Please try again.');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-microscope"></i> Analyze X-Ray';
        }
    });
</script>

<style>
    .predict-container {
        max-width: 600px;
        margin: 0 auto;
        background: var(--card-bg);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid var(--border-color);
    }
    .upload-area {
        border: 2px dashed var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .upload-area:hover {
        border-color: var(--primary-color);
        background: rgba(74,144,226,0.05);
    }
    .upload-area.dragover {
        border-color: var(--secondary-color);
        background: rgba(80,227,194,0.05);
    }
    .upload-icon {
        font-size: 3rem;
        color: var(--primary-color);
        margin-bottom: 1rem;
    }
    .preview-image {
        max-width: 100%;
        max-height: 300px;
        border-radius: 12px;
        margin-top: 1rem;
    }
    .btn-block {
        width: 100%;
        margin-top: 1rem;
    }
</style>
{% endblock %}
'''
        with open(disease_predict, 'w', encoding='utf-8') as f:
            f.write(disease_predict_content)
        print(f"  Created: {disease_predict}")
    
    print("\n" + "="*50)
    print("All missing template files created successfully!")
    print("="*50)

if __name__ == '__main__':
    create_missing_templates()