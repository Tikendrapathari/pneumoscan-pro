#!/usr/bin/env python3
"""
PneumoScan Pro - Complete Setup Script
Multi-Disease Medical Imaging Platform
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

class PneumoScanSetup:
    def __init__(self):
        self.project_name = "PneumoScan Pro"
        self.version = "2.0.0"
        self.root_dir = Path(__file__).parent.absolute()
        
    def print_banner(self):
        banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████  ███    ██ ███████ ██    ██ ███    ███  ██████      ║
║   ██   ██ ████   ██ ██      ██    ██ ████  ████ ██           ║
║   ██████  ██ ██  ██ █████   ██    ██ ██ ████ ██ ██           ║
║   ██      ██  ██ ██ ██      ██    ██ ██  ██  ██ ██           ║
║   ██      ██   ████ ███████  ██████  ██      ██  ██████      ║
║                                                              ║
║                   MULTI-DISEASE DETECTION                   ║
║                         Version {self.version}                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(f"\n📁 Installation Directory: {self.root_dir}")
        print(f"🐍 Python Version: {platform.python_version()}")
        print(f"💻 Platform: {platform.system()} {platform.release()}\n")

    def create_directory_structure(self):
        """Create complete folder structure"""
        print("\n📂 Creating Directory Structure...")
        
        folders = [
            'models',
            'datasets',
            'training',
            'utils',
            'static/css',
            'static/js',
            'static/images/disease-samples',
            'static/images/backgrounds',
            'static/fonts',
            'static/reports',
            'templates',
            'config',
            'logs',
            'uploads'
        ]
        
        for folder in folders:
            folder_path = self.root_dir / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {folder}/")

    def create_init_files(self):
        """Create __init__.py files for Python packages"""
        init_folders = ['utils', 'training', 'config']
        
        for folder in init_folders:
            init_file = self.root_dir / folder / '__init__.py'
            if not init_file.exists():
                with open(init_file, 'w') as f:
                    f.write(f'# {folder} package\n')
                print(f"   ✅ Created: {folder}/__init__.py")

    def create_requirements(self):
        """Generate requirements.txt with all dependencies"""
        requirements = """# Core
flask==2.3.3
flask-cors==4.0.0
python-dotenv==1.0.0

# Deep Learning
tensorflow==2.13.0
keras==2.13.0
torch==2.0.1
torchvision==0.15.2

# Data Processing
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
scikit-image==0.21.0
opencv-python==4.8.0.74
pillow==10.0.0

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0

# Medical Imaging
pydicom==2.4.3
nibabel==5.1.0
medpy==0.4.0
radiomics==3.0.1

# Image Processing
scipy==1.11.1
imageio==2.31.1
imageio-ffmpeg==0.4.8

# PDF Generation
reportlab==4.0.4
weasyprint==59.0

# Web Scraping / Download
requests==2.31.0
beautifulsoup4==4.12.2
tqdm==4.65.0

# Utilities
joblib==1.3.1
colorama==0.4.6
tabulate==0.9.0
termcolor==2.3.0

# Monitoring
psutil==5.9.5
watchdog==3.0.0

# Database (optional)
sqlalchemy==2.0.19
alembic==1.11.1

# Testing
pytest==7.4.0
pytest-cov==4.1.0

# Deployment
gunicorn==21.2.0
waitress==2.1.2
"""
        with open(self.root_dir / 'requirements.txt', 'w') as f:
            f.write(requirements)
        print("\n📦 Created: requirements.txt")

    def create_env_file(self):
        """Create .env file with configuration"""
        env_content = """# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=your-secret-key-here-change-in-production

# Server Configuration
HOST=0.0.0.0
PORT=5000

# Database (optional)
DATABASE_URL=sqlite:///pneumoscan.db

# Model Paths
PNEUMONIA_MODEL=models/pneumonia_model.h5
COVID_MODEL=models/covid19_model.h5
TB_MODEL=models/tuberculosis_model.h5
LUNG_OPACITY_MODEL=models/lung_opacity_model.h5

# API Keys
SENDGRID_API_KEY=your-sendgrid-key
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token

# File Upload
MAX_CONTENT_LENGTH=16777216  # 16MB
UPLOAD_FOLDER=uploads
ALLOWED_EXTENSIONS=png,jpg,jpeg,dcm

# Report Settings
REPORT_FOLDER=static/reports
REPORT_TEMPLATE=templates/report_template.html

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
"""
        with open(self.root_dir / '.env', 'w') as f:
            f.write(env_content)
        print("🔐 Created: .env")

    def create_gitignore(self):
        """Create .gitignore file"""
        gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
pythonenv*

# Flask
instance/
.webassets-cache
.env
.flaskenv

# Models
*.h5
*.hdf5
*.pkl
*.joblib
*.pt
*.pth

# Datasets
datasets/*
!datasets/dataset_info.md

# Uploads
uploads/*
!uploads/.gitkeep

# Reports
static/reports/*.pdf
!static/reports/.gitkeep

# Logs
logs/*
!logs/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Distribution
dist/
build/
*.egg-info/

# Docker
.dockerignore
Dockerfile
docker-compose.yml

# Misc
*.db
*.sqlite3
*.log
*.bak
"""
        with open(self.root_dir / '.gitignore', 'w') as f:
            f.write(gitignore)
        print("🔒 Created: .gitignore")

    def install_dependencies(self):
        """Install Python dependencies"""
        print("\n📥 Installing Dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--upgrade", "pip"
            ])
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "-r", "requirements.txt"
            ])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            print("\nRun manually: pip install -r requirements.txt")

    def create_readme(self):
        """Create README.md file"""
        readme = f"""# {self.project_name} 🏥

Advanced Multi-Disease Medical Imaging Platform using Deep Learning.

## ✨ Features

- 🔬 Detect 4 diseases: Pneumonia, COVID-19, Tuberculosis, Lung Opacity
- 🎨 Modern, responsive UI with medical animations
- 📊 Real-time disease probability visualization
- 📄 Automated PDF report generation
- 🔍 Chest X-ray validation
- 📈 Training history and model performance metrics
- 🌙 Dark/Light mode support
- 📱 Mobile responsive design

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pneumoscan-pro.git
cd pneumoscan-pro"""