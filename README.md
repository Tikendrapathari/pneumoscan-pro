# 🫁 PneumoScan Pro - AI-Powered Chest X-Ray Analysis

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Flask](https://img.shields.io/badge/Flask-2.3-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Render](https://img.shields.io/badge/Deploy-Render-purple)

> **AI-powered medical imaging platform that detects 5 lung diseases from chest X-rays with 91.6% average accuracy in under 5 seconds.**

[Live Demo](https://pneumoscan-pro.onrender.com) | [Report Issue](https://github.com/TikendraPathe/pneumoscan-pro/issues)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Diseases & Accuracy](#-diseases--accuracy)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Deployment](#-deployment)
- [Screenshots](#-screenshots)
- [Future Scope](#-future-scope)
- [Contributors](#-contributors)
- [License](#-license)

---

## 📋 Overview

**PneumoScan Pro** is an AI-powered web platform that detects **5 lung diseases** from chest X-ray images. It uses state-of-the-art deep learning models (VGG16, DenseNet121) with explainable AI (GradCAM) to provide fast, accurate, and interpretable diagnosis.

### Problem Statement

| Problem | Our Solution |
|---------|--------------|
| Manual X-ray analysis takes 15-20 minutes | ⚡ **<5 seconds** analysis |
| Only 0.9 radiologist per 100,000 people in India | 🤖 **AI assistance** anywhere |
| Doctors don't trust black-box AI | 🔥 **Explainable AI** heatmaps |
| 65% rural population lacks specialist access | 🌐 **Web-based** accessible platform |

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🫁 **Multi-Disease Detection** | 5 diseases from single X-ray |
| 🔥 **Explainable AI** | GradCAM heatmaps showing affected areas |
| 📊 **Severity Classification** | Mild → Moderate → Severe → Critical |
| 🩻 **3D Visualization** | Interactive lung anatomy (Three.js) |
| 🎥 **Telemedicine** | Video consultation + AI assistant |
| 📄 **PDF Reports** | Downloadable medical reports |
| 🌐 **Multi-Language** | English, Hindi, Spanish, French, German |
| 🌙 **Dark Mode** | Eye-friendly interface |
| 📈 **Analytics Dashboard** | Real-time statistics & charts |

---

## 📊 Diseases & Accuracy

| Disease | Model | Accuracy | Recall | F1-Score |
|---------|-------|----------|--------|----------|
| Pneumonia | VGG16 | **93%** | 94% | 93% |
| COVID-19 | DenseNet121 | **94%** | 95% | 94% |
| Tuberculosis | DenseNet121 | **98%** | 99% | 98% |
| Lung Opacity | DenseNet121 | **89%** | 90% | 89% |
| Lung Cancer | DenseNet121 | **84%** | 100% | 92% |
| **Average** | - | **91.6%** | **95.6%** | **93.2%** |

### Severity Levels

| Severity | Confidence | Action |
|----------|------------|--------|
| Mild | 50-69% | Monitor |
| Moderate | 70-84% | Outpatient care |
| Severe | 85-94% | Hospital admission |
| Critical | 95-100% | ICU required |

---

## 🛠️ Technology Stack

### Backend
| Technology | Purpose |
|------------|---------|
| Python 3.12 | Core programming language |
| Flask 2.3 | Web framework |
| TensorFlow 2.16 | Deep learning |
| Keras | Model API |
| SQLAlchemy | ORM for database |

### Frontend
| Technology | Purpose |
|------------|---------|
| HTML5 | Structure |
| CSS3 | Styling |
| JavaScript | Interactivity |
| Three.js | 3D visualization |
| Chart.js | Charts & graphs |

### Database & Deployment
| Technology | Purpose |
|------------|---------|
| SQLite | Development database |
| PostgreSQL | Production database |
| Gunicorn | WSGI server |
| Render.com | Cloud deployment |

---

## 📂 Project Structure
