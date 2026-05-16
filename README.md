# 🫁 PneumoScan Pro - AI-Powered Lung Disease Detection System

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![Flask](https://img.shields.io/badge/Flask-2.0+-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Diseases & Accuracy](#-diseases--accuracy)
4. [System Architecture](#-system-architecture)
5. [Technology Stack](#-technology-stack)
6. [Project Structure](#-project-structure)
7. [Dataset Information](#-dataset-information)
8. [Installation Guide](#-installation-guide)
9. [Usage Guide](#-usage-guide)
10. [API Endpoints](#-api-endpoints)
11. [Configuration](#-configuration)
12. [Performance Metrics](#-performance-metrics)
13. [Screenshots](#-screenshots)
14. [Troubleshooting](#-troubleshooting)
15. [Future Work](#-future-work)
16. [Contributors](#-contributors)
17. [License](#-license)
18. [Contact](#-contact)
19. [Disclaimer](#-disclaimer)

---

## 📌 Project Overview

**PneumoScan Pro** is an advanced **AI-powered medical imaging platform** that detects **5 lung diseases** from chest X-ray images with high accuracy (91.6% average). The system uses state-of-the-art deep learning models (DenseNet121, EfficientNet) with explainable AI (GradCAM) and telemedicine integration.

### 🎯 Problem Statement

- **Delayed Diagnosis**: Manual analysis takes 15-20 minutes
- **Expert Shortage**: Only 0.9 radiologist per 100,000 people in India
- **Multiple Diseases**: Patients may have overlapping conditions
- **Black Box AI**: Doctors don't trust unexplained predictions
- **Remote Access**: 65% rural population lacks specialist access

### ✅ Our Solution

PneumoScan Pro addresses all these challenges with:
- ⚡ **<5 second** analysis time
- 🤖 **91.6% average accuracy**
- 🔍 **Explainable AI** with GradCAM heatmaps
- 🏥 **Telemedicine** with AI assistant
- 🌐 **Web-based** accessible platform

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🫁 **Multi-Disease Detection** | Detects 5 lung diseases from single X-ray |
| 🔥 **Explainable AI (GradCAM)** | Heatmaps showing AI decision regions |
| 📊 **Severity Classification** | Mild, Moderate, Severe, Critical levels |
| 🎥 **Telemedicine Platform** | Video consultation with AI assistant |
| 📄 **PDF Report Generation** | Downloadable medical reports |
| 🩻 **3D Medical Visualization** | Interactive lung anatomy model |
| 🌐 **Multi-Language Support** | English, Hindi, Spanish, French, German |
| 🌙 **Dark Mode** | Eye-friendly interface |
| 👤 **User Authentication** | Login/Register system |
| 📈 **Analytics Dashboard** | Real-time statistics and charts |

---

## 📊 Diseases & Accuracy

### Disease-wise Performance

| Disease | Accuracy | Precision | Recall | F1-Score | AUC |
|---------|----------|-----------|--------|----------|-----|
| **Pneumonia** | 93% | 92% | 94% | 93% | 0.97 |
| **COVID-19** | 94% | 93% | 95% | 94% | 0.98 |
| **Tuberculosis** | 98% | 97% | 99% | 98% | 0.99 |
| **Lung Opacity** | 89% | 88% | 90% | 89% | 0.94 |
| **Lung Cancer** | 84% | 85% | 100% | 92% | 0.91 |
| **Average** | **91.6%** | **91%** | **95.6%** | **93.2%** | **0.958** |

### Severity Levels

| Severity | Confidence Range | Action Required |
|----------|------------------|-----------------|
| Mild | 50-69% | Monitor |
| Moderate | 70-84% | Outpatient care |
| Severe | 85-94% | Hospital admission |
| Critical | 95-100% | ICU required |

---

## 🏗️ System Architecture
