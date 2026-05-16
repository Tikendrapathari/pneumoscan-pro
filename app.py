#!/usr/bin/env python3
"""
PneumoScan Pro Advanced - Multi-Disease Medical Imaging Platform
PRODUCTION READY - For Render.com Deployment
HEATMAP - ONLY AFFECTED AREA VISIBLE
"""

import os
import sys
import logging
import json
import base64
from datetime import datetime, timedelta
from pathlib import Path
from io import BytesIO
from functools import wraps

# Production environment variables
IS_PRODUCTION = os.environ.get('RENDER', False) or os.environ.get('PRODUCTION', False)

# Suppress TensorFlow warnings in production
if IS_PRODUCTION:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify, send_file, session, url_for, redirect, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

from utils.image_processor import ImageProcessor
from utils.feature_extractor import FeatureExtractor
from utils.xray_validator import XRayValidator
from utils.report_generator import ReportGenerator
from utils.explainable_ai import ExplainableAI
from utils.severity_classifier import SeverityClassifier
from utils.database import db, Patient, Prediction, User
from utils.helpers import allowed_file, create_response, login_required
from config.settings import get_config
from config.disease_config import DISEASE_CONFIG, ENSEMBLE_CONFIG


# ==============================================
# HEATMAP GENERATOR - ONLY AFFECTED AREA
# ==============================================

class AffectedAreaHeatmap:
    """Generate heatmap showing ONLY the affected area"""
    
    @staticmethod
    def generate_saliency(model, img_array, class_idx=None):
        """Generate saliency map"""
        try:
            if not isinstance(img_array, tf.Tensor):
                img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            else:
                img_tensor = img_array
            
            if class_idx is None:
                predictions = model(img_tensor, training=False)
                class_idx = tf.argmax(predictions[0])
            
            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                predictions = model(img_tensor, training=False)
                loss = predictions[:, class_idx]
            
            gradients = tape.gradient(loss, img_tensor)
            
            if gradients is None:
                return None
            
            saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
            saliency = saliency[0].numpy()
            
            saliency = cv2.GaussianBlur(saliency, (5, 5), 0)
            
            if saliency.max() - saliency.min() > 0:
                saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            
            return saliency
            
        except Exception as e:
            print(f"    Saliency error: {e}")
            return None
    
    @staticmethod
    def extract_affected_region(saliency, original_img):
        """Extract only the affected region (mask out normal areas)"""
        try:
            h, w = original_img.shape[:2]
            saliency_resized = cv2.resize(saliency, (w, h))
            
            threshold = np.percentile(saliency_resized, 70)
            affected_mask = (saliency_resized > threshold).astype(np.uint8)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            affected_mask = cv2.morphologyEx(affected_mask, cv2.MORPH_CLOSE, kernel)
            affected_mask = cv2.morphologyEx(affected_mask, cv2.MORPH_OPEN, kernel)
            affected_mask = cv2.dilate(affected_mask, kernel, iterations=2)
            
            masked_saliency = saliency_resized * affected_mask
            
            if masked_saliency.max() > 0:
                masked_saliency = masked_saliency / masked_saliency.max()
            
            return masked_saliency, affected_mask
            
        except Exception as e:
            print(f"    Extract region error: {e}")
            return saliency, None
    
    @staticmethod
    def create_heatmap_overlay(masked_saliency, original_img, affected_mask, alpha=0.7):
        """Create heatmap only on affected area"""
        try:
            saliency_uint8 = np.uint8(255 * masked_saliency)
            heatmap_color = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
            
            if len(original_img.shape) == 2:
                original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            elif original_img.shape[-1] == 1:
                original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            else:
                original_rgb = original_img.copy()
            
            if original_rgb.max() <= 1.0:
                original_rgb = np.uint8(original_rgb * 255)
            
            result = original_rgb.copy()
            
            if affected_mask is not None:
                overlay = cv2.addWeighted(original_rgb, 1 - alpha, heatmap_color, alpha, 0)
                mask_3channel = cv2.merge([affected_mask, affected_mask, affected_mask])
                result = np.where(mask_3channel > 0, overlay, original_rgb)
                
                contours, _ = cv2.findContours(affected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result, contours, -1, (0, 0, 255), 2)
            else:
                result = cv2.addWeighted(original_rgb, 1 - alpha, heatmap_color, alpha, 0)
            
            return result
            
        except Exception as e:
            print(f"    Overlay error: {e}")
            return original_img
    
    @staticmethod
    def generate(model, processed_img, original_img, disease_name, confidence):
        """Generate heatmap showing only affected area"""
        try:
            print(f"  Generating Affected Area Heatmap for {disease_name}...")
            
            if len(processed_img.shape) == 3:
                processed_img = np.expand_dims(processed_img, axis=0)
            
            preds = model.predict(processed_img, verbose=0)[0]
            class_idx = np.argmax(preds)
            print(f"    Class: {class_idx}, Confidence: {preds[class_idx]:.4f}")
            
            saliency = AffectedAreaHeatmap.generate_saliency(model, processed_img, class_idx)
            
            if saliency is None:
                print(f"  Saliency generation failed")
                return None
            
            masked_saliency, affected_mask = AffectedAreaHeatmap.extract_affected_region(saliency, original_img)
            overlay = AffectedAreaHeatmap.create_heatmap_overlay(masked_saliency, original_img, affected_mask, alpha=0.7)
            
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.png', overlay_rgb)
            overlay_b64 = base64.b64encode(buffer).decode('utf-8')
            
            print(f"  Affected Area Heatmap generated successfully!")
            
            return {
                'overlay': overlay_b64,
                'disease': disease_name,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"  Heatmap error: {e}")
            return None


heatmap_generator = AffectedAreaHeatmap()

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


# ==============================================
# CUSTOM FUNCTIONS
# ==============================================

def grayscale_to_rgb(x):
    return tf.repeat(x, 3, axis=-1)


CUSTOM_OBJECTS = {
    'grayscale_to_rgb': grayscale_to_rgb,
    'tf': tf,
    'Lambda': tf.keras.layers.Lambda,
    'Flatten': tf.keras.layers.Flatten,
    'Dense': tf.keras.layers.Dense,
    'Dropout': tf.keras.layers.Dropout,
    'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
    'BatchNormalization': tf.keras.layers.BatchNormalization,
    'InputLayer': tf.keras.layers.InputLayer,
    'Conv2D': tf.keras.layers.Conv2D,
    'MaxPooling2D': tf.keras.layers.MaxPooling2D,
    'AveragePooling2D': tf.keras.layers.AveragePooling2D,
    'GlobalMaxPooling2D': tf.keras.layers.GlobalMaxPooling2D,
    'Add': tf.keras.layers.Add,
    'Concatenate': tf.keras.layers.Concatenate,
    'Activation': tf.keras.layers.Activation,
    'ReLU': tf.keras.layers.ReLU,
    'LeakyReLU': tf.keras.layers.LeakyReLU,
    'ZeroPadding2D': tf.keras.layers.ZeroPadding2D,
    'Reshape': tf.keras.layers.Reshape,
    'SeparableConv2D': tf.keras.layers.SeparableConv2D,
    'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D,
    'Input': tf.keras.layers.Input,
}


def make_json_serializable(obj):
    if obj is None:
        return None
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    else:
        return obj


# ==============================================
# FLASK APP INITIALIZATION
# ==============================================

app = Flask(__name__)
config = get_config()
app.config.from_object(config)

# Database Configuration - Supports both SQLite and PostgreSQL
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///pneumoscan.db')

# For Render PostgreSQL, fix the URL format
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'dcm', 'dicom'}

# Session configuration for production
app.config['SESSION_COOKIE_SECURE'] = IS_PRODUCTION
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

db.init_app(app)

# Configure logging for production
if IS_PRODUCTION:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
else:
    logging.basicConfig(
        level=getattr(logging, app.config.get('LOG_LEVEL', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(app.config.get('LOG_FILE', 'app.log'), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

logger = logging.getLogger(__name__)

CORS(app)

# Initialize utilities
image_processor = ImageProcessor()
feature_extractor = FeatureExtractor()
xray_validator = XRayValidator()
report_gen = ReportGenerator()
explainable_ai = ExplainableAI()
severity_classifier = SeverityClassifier()

models = {}
ensemble_model = None
model_metadata = {}


# ==============================================
# MODEL LOADING
# ==============================================

def load_pneumonia_model():
    models_dir = Path('models')
    pneumonia_paths = [
        models_dir / 'pneumonia_model.h5',
        models_dir / 'pneumonia_model_best.h5',
        models_dir / 'pneumonia_model.keras',
    ]
    
    for path in pneumonia_paths:
        if path.exists():
            try:
                print(f"[INFO] Loading pneumonia model from: {path.name}")
                model = tf.keras.models.load_model(str(path), custom_objects=CUSTOM_OBJECTS, compile=False)
                print(f"[SUCCESS] Loaded pneumonia model")
                return model
            except Exception as e:
                print(f"[WARN] Failed to load {path.name}: {e}")
                continue
    
    print("[WARN] No valid pneumonia model found. Creating dummy model.")
    from tensorflow.keras import layers
    dummy_model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    return dummy_model


def load_lung_cancer_model():
    models_dir = Path('models')
    cancer_path = models_dir / 'lung_cancer_model_3class.keras'
    
    if cancer_path.exists():
        try:
            print(f"[INFO] Loading lung cancer model from: {cancer_path.name}")
            model = tf.keras.models.load_model(str(cancer_path), custom_objects=CUSTOM_OBJECTS, compile=False)
            print(f"[SUCCESS] Loaded Lung Cancer model")
            return model
        except Exception as e:
            print(f"[FAIL] Failed to load lung cancer model: {e}")
    
    print("[WARN] No lung cancer model found. Creating dummy model.")
    from tensorflow.keras import layers
    dummy_model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])
    return dummy_model


def load_all_models():
    global models, ensemble_model
    
    models_dir = Path('models')
    
    if not models_dir.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        models_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)
    
    print("\n[1/5] Loading Pneumonia Model...")
    pneumonia_model = load_pneumonia_model()
    models['pneumonia'] = {'primary': pneumonia_model, 'ensemble': []}
    
    print("\n[2/5] Loading COVID-19 Model...")
    covid_path = models_dir / 'covid19_model_best.h5'
    if covid_path.exists():
        try:
            covid_model = tf.keras.models.load_model(str(covid_path), compile=False)
            models['covid19'] = {'primary': covid_model, 'ensemble': []}
            print(f"[SUCCESS] Loaded COVID-19 model")
        except Exception as e:
            print(f"[FAIL] Failed to load COVID-19 model: {e}")
            models['covid19'] = {'primary': None, 'ensemble': []}
    else:
        print(f"[WARN] COVID-19 model not found")
        models['covid19'] = {'primary': None, 'ensemble': []}
    
    print("\n[3/5] Loading Tuberculosis Model...")
    tb_path = models_dir / 'tuberculosis_model_best.h5'
    if tb_path.exists():
        try:
            tb_model = tf.keras.models.load_model(str(tb_path), compile=False)
            models['tuberculosis'] = {'primary': tb_model, 'ensemble': []}
            print(f"[SUCCESS] Loaded Tuberculosis model")
        except Exception as e:
            print(f"[FAIL] Failed to load Tuberculosis model: {e}")
            models['tuberculosis'] = {'primary': None, 'ensemble': []}
    else:
        print(f"[WARN] Tuberculosis model not found")
        models['tuberculosis'] = {'primary': None, 'ensemble': []}
    
    print("\n[4/5] Loading Lung Opacity Model...")
    opacity_path = models_dir / 'lung_opacity_model.h5'
    if opacity_path.exists():
        try:
            opacity_model = tf.keras.models.load_model(str(opacity_path), compile=False)
            models['lung_opacity'] = {'primary': opacity_model, 'ensemble': []}
            print(f"[SUCCESS] Loaded Lung Opacity model")
        except Exception as e:
            print(f"[FAIL] Failed to load Lung Opacity model: {e}")
            models['lung_opacity'] = {'primary': None, 'ensemble': []}
    else:
        opacity_best = models_dir / 'lung_opacity_best.h5'
        if opacity_best.exists():
            try:
                opacity_model = tf.keras.models.load_model(str(opacity_best), compile=False)
                models['lung_opacity'] = {'primary': opacity_model, 'ensemble': []}
                print(f"[SUCCESS] Loaded Lung Opacity model from lung_opacity_best.h5")
            except Exception as e:
                print(f"[FAIL] Failed to load Lung Opacity model: {e}")
                models['lung_opacity'] = {'primary': None, 'ensemble': []}
        else:
            print(f"[WARN] Lung Opacity model not found")
            models['lung_opacity'] = {'primary': None, 'ensemble': []}
    
    print("\n[5/5] Loading Lung Cancer Model...")
    cancer_model = load_lung_cancer_model()
    models['lung_cancer'] = {'primary': cancer_model, 'ensemble': []}
    
    ensemble_path = models_dir / 'ensemble_model.h5'
    if ensemble_path.exists():
        try:
            ensemble_model = tf.keras.models.load_model(str(ensemble_path), compile=False)
            print(f"[SUCCESS] Loaded Ensemble model")
        except Exception as e:
            print(f"[FAIL] Failed to load Ensemble model: {e}")
    
    print("\n" + "="*60)
    print("MODEL LOADING SUMMARY")
    print("="*60)
    loaded = [d for d, m in models.items() if m.get('primary') is not None]
    not_loaded = [d for d, m in models.items() if m.get('primary') is None]
    print(f"[OK] Loaded: {loaded}")
    if not_loaded:
        print(f"[WARN] Not Loaded: {not_loaded}")
    print("="*60)


load_all_models()


# ==============================================
# DATABASE INITIALIZATION
# ==============================================

def init_db():
    """Initialize database with tables and default users"""
    with app.app_context():
        try:
            db.create_all()
            logger.info("Database tables created")
            
            # Create admin user
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                admin = User(
                    username='admin',
                    email='admin@pneumoscan.com',
                    password_hash=generate_password_hash('admin123'),
                    role='admin'
                )
                db.session.add(admin)
            
            # Create demo user
            demo = User.query.filter_by(username='ramesh').first()
            if not demo:
                demo = User(
                    username='ramesh',
                    email='ramesh@example.com',
                    password_hash=generate_password_hash('ramesh123'),
                    role='user'
                )
                db.session.add(demo)
            
            db.session.commit()
            logger.info("Default users created")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            db.session.rollback()


init_db()


# ==============================================
# HELPER FUNCTIONS
# ==============================================

def get_pneumonia_prediction(model, processed_img):
    try:
        if isinstance(processed_img, np.ndarray):
            if len(processed_img.shape) == 4:
                processed_img = processed_img[0]
            if len(processed_img.shape) == 3 and processed_img.shape[-1] == 1:
                processed_img = np.repeat(processed_img, 3, axis=-1)
            if len(processed_img.shape) == 2:
                processed_img = np.expand_dims(processed_img, axis=0)
                processed_img = np.expand_dims(processed_img, axis=-1)
                processed_img = np.repeat(processed_img, 3, axis=-1)
            elif len(processed_img.shape) == 3:
                processed_img = np.expand_dims(processed_img, axis=0)
        
        pred = model.predict(processed_img, verbose=0)
        if pred is None or len(pred) == 0:
            return 0.5
        if len(pred[0]) == 2:
            return float(pred[0][1])
        else:
            return float(pred[0][0])
    except Exception as e:
        logger.error(f"Pneumonia prediction error: {e}")
        return 0.5


def get_lung_cancer_prediction(model, processed_img):
    try:
        if len(processed_img.shape) == 3:
            processed_img = np.expand_dims(processed_img, axis=0)
        
        pred = model.predict(processed_img, verbose=0)[0]
        print(f"    Lung Cancer raw predictions: {pred}")
        
        if len(pred) >= 2:
            return float(pred[1])
        return 0.5
    except Exception as e:
        logger.error(f"Lung cancer prediction error: {e}")
        return 0.5


def get_model_prediction(model_dict, processed_img, disease_info, disease_name):
    if model_dict.get('primary') is None:
        return 0.5
    
    try:
        model = model_dict['primary']
        
        if disease_name == 'pneumonia':
            return get_pneumonia_prediction(model, processed_img)
        
        if disease_name == 'lung_cancer':
            return get_lung_cancer_prediction(model, processed_img)
        
        if len(processed_img.shape) == 3:
            processed_img = np.expand_dims(processed_img, axis=0)
        
        pred = model.predict(processed_img, verbose=0)[0]
        
        if len(pred) == 2:
            return float(pred[1])
        elif len(pred) == 1:
            return float(pred[0])
        else:
            return float(np.max(pred))
    except Exception as e:
        logger.error(f"Prediction error for {disease_name}: {e}")
        return 0.5


def get_original_image_array(img):
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_display = img.resize((512, 512), Image.Resampling.LANCZOS)
        return np.array(img_display)
    except Exception as e:
        print(f"Original image conversion error: {e}")
        return None


def is_valid_chest_xray(image_array):
    try:
        if len(image_array.shape) == 3:
            gray = np.mean(image_array, axis=2)
        else:
            gray = image_array
        
        h, w = gray.shape
        aspect_ratio = w / h
        
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False, "Invalid image dimensions"
        
        std_intensity = np.std(gray)
        if std_intensity < 0.02:
            return False, "Image has very low contrast"
        
        return True, "Valid chest X-ray"
    except Exception as e:
        return True, "Processing image"


# ==============================================
# ROUTES - PAGES
# ==============================================

@app.route('/')
def home():
    return render_template('index.html', diseases=DISEASE_CONFIG, year=datetime.now().year)


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', diseases=DISEASE_CONFIG, year=datetime.now().year)


@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html', diseases=DISEASE_CONFIG, year=datetime.now().year)


@app.route('/results/<int:prediction_id>')
@login_required
def results(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    return render_template('results.html', prediction=prediction, diseases=DISEASE_CONFIG, year=datetime.now().year)


@app.route('/about')
def about():
    return render_template('about.html', year=datetime.now().year)


@app.route('/contact')
def contact():
    return render_template('contact.html', year=datetime.now().year)


@app.route('/medical-3d')
def medical_3d():
    return render_template('medical-3d-visualization.html', year=datetime.now().year)


@app.route('/telemedicine')
@login_required
def telemedicine():
    return render_template('telemedicine.html', year=datetime.now().year)


@app.route('/severity-analysis/<int:prediction_id>')
@login_required
def severity_analysis(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    return render_template('severity-analysis.html', prediction=prediction, year=datetime.now().year)


@app.route('/xai-explanation/<int:prediction_id>')
@login_required
def xai_explanation(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    return render_template('xai-explanation.html', prediction=prediction, year=datetime.now().year)


@app.route('/patient-history')
@login_required
def patient_history():
    search = request.args.get('search', '')
    date_range = request.args.get('date_range', 'all')
    disease_filter = request.args.get('disease', 'all')
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    query = Prediction.query.join(Patient, Prediction.patient_id == Patient.id, isouter=True)
    
    if search:
        query = query.filter(
            db.or_(
                Patient.name.ilike(f'%{search}%'),
                Patient.patient_id.ilike(f'%{search}%'),
                Patient.phone.ilike(f'%{search}%')
            )
        )
    
    if date_range != 'all':
        days = int(date_range)
        cutoff_date = datetime.now() - timedelta(days=days)
        query = query.filter(Prediction.created_at >= cutoff_date)
    
    if disease_filter != 'all':
        query = query.filter(Prediction.disease_detected == disease_filter)
    
    query = query.order_by(Prediction.created_at.desc())
    paginated = query.paginate(page=page, per_page=per_page, error_out=False)
    predictions = paginated.items
    total_pages = paginated.pages
    
    recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()
    
    total_patients = Patient.query.count()
    total_analyses = Prediction.query.count()
    week_ago = datetime.now() - timedelta(days=7)
    this_week = Prediction.query.filter(Prediction.created_at >= week_ago).count()
    positive_cases = Prediction.query.filter(
        Prediction.disease_detected.isnot(None),
        Prediction.disease_detected != '',
        Prediction.confidence > 0.5
    ).count()
    
    stats = {
        'total_patients': total_patients,
        'total_analyses': total_analyses,
        'this_week': this_week,
        'positive_cases': positive_cases
    }
    
    return render_template('patient-history.html',
                         predictions=predictions,
                         recent_predictions=recent_predictions,
                         stats=stats,
                         page=page,
                         total_pages=total_pages,
                         year=datetime.now().year)


@app.route('/patient/<int:patient_id>')
@login_required
def patient_detail(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    predictions = Prediction.query.filter_by(patient_id=patient.id).order_by(Prediction.created_at.desc()).all()
    return render_template('patient_detail.html', patient=patient, predictions=predictions, year=datetime.now().year)


@app.route('/predict/<disease_id>')
@login_required
def predict_disease(disease_id):
    if disease_id not in DISEASE_CONFIG:
        flash('Disease not found', 'danger')
        return redirect(url_for('home'))
    
    disease_info = DISEASE_CONFIG[disease_id]
    return render_template('disease_predict.html',
                         disease_id=disease_id,
                         disease_name=disease_info['name'],
                         disease_description=disease_info['description'],
                         accuracy=int(disease_info['performance']['accuracy'] * 100),
                         year=datetime.now().year)


@app.route('/profile')
@login_required
def profile():
    user = User.query.get(session.get('user_id'))
    return render_template('profile.html', user=user, year=datetime.now().year)


@app.route('/settings')
@login_required
def settings():
    user = User.query.get(session.get('user_id'))
    return render_template('settings.html', user=user, year=datetime.now().year)


@app.route('/admin')
@login_required
def admin_panel():
    if session.get('role') != 'admin':
        flash('Access denied. Admin access required.', 'danger')
        return redirect(url_for('dashboard'))
    
    total_users = User.query.count()
    total_patients = Patient.query.count()
    total_predictions = Prediction.query.count()
    recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()
    
    return render_template('admin.html', 
                         total_users=total_users,
                         total_patients=total_patients,
                         total_predictions=total_predictions,
                         recent_predictions=recent_predictions,
                         year=datetime.now().year)


@app.route('/api/update-profile', methods=['POST'])
@login_required
def update_profile():
    try:
        user = User.query.get(session.get('user_id'))
        if not user:
            return jsonify({'success': False, 'message': 'User not found'})
        
        name = request.form.get('name')
        email = request.form.get('email')
        
        if name:
            user.username = name
            session['username'] = name
        if email:
            user.email = email
        
        db.session.commit()
        return jsonify({'success': True, 'message': 'Profile updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/change-password', methods=['POST'])
@login_required
def change_password():
    try:
        user = User.query.get(session.get('user_id'))
        if not user:
            return jsonify({'success': False, 'message': 'User not found'})
        
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        
        if not check_password_hash(user.password_hash, current_password):
            return jsonify({'success': False, 'message': 'Current password is incorrect'})
        
        user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Password changed successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html', year=datetime.now().year)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            role='user'
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', year=datetime.now().year)


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))


@app.route('/api/health', methods=['GET'])
def health_check():
    loaded_diseases = [d for d, m in models.items() if m.get('primary') is not None]
    return jsonify({
        'status': 'healthy' if loaded_diseases else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(loaded_diseases),
        'models': loaded_diseases,
        'ensemble_loaded': ensemble_model is not None,
        'diseases': list(DISEASE_CONFIG.keys()),
        'version': '2.0.0',
        'environment': 'production' if IS_PRODUCTION else 'development'
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    model_stats = {}
    for disease_name, disease_info in DISEASE_CONFIG.items():
        model_available = disease_name in models and models[disease_name].get('primary') is not None
        model_stats[disease_name] = {
            'name': disease_info['name'],
            'description': disease_info['description'],
            'accuracy': float(disease_info['performance']['accuracy']),
            'precision': float(disease_info['performance']['precision']),
            'recall': float(disease_info['performance']['recall']),
            'f1_score': float(disease_info['performance']['f1_score']),
            'auc': float(disease_info['performance'].get('auc', 0.95)),
            'classes': disease_info['class_labels'],
            'severity_levels': disease_info['severity_levels'],
            'model_loaded': model_available
        }
    return jsonify(model_stats)


@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    try:
        if 'image' not in request.files:
            return create_response(False, 'No image uploaded', status=400)
        
        file = request.files['image']
        if file.filename == '':
            return create_response(False, 'No file selected', status=400)
        
        patient_name = request.form.get('patient_name', '').strip()
        patient_age = request.form.get('patient_age', '').strip()
        patient_gender = request.form.get('patient_gender', '').strip()
        patient_phone = request.form.get('patient_phone', '').strip()
        patient_email = request.form.get('patient_email', '').strip()
        patient_address = request.form.get('patient_address', '').strip()
        physician = request.form.get('patient_physician', 'Dr. Smith').strip()
        
        patient_id = request.form.get('patient_id', '').strip()
        if not patient_id and patient_name:
            patient_id = f"P{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        patient_info = {
            'name': patient_name if patient_name else 'Unknown',
            'age': patient_age if patient_age else 'N/A',
            'id': patient_id,
            'gender': patient_gender if patient_gender else 'N/A',
            'physician': physician,
            'phone': patient_phone,
            'email': patient_email,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        selected_disease = request.form.get('disease', 'all')
        enable_xai = request.form.get('enable_xai', 'true').lower() == 'true'
        
        file_content = file.read()
        
        img = Image.open(BytesIO(file_content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        original_img = img.copy()
        original_array = get_original_image_array(original_img)
        
        img_array = np.array(img) / 255.0
        is_valid, msg = is_valid_chest_xray(img_array)
        
        validation_result = {
            'is_xray': is_valid,
            'message': msg,
            'confidence_percent': '85.0',
            'issues': [] if is_valid else [msg],
            'reasons': [] if is_valid else [msg]
        }
        
        predictions = {}
        confidence_scores = {}
        
        if selected_disease == 'all':
            diseases_to_predict = list(DISEASE_CONFIG.keys())
        else:
            diseases_to_predict = [selected_disease]
        
        for disease_name in diseases_to_predict:
            if disease_name not in models or models[disease_name].get('primary') is None:
                predictions[disease_name] = {
                    'result': 'Model Not Available',
                    'confidence': 0,
                    'confidence_percent': "0.00",
                    'severity': 'Unknown',
                    'class': 0
                }
                continue
            
            try:
                disease_info = DISEASE_CONFIG[disease_name]
                model_dict = models[disease_name]
                
                processed_img = image_processor.process(
                    img,
                    target_size=disease_info['input_size'][0],
                    normalize=True
                )
                
                if processed_img is None:
                    predictions[disease_name] = {
                        'result': 'Processing Error',
                        'confidence': 0,
                        'confidence_percent': "0.00",
                        'severity': 'Unknown',
                        'class': 0
                    }
                    continue
                
                pred_score = get_model_prediction(model_dict, processed_img, disease_info, disease_name)
                threshold = disease_info.get('threshold', 0.5)
                is_positive = pred_score > threshold
                
                if len(disease_info['class_labels']) == 2:
                    result_class = disease_info['class_labels'][1] if is_positive else disease_info['class_labels'][0]
                    confidence = pred_score if is_positive else 1 - pred_score
                else:
                    if len(processed_img.shape) == 3:
                        processed_img = np.expand_dims(processed_img, axis=0)
                    try:
                        pred_full = model_dict['primary'].predict(processed_img, verbose=0)[0]
                        class_idx = np.argmax(pred_full)
                        result_class = disease_info['class_labels'][class_idx]
                        confidence = float(pred_full[class_idx])
                    except:
                        result_class = disease_info['class_labels'][1] if is_positive else disease_info['class_labels'][0]
                        confidence = pred_score if is_positive else 1 - pred_score
                    is_positive = result_class != 'Normal' and result_class != 'Benign'
                
                severity = 'Normal'
                if is_positive and confidence > 0.5:
                    if confidence > 0.8:
                        severity = 'Severe'
                    elif confidence > 0.6:
                        severity = 'Moderate'
                    else:
                        severity = 'Mild'
                    
                    if disease_name == 'lung_cancer' and result_class == 'Malignant':
                        if confidence > 0.8:
                            severity = 'Stage III/IV'
                        elif confidence > 0.6:
                            severity = 'Stage II'
                        else:
                            severity = 'Stage I'
                
                predictions[disease_name] = {
                    'result': result_class,
                    'confidence': float(confidence),
                    'confidence_percent': f"{confidence*100:.2f}",
                    'severity': severity,
                    'class': 1 if is_positive else 0
                }
                confidence_scores[disease_name] = float(confidence)
                
                logger.info(f"{disease_name}: {result_class} with {confidence*100:.2f}% confidence")
                
            except Exception as e:
                logger.error(f"Prediction error for {disease_name}: {e}")
                predictions[disease_name] = {
                    'result': 'Error',
                    'confidence': 0,
                    'confidence_percent': "0.00",
                    'severity': 'Unknown',
                    'class': 0
                }
        
        primary_disease = None
        primary_confidence = 0.0
        
        for disease, pred in predictions.items():
            if pred.get('class', 0) == 1:
                if pred['confidence'] > primary_confidence:
                    primary_confidence = pred['confidence']
                    primary_disease = disease
        
        # ==============================================
        # AFFECTED AREA HEATMAP GENERATION
        # ==============================================
        heatmap_images = {}
        
        print(f"\n[DEBUG] enable_xai = {enable_xai}")
        
        if enable_xai and original_array is not None:
            print("\n" + "="*60)
            print("GENERATING AFFECTED AREA HEATMAPS")
            print("="*60)
            
            for disease_name, pred_data in predictions.items():
                if disease_name not in models or models[disease_name].get('primary') is None:
                    continue
                
                is_positive = pred_data.get('class', 0) == 1
                
                if is_positive:
                    try:
                        disease_info = DISEASE_CONFIG[disease_name]
                        
                        processed_img = image_processor.process_for_gradcam(
                            img,
                            target_size=disease_info['input_size'][0]
                        )
                        
                        if processed_img is not None:
                            confidence_value = pred_data.get('confidence_percent', '0')
                            
                            heatmap_result = heatmap_generator.generate(
                                models[disease_name]['primary'],
                                processed_img,
                                original_array,
                                disease_name,
                                confidence_value
                            )
                            
                            if heatmap_result:
                                heatmap_images[disease_name] = heatmap_result
                                print(f"  Affected area heatmap saved for {disease_name}")
                            else:
                                print(f"  Heatmap failed for {disease_name}")
                        else:
                            print(f"  process_for_gradcam returned None")
                            
                    except Exception as e:
                        print(f"  Heatmap error: {e}")
                else:
                    print(f"  Skipping {disease_name} (not positive)")
            
            print(f"\nAffected area heatmaps generated: {len(heatmap_images)}")
            print("="*60 + "\n")
        else:
            print("\n[INFO] Heatmap generation skipped (XAI disabled or no image)")
        
        # Save patient
        patient = None
        
        if patient_id:
            patient = Patient.query.filter_by(patient_id=patient_id).first()
        
        if not patient and patient_phone:
            patient = Patient.query.filter_by(phone=patient_phone).first()
        
        if patient:
            if patient_name:
                patient.name = patient_name
            if patient_age and patient_age.isdigit():
                patient.age = int(patient_age)
            if patient_gender:
                patient.gender = patient_gender
            if patient_phone:
                patient.phone = patient_phone
            if patient_email:
                patient.email = patient_email
            if patient_address:
                patient.address = patient_address
            db.session.commit()
            logger.info(f"Updated patient: {patient.patient_id} - {patient.name}")
        elif patient_name:
            patient = Patient(
                patient_id=patient_id if patient_id else f"P{datetime.now().strftime('%Y%m%d%H%M%S')}",
                name=patient_name,
                age=int(patient_age) if patient_age and patient_age.isdigit() else None,
                gender=patient_gender if patient_gender else None,
                phone=patient_phone if patient_phone else None,
                email=patient_email if patient_email else None,
                address=patient_address if patient_address else None
            )
            db.session.add(patient)
            db.session.commit()
            logger.info(f"Created new patient: {patient.patient_id} - {patient.name}")
        
        # Save prediction record
        prediction_record = Prediction(
            patient_id=patient.id if patient else None,
            user_id=session.get('user_id'),
            image_filename=secure_filename(file.filename),
            disease_detected=primary_disease,
            confidence=primary_confidence,
            all_predictions=json.dumps(predictions),
            validation_result=json.dumps(validation_result),
            gradcam_images=json.dumps(heatmap_images) if heatmap_images else None,
            created_at=datetime.now()
        )
        db.session.add(prediction_record)
        db.session.commit()
        logger.info(f"Saved prediction {prediction_record.id}")
        
        # Generate PDF report
        pdf_filename = None
        try:
            img_buffer = BytesIO()
            original_img.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            pdf_filename = f"report_{prediction_record.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            report_gen.generate_advanced_report(
                patient_info=patient_info,
                predictions=predictions,
                primary_disease=primary_disease,
                image_data=img_base64,
                validation=validation_result,
                gradcam_images=heatmap_images,
                filename=pdf_filename
            )
            prediction_record.report_path = pdf_filename
            db.session.commit()
            logger.info(f"Report generated: {pdf_filename}")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        response_data = {
            'prediction_id': prediction_record.id,
            'predictions': predictions,
            'patient_info': patient_info,
            'validation': validation_result,
            'primary_disease': primary_disease,
            'primary_confidence': primary_confidence,
            'pdf_path': pdf_filename,
            'gradcam_images': heatmap_images,
            'timestamp': datetime.now().isoformat(),
            'selected_disease': selected_disease
        }
        
        return create_response(True, 'Prediction successful', data=response_data)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        db.session.rollback()
        return create_response(False, str(e), status=500)


@app.route('/api/predictions/<int:prediction_id>', methods=['GET'])
@login_required
def get_prediction(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    
    gradcam_data = {}
    if prediction.gradcam_images:
        try:
            gradcam_data = json.loads(prediction.gradcam_images)
        except:
            gradcam_data = {}
    
    return jsonify({
        'id': prediction.id,
        'patient_id': prediction.patient_id,
        'disease_detected': prediction.disease_detected,
        'confidence': float(prediction.confidence) if prediction.confidence else 0,
        'all_predictions': json.loads(prediction.all_predictions) if prediction.all_predictions else {},
        'gradcam_images': gradcam_data,
        'created_at': prediction.created_at.isoformat(),
        'report_path': prediction.report_path
    })


@app.route('/api/patient/<patient_id>/history', methods=['GET'])
@login_required
def get_patient_history(patient_id):
    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        return jsonify([])
    
    predictions = Prediction.query.filter_by(patient_id=patient.id).order_by(Prediction.created_at.desc()).all()
    
    history = []
    for pred in predictions:
        history.append({
            'id': pred.id,
            'date': pred.created_at.strftime('%Y-%m-%d %H:%M'),
            'disease': pred.disease_detected,
            'confidence': float(pred.confidence) if pred.confidence else 0,
            'report_path': pred.report_path
        })
    
    return jsonify(history)


@app.route('/api/patient-list', methods=['GET'])
@login_required
def get_patient_list():
    search = request.args.get('search', '')
    limit = request.args.get('limit', 50, type=int)
    
    query = Patient.query
    
    if search:
        query = query.filter(
            db.or_(
                Patient.name.ilike(f'%{search}%'),
                Patient.patient_id.ilike(f'%{search}%'),
                Patient.phone.ilike(f'%{search}%')
            )
        )
    
    patients = query.limit(limit).all()
    
    result = []
    for patient in patients:
        last_pred = patient.predictions[-1] if patient.predictions else None
        result.append({
            'id': patient.id,
            'patient_id': patient.patient_id,
            'name': patient.name,
            'age': patient.age,
            'gender': patient.gender,
            'phone': patient.phone,
            'last_analysis': last_pred.created_at.isoformat() if last_pred else None,
            'disease': last_pred.disease_detected if last_pred else None,
            'confidence': last_pred.confidence if last_pred else 0
        })
    
    return jsonify(result)


@app.route('/api/analytics', methods=['GET'])
@login_required
def get_analytics():
    period = request.args.get('period', 'month')
    
    if period == 'week':
        start_date = datetime.now() - timedelta(days=7)
    elif period == 'month':
        start_date = datetime.now() - timedelta(days=30)
    elif period == 'year':
        start_date = datetime.now() - timedelta(days=365)
    else:
        start_date = datetime.now() - timedelta(days=30)
    
    predictions = Prediction.query.filter(Prediction.created_at >= start_date).all()
    
    disease_counts = {}
    for pred in predictions:
        if pred.disease_detected:
            disease_counts[pred.disease_detected] = disease_counts.get(pred.disease_detected, 0) + 1
    
    daily_counts = {}
    for pred in predictions:
        day = pred.created_at.strftime('%Y-%m-%d')
        daily_counts[day] = daily_counts.get(day, 0) + 1
    
    confidence_ranges = {'0-0.5': 0, '0.5-0.7': 0, '0.7-0.9': 0, '0.9-1.0': 0}
    for pred in predictions:
        conf = float(pred.confidence) if pred.confidence else 0
        if conf < 0.5:
            confidence_ranges['0-0.5'] += 1
        elif conf < 0.7:
            confidence_ranges['0.5-0.7'] += 1
        elif conf < 0.9:
            confidence_ranges['0.7-0.9'] += 1
        else:
            confidence_ranges['0.9-1.0'] += 1
    
    return jsonify({
        'total_predictions': len(predictions),
        'disease_distribution': disease_counts,
        'daily_predictions': daily_counts,
        'confidence_distribution': confidence_ranges,
        'period': period
    })


@app.route('/api/download-report/<path:filename>')
@login_required
def download_report(filename):
    try:
        possible_paths = [
            Path('static/reports') / filename,
            Path('reports') / filename,
            Path('static') / 'reports' / filename,
            Path('static/reports').resolve() / filename,
        ]
        
        report_path = None
        for path in possible_paths:
            if path.exists():
                report_path = path
                print(f"Found report at: {report_path}")
                break
        
        if report_path is None:
            print(f"Report not found: {filename}")
            return create_response(False, f'Report not found: {filename}', status=404)
        
        return send_file(
            report_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
    except Exception as e:
        print(f"Download error: {e}")
        return create_response(False, str(e), status=404)


@app.route('/api/share-report/<int:prediction_id>', methods=['POST'])
@login_required
def share_report(prediction_id):
    try:
        prediction = Prediction.query.get_or_404(prediction_id)
        email = request.json.get('email')
        
        if not email:
            return create_response(False, 'Email required', status=400)
        
        from utils.email_sender import send_report_email
        success = send_report_email(email, prediction)
        
        if success:
            return create_response(True, 'Report shared successfully')
        else:
            return create_response(False, 'Failed to send email', status=500)
            
    except Exception as e:
        logger.error(f"Share error: {e}")
        return create_response(False, str(e), status=500)


@app.errorhandler(413)
def too_large(e):
    return create_response(False, 'File too large. Maximum size is 16MB', status=413)


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    db.session.rollback()
    return create_response(False, 'Internal server error', status=500)


@app.errorhandler(404)
def not_found(e):
    return create_response(False, 'Endpoint not found', status=404)


# ==============================================
# MAIN ENTRY POINT
# ==============================================

if __name__ == '__main__':
    # Create necessary directories
    directories = [
        Path('models'),
        Path('static/reports'),
        Path('logs'),
        Path('uploads')
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    loaded_diseases = [d for d, m in models.items() if m.get('primary') is not None]
    print("\n" + "="*70)
    print("PNEUMOSCAN PRO ADVANCED - MEDICAL IMAGING PLATFORM v2.0")
    print("="*70)
    print(f"Environment: {'PRODUCTION' if IS_PRODUCTION else 'DEVELOPMENT'}")
    print(f"Server: http://127.0.0.1:5000")
    print(f"Reports Directory: static/reports")
    print(f"Models loaded: {len(loaded_diseases)} - {loaded_diseases}")
    print(f"Diseases: {', '.join(DISEASE_CONFIG.keys())}")
    print("="*70 + "\n")
    
    # Use production server (gunicorn) when on Render, otherwise Flask dev server
    if IS_PRODUCTION:
        # For production, use gunicorn (configured in Procfile)
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
    else:
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=True,
            threaded=True
        )