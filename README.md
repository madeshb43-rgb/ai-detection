# ai-detection
ai based animal footprint detection
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from models import db, User

app = Flask(_name_)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Prediction Helper ---
MODEL_PATH = './models/footprint_model.h5'
CLASSES_PATH = './models/classes.txt'
IMG_SIZE = (160, 160)
PREDICTION_THRESHOLD = 0.20


def load_model_and_classes():
    model = None
    classes = []

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)

    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]

    return model, classes


MODEL, CLASSES = load_model_and_classes()


def preprocess_for_model(img, input_shape):
    if len(input_shape) != 4:
        raise ValueError(f"Unexpected model input shape: {input_shape}")

    if input_shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = preprocess_input(img.astype(np.float32))
        return img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 3)

    if input_shape[-1] == 1:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMG_SIZE)
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        normalized = blurred / 255.0
        return normalized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

    raise ValueError(f"Unsupported model channel count: {input_shape[-1]}")


def predict_animal(img_path):
    global MODEL, CLASSES

    if MODEL is None or not CLASSES:
        MODEL, CLASSES = load_model_and_classes()

    if MODEL is None or not CLASSES:
        return "Model not trained yet", 0.0

    img = cv2.imread(img_path)
    if img is None:
        return "Unable to read uploaded image", 0.0

    input_data = preprocess_for_model(img, MODEL.input_shape)
    predictions = MODEL.predict(input_data)
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx])
    prediction = CLASSES[class_idx]

    if confidence < PREDICTION_THRESHOLD:
        prediction = f"{prediction} (low confidence)"

    return prediction, confidence

# --- Routes ---

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
            
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user_exists = User.query.filter_by(username=username).first()
        if user_exists:
            flash('Username already exists')
        else:
            new_user = User(username=username, password=generate_password_hash(password))
            db.session.add(new_user)
            db.session.commit()
            flash('Account created! Please login.')
            return redirect(url_for('login'))
            
    return render_template('login.html', signup=True)

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required

def dashboard():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            prediction, confidence = predict_animal(file_path)
            return render_template('result.html', 
                                 prediction=prediction, 
                                 confidence=f"{confidence*100:.2f}%",
                                 image_path=url_for('static', filename='uploads/' + filename))
            
    return render_template('dashboard.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if _name_ == '_main_':
    with app.app_context():
        db.create_all()
        # Create a default user if none exists
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', password=generate_password_hash('password'))
            db.session.add(admin)
            db.session.commit()
            
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=False)
