from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import joblib
import re
import os
import pandas as pd
from datetime import datetime
import shap
import numpy as np


app = Flask(__name__, template_folder='templates')
app.secret_key = 'jobshield_secret_key_2024'

# Database setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/pc/jobshield.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load models
rf_model = joblib.load(os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl'))
tfidf = joblib.load(os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl'))

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200))
    company = db.Column(db.String(200))
    result = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_shap_explanation(text, model, vectorizer, top_n=5):
    """Get SHAP values to explain the prediction"""
    X = vectorizer.transform([text])
    X_dense = X.toarray()
    feature_names = vectorizer.get_feature_names_out()
    
    # Get non-zero features only (words actually in the text)
    nonzero_indices = np.where(X_dense[0] != 0)[0]
    
    if len(nonzero_indices) == 0:
        return []
    
    # Use feature importances from Random Forest
    importances = model.feature_importances_
    
    # Get importance scores for words in this text
    explanations = []
    for idx in nonzero_indices:
        word = feature_names[idx]
        importance = importances[idx]
        tfidf_score = X_dense[0][idx]
        combined_score = importance * tfidf_score
        explanations.append({
            'word': word,
            'importance': float(combined_score),
            'direction': 'Suggests Fake' if combined_score > 0 else 'Suggests Real'
        })
    
    # Sort by importance and return top N
    explanations = sorted(explanations, 
                         key=lambda x: abs(x['importance']), 
                         reverse=True)[:top_n]
    
    return explanations

# Create tables
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            error = 'Username already exists!'
        else:
            new_user = User(
                username=username,
                password=generate_password_hash(password)
            )
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
    return render_template('signup.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        error = 'Invalid username or password!'
    return render_template('login.html', error=error)

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get user stats
    user_id = session['user_id']
    total = Prediction.query.filter_by(user_id=user_id).count()
    fake_count = Prediction.query.filter_by(user_id=user_id, result='FAKE JOB POSTING').count()
    real_count = Prediction.query.filter_by(user_id=user_id, result='REAL JOB POSTING').count()
    recent = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.timestamp.desc()).limit(5).all()
    
    return render_template('dashboard.html',
        username=session['username'],
        total=total,
        fake_count=fake_count,
        real_count=real_count,
        recent=recent
    )

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    title = request.form.get('title', '')
    company = request.form.get('company', '')
    description = request.form.get('description', '')
    requirements = request.form.get('requirements', '')
    benefits = request.form.get('benefits', '')

    combined_text = f"{title} {company} {description} {requirements} {benefits}"
    cleaned = clean_text(combined_text)
    vectorized = tfidf.transform([cleaned])
    prediction = rf_model.predict(vectorized)[0]
    probability = rf_model.predict_proba(vectorized)[0]

    result = "FAKE JOB POSTING" if prediction == 1 else "REAL JOB POSTING"
    confidence = round(float(max(probability)) * 100, 2)

    # Save to database
    new_prediction = Prediction(
        user_id=session['user_id'],
        title=title,
        company=company,
        result=result,
        confidence=confidence
    )
    db.session.add(new_prediction)
    db.session.commit()
    
    # Get SHAP explanation
    try:
        explanations = get_shap_explanation(cleaned, rf_model, tfidf, top_n=5)
    except Exception as e:
        explanations = []

    return jsonify({
        'result': result,
        'confidence': confidence,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'explanations': explanations
    })
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        df = pd.read_csv(file)
        results = []

        for _, row in df.iterrows():
            text = ' '.join([str(row.get(col, '')) for col in ['title', 'company_profile', 'description', 'requirements', 'benefits']])
            cleaned = clean_text(text)
            vectorized = tfidf.transform([cleaned])
            prediction = rf_model.predict(vectorized)[0]
            probability = rf_model.predict_proba(vectorized)[0]
            result = "FAKE" if prediction == 1 else "REAL"
            confidence = round(float(max(probability)) * 100, 2)

            # Save each prediction
            new_prediction = Prediction(
                user_id=session['user_id'],
                title=str(row.get('title', 'Unknown')),
                company=str(row.get('company_profile', 'Unknown')),
                result=result + ' JOB POSTING',
                confidence=confidence
            )
            db.session.add(new_prediction)
            results.append({
                'title': str(row.get('title', 'Unknown')),
                'result': result,
                'confidence': confidence
            })

        db.session.commit()
        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    predictions = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.timestamp.desc()).all()
    return render_template('history.html', predictions=predictions)

@app.route('/analytics')
def analytics():
    if 'username' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    total = Prediction.query.filter_by(user_id=user_id).count()
    fake_count = Prediction.query.filter_by(user_id=user_id, result='FAKE JOB POSTING').count()
    real_count = Prediction.query.filter_by(user_id=user_id, result='REAL JOB POSTING').count()
    fake_percentage = round((fake_count / total * 100), 2) if total > 0 else 0
    real_percentage = round((real_count / total * 100), 2) if total > 0 else 0
    return render_template('analytics.html',
        total=total,
        fake_count=fake_count,
        real_count=real_count,
        fake_percentage=fake_percentage,
        real_percentage=real_percentage
    )

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)