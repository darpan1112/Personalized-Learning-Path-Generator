import os, json, re
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash)
from flask_login import (LoginManager, login_user, logout_user,
                         login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from database.models import db, User, QuizResult, LearningPath, Badge, Note, AINote, ChatMessage
from database.seed_data import QUIZ_QUESTIONS, SUBJECTS, CLASS_LEVELS, CLASS_SUBJECTS
from ml.deep_model import (analyze_skills, predict_skill_level,
                            generate_learning_path, calculate_points,
                            get_level_from_points, check_badges)

# ── Firebase import ───────────────────────────────────────
from firebase_init import init_firebase, verify_token

# ── App setup ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ai-learning-secret-2024')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ai_learning.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', '')

# ── Firebase initialize ───────────────────────────────────
init_firebase()

import google.generativeai as genai
genai.configure(api_key="AIzaSyDTjSx1A17ZwysvI3OoKpnTa6tt1qfBjoM")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── Helpers ───────────────────────────────────────────────
def get_user_quiz_results(user_id):
    results = QuizResult.query.filter_by(user_id=user_id).all()
    return [{'subject': r.subject, 'score': r.score,
             'total': r.total, 'percentage': r.percentage,
             'taken_at': r.taken_at} for r in results]

def call_ai(prompt, system=None, max_tokens=800):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash',
            system_instruction=system) if system else genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"AI API Error: {e}")
        return None

def update_user_progress(user):
    results = get_user_quiz_results(user.id)
    if not results:
        return
    scores = analyze_skills(results)
    level_name, conf, probs = predict_skill_level(scores)
    points = calculate_points(results)
    user.points = points
    user.level  = get_level_from_points(points)
    existing_path = LearningPath.query.filter_by(user_id=user.id).first()
    path = generate_learning_path(scores)
    if existing_path:
        existing_path.path_data = json.dumps(path)
    else:
        db.session.add(LearningPath(user_id=user.id, path_data=json.dumps(path)))
    existing = [{'badge_id': b.badge_id, 'name': b.name,
                 'icon': b.icon} for b in Badge.query.filter_by(user_id=user.id).all()]
    new_badges = check_badges(results, existing)
    for b in new_badges:
        if not Badge.query.filter_by(user_id=user.id, badge_id=b['badge_id']).first():
            db.session.add(Badge(user_id=user.id, badge_id=b['badge_id'],
                                 name=b['name'], icon=b['icon']))
    db.session.commit()

# ══════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for(f"{current_user.role}.dashboard"))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for(f"{current_user.role}.dashboard"))

    if request.method == 'POST':
        # ✅ FIXED: Accept both JSON (from login.html fetch) and form POST
        is_json  = request.is_json
        data     = request.get_json() if is_json else request.form

        email    = data.get('email', '').strip()
        password = data.get('password', '')
        remember = data.get('remember') == 'on'

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=remember)
            redirect_url = url_for(f"{user.role}.dashboard")
            # ✅ JSON response for fetch call in login.html
            if is_json:
                return jsonify({'success': True, 'redirect': redirect_url})
            return redirect(redirect_url)

        # Wrong credentials
        if is_json:
            return jsonify({'success': False, 'error': 'Invalid email or password'}), 401
        flash('Invalid email or password', 'error')

    return render_template('login.html', class_levels=CLASS_LEVELS)


# ── Firebase Google Login Route ────────────────────────────
@app.route('/firebase-login', methods=['POST'])
def firebase_login():
    if current_user.is_authenticated:
        return jsonify({'success': True, 'redirect': url_for(f"{current_user.role}.dashboard")})

    data = request.get_json() or {}
    id_token = data.get('idToken')
    role = data.get('role', 'student')

    if not id_token:
        return jsonify({'success': False, 'error': 'No token provided'}), 400

    decoded_token = verify_token(id_token)
    if not decoded_token:
        return jsonify({'success': False, 'error': 'Invalid or expired token'}), 401

    email = decoded_token.get('email')
    firebase_uid = decoded_token.get('uid')
    name = decoded_token.get('name', email.split('@')[0] if email else 'User')

    if not email:
        return jsonify({'success': False, 'error': 'Email not found in token'}), 400

    user = User.query.filter_by(email=email).first()

    if not user:
        # Create new user
        random_pwd = os.urandom(24).hex()
        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(random_pwd),
            role=role,
            firebase_uid=firebase_uid
        )
        db.session.add(user)
        db.session.commit()
    elif not user.firebase_uid:
        # Update existing user to link google account
        user.firebase_uid = firebase_uid
        db.session.commit()

    login_user(user, remember=True)
    return jsonify({
        'success': True,
        'redirect': url_for(f"{user.role}.dashboard")
    })

@app.route('/signup', methods=['GET'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for(f"{current_user.role}.dashboard"))
    return render_template('login.html', active_tab='signup', class_levels=CLASS_LEVELS)

@app.route('/register', methods=['POST'])
def register():
    # ✅ FIXED: Accept both JSON (from login.html fetch) and form POST
    is_json  = request.is_json
    data     = request.get_json() if is_json else request.form

    name        = data.get('name', '').strip()
    email       = data.get('email', '').strip()
    password    = data.get('password', '')
    role        = data.get('role', 'student')
    subject     = data.get('subject', '')
    class_level = data.get('class_level', '').strip()

    if User.query.filter_by(email=email).first():
        if is_json:
            return jsonify({'success': False, 'error': 'Email already registered'}), 400
        flash('Email already registered', 'error')
        return redirect(url_for('login'))

    user = User(name=name, email=email,
                password_hash=generate_password_hash(password),
                role=role, subject=subject, class_level=class_level if role=='student' else None)
    db.session.add(user)
    db.session.commit()

    if is_json:
        return jsonify({'success': True, 'message': 'Account created! Please log in.'})
        
    flash('Account created successfully! Please log in.', 'success')
    return redirect(url_for('login'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ══════════════════════════════════════════════════════════
# STUDENT BLUEPRINT
# ══════════════════════════════════════════════════════════
from flask import Blueprint
student_bp = Blueprint('student', __name__, url_prefix='/student')

@student_bp.before_request
def student_auth():
    # ✅ FIXED: Manually check auth instead of stacking @login_required
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    if current_user.role != 'student':
        return redirect(url_for('teacher.dashboard'))

@student_bp.route('/dashboard')
def dashboard():
    results    = get_user_quiz_results(current_user.id)
    scores     = analyze_skills(results) if results else {}
    level_name, conf, probs = predict_skill_level(scores)
    badges     = Badge.query.filter_by(user_id=current_user.id).all()
    path_row   = LearningPath.query.filter_by(user_id=current_user.id).first()
    weak       = sorted([(s,v) for s,v in scores.items() if v < 60], key=lambda x: x[1])
    return render_template('student/student_dashboard.html',
        user=current_user, results=results[:6], scores=scores,
        level_name=level_name, conf=round(conf*100), probs=probs,
        badges=badges, weak_topics=weak,
        path=json.loads(path_row.path_data) if path_row else [])

@student_bp.route('/subjects')
def subjects():
    results = get_user_quiz_results(current_user.id)
    scores  = analyze_skills(results) if results else {}
    return render_template('student/student_subjects.html',
        user=current_user, subjects=SUBJECTS, scores=scores)

@student_bp.route('/api/generate-quiz')
def api_generate_quiz():
    subject = request.args.get('subject', 'General')
    class_level = getattr(current_user, 'class_level', None) or 'Class 8'
    system_prompt = (
        f"You are a quiz generator for school students. Generate exactly 5 multiple choice questions about {subject} "
        f"for a {class_level} student. Use age-appropriate language and difficulty. "
        "Return ONLY a raw JSON array of objects. Do not use markdown code blocks like ```json. "
        'Each object MUST have this exact structure: '
        '{"q": "The question text", "o": ["Option A", "Option B", "Option C", "Option D"], "a": 1} '
        "where 'a' is the 0-indexed integer of the correct option."
    )
    
    response = call_ai("Generate the quiz in JSON now.", system=system_prompt)
    if response:
        cleaned = re.sub(r"```json\s*", "", response)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()
        try:
            questions = json.loads(cleaned)
            if isinstance(questions, list) and len(questions) > 0:
                return jsonify({'success': True, 'questions': questions[:5]})
        except Exception as e:
            print("Failed to parse AI quiz JSON:", e)
            print("Response was:", response)
            
    # Fallback to hardcoded seed data if AI fails
    fallback = QUIZ_QUESTIONS.get(subject, [])
    # Re-map hardcoded structure: seed data is {'q', 'opts', 'ans'} where ans is index
    formatted = []
    for q in fallback:
        formatted.append({
            'q': q['q'],
            'o': q['opts'],
            'a': int(q['ans'])
        })
    return jsonify({'success': False, 'questions': formatted})

@student_bp.route('/quiz/<subject>')
def quiz(subject):
    class_level = getattr(current_user, 'class_level', None) or 'Class 8'
    # Filter subjects available for this student's class level
    allowed_ids = CLASS_SUBJECTS.get(class_level, [s['id'] for s in SUBJECTS])
    class_subjects = [s for s in SUBJECTS if s['id'] in allowed_ids]
    # Fallback: if class map doesn't match, show all
    if not class_subjects:
        class_subjects = SUBJECTS
    return render_template('student/student_quiz.html',
        user=current_user, subject=subject,
        subjects=class_subjects, class_level=class_level)

@student_bp.route('/quiz/submit', methods=['POST'])
def quiz_submit():
    data       = request.get_json()
    subject    = data.get('subject')
    score      = int(data.get('score', 0))
    total      = int(data.get('total', 5))
    percentage = round((score / total) * 100) if total else 0
    weak       = data.get('weak', [])
    
    qr = QuizResult(user_id=current_user.id, subject=subject,
                    score=score, total=total, percentage=percentage,
                    weak_topics=json.dumps(weak))
    db.session.add(qr)
    db.session.commit()
    update_user_progress(current_user)
    results = get_user_quiz_results(current_user.id)
    scores  = analyze_skills(results)
    level_name, conf, _ = predict_skill_level(scores)
    return jsonify({'score': score, 'total': total, 'percentage': percentage, 'level': level_name})

@student_bp.route('/path')
def path():
    path_row = LearningPath.query.filter_by(user_id=current_user.id).first()
    return render_template('student/student_path.html', user=current_user,
        path=json.loads(path_row.path_data) if path_row else [])

@student_bp.route('/ai-notes/<subject>')
def ai_notes(subject):
    existing = AINote.query.filter_by(user_id=current_user.id, subject=subject).first()
    if existing:
        return jsonify({'notes': existing.notes, 'cached': True})
    results = get_user_quiz_results(current_user.id)
    scores  = analyze_skills(results)
    score   = scores.get(subject, 50)
    prompt  = f'Generate study notes for "{subject}". Student scored {score}%.'
    notes   = call_ai(prompt) or f'## {subject}\n\nAdd Gemini API key'
    db.session.add(AINote(user_id=current_user.id, subject=subject, notes=notes))
    db.session.commit()
    return jsonify({'notes': notes})

@student_bp.route('/notes', methods=['GET', 'POST'])
def notes():
    if request.method == 'POST':
        subject  = request.form.get('subject', '')
        file     = request.files.get('file')
        if file and file.filename:
            content  = file.read().decode('utf-8', errors='ignore')
            analysis = call_ai(f'Analyze: {content[:3000]}') or 'Add Gemini API key.'
            db.session.add(Note(user_id=current_user.id, subject=subject,
                                file_name=file.filename, content=content[:5000],
                                ai_analysis=analysis))
            db.session.commit()
        return redirect(url_for('student.notes'))
    all_notes = Note.query.filter_by(user_id=current_user.id).order_by(Note.created_at.desc()).all()
    return render_template('student/student_notes.html', user=current_user,
        notes=all_notes, subjects=SUBJECTS)

@student_bp.route('/chatbot')
def chatbot():
    history = ChatMessage.query.filter_by(user_id=current_user.id).order_by(ChatMessage.sent_at).all()
    return render_template('student/student_chatbot.html', user=current_user,
        history=history, subjects=SUBJECTS)

@student_bp.route('/chatbot/send', methods=['POST'])
def chatbot_send():
    data    = request.get_json()
    msg     = data.get('message', '').strip()
    subject = data.get('subject', 'General')
    system  = f'You are a helpful AI tutor. Subject: {subject}. Be clear. Under 150 words.'
    reply   = call_ai(msg, system=system) or 'Add Gemini API key'
    db.session.add(ChatMessage(user_id=current_user.id, role='user', content=msg, subject=subject))
    db.session.add(ChatMessage(user_id=current_user.id, role='assistant', content=reply, subject=subject))
    db.session.commit()
    return jsonify({'reply': reply})

@student_bp.route('/profile')
def profile():
    results  = get_user_quiz_results(current_user.id)
    scores   = analyze_skills(results) if results else {}
    level_name, conf, probs = predict_skill_level(scores)
    badges   = Badge.query.filter_by(user_id=current_user.id).all()
    path_row = LearningPath.query.filter_by(user_id=current_user.id).first()
    return render_template('student/student_profile.html', user=current_user,
        results=results, scores=scores, level_name=level_name,
        conf=round(conf*100), probs=probs, badges=badges,
        path=json.loads(path_row.path_data) if path_row else [])

app.register_blueprint(student_bp)

# ══════════════════════════════════════════════════════════
# TEACHER BLUEPRINT
# ══════════════════════════════════════════════════════════
teacher_bp = Blueprint('teacher', __name__, url_prefix='/teacher')

@teacher_bp.before_request
def teacher_auth():
    # ✅ FIXED: Manually check auth instead of stacking @login_required
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    if current_user.role != 'teacher':
        return redirect(url_for('student.dashboard'))

@teacher_bp.route('/dashboard')
def dashboard():
    results  = get_user_quiz_results(current_user.id)
    scores   = analyze_skills(results) if results else {}
    level_name, conf, probs = predict_skill_level(scores)
    badges   = Badge.query.filter_by(user_id=current_user.id).all()
    path_row = LearningPath.query.filter_by(user_id=current_user.id).first()
    weak     = sorted([(s,v) for s,v in scores.items() if v < 60], key=lambda x: x[1])
    return render_template('teacher/teacher_dashboard.html',
        user=current_user, results=results[:6], scores=scores,
        level_name=level_name, conf=round(conf*100), probs=probs,
        badges=badges, weak_topics=weak,
        path=json.loads(path_row.path_data) if path_row else [])

@teacher_bp.route('/students')
def students():
    return render_template('teacher/teacher_students.html', user=current_user)

@teacher_bp.route('/reports')
def reports():
    return render_template('teacher/teacher_reports.html', user=current_user)

@teacher_bp.route('/subjects')
def subjects():
    results = get_user_quiz_results(current_user.id)
    scores  = analyze_skills(results) if results else {}
    return render_template('teacher/teacher_subjects.html',
        user=current_user, subjects=SUBJECTS, scores=scores)

@teacher_bp.route('/quiz/<subject>')
def quiz(subject):
    # For teachers, we still fetch questions to show them in the manage list
    questions = QUIZ_QUESTIONS.get(subject, [])
    return render_template('teacher/teacher_quiz.html',
        user=current_user, subject=subject, questions=questions, subjects=SUBJECTS)

@teacher_bp.route('/quiz/submit', methods=['POST'])
def quiz_submit():
    data       = request.get_json()
    subject    = data.get('subject')
    score      = int(data.get('score', 0))
    total      = int(data.get('total', 5))
    percentage = round((score / total) * 100) if total else 0
    weak       = data.get('weak', [])
    
    qr = QuizResult(user_id=current_user.id, subject=subject,
                    score=score, total=total, percentage=percentage,
                    weak_topics=json.dumps(weak))
    db.session.add(qr)
    db.session.commit()
    update_user_progress(current_user)
    results = get_user_quiz_results(current_user.id)
    scores  = analyze_skills(results)
    level_name, conf, _ = predict_skill_level(scores)
    return jsonify({'score': score, 'total': total, 'percentage': percentage, 'level': level_name})

@teacher_bp.route('/path')
def path():
    path_row = LearningPath.query.filter_by(user_id=current_user.id).first()
    return render_template('teacher/teacher_path.html', user=current_user,
        path=json.loads(path_row.path_data) if path_row else [])

@teacher_bp.route('/ai-notes/<subject>')
def ai_notes(subject):
    existing = AINote.query.filter_by(user_id=current_user.id, subject=subject).first()
    if existing:
        return jsonify({'notes': existing.notes, 'cached': True})
    results = get_user_quiz_results(current_user.id)
    scores  = analyze_skills(results)
    score   = scores.get(subject, 50)
    prompt  = f'Generate study notes for "{subject}". Teacher scored {score}%.'
    notes   = call_ai(prompt) or f'## {subject}\n\nAdd Gemini API key'
    db.session.add(AINote(user_id=current_user.id, subject=subject, notes=notes))
    db.session.commit()
    return jsonify({'notes': notes})

@teacher_bp.route('/notes', methods=['GET', 'POST'])
def notes():
    if request.method == 'POST':
        subject  = request.form.get('subject', '')
        file     = request.files.get('file')
        if file and file.filename:
            content  = file.read().decode('utf-8', errors='ignore')
            analysis = call_ai(f'Analyze: {content[:3000]}') or 'Add Gemini API key.'
            db.session.add(Note(user_id=current_user.id, subject=subject,
                                file_name=file.filename, content=content[:5000],
                                ai_analysis=analysis))
            db.session.commit()
        return redirect(url_for('teacher.notes'))
    all_notes = Note.query.filter_by(user_id=current_user.id).order_by(Note.created_at.desc()).all()
    return render_template('teacher/teacher_notes.html', user=current_user,
        notes=all_notes, subjects=SUBJECTS)

@teacher_bp.route('/chatbot')
def chatbot():
    history = ChatMessage.query.filter_by(user_id=current_user.id).order_by(ChatMessage.sent_at).all()
    return render_template('teacher/teacher_chatbot.html', user=current_user,
        history=history, subjects=SUBJECTS)

@teacher_bp.route('/chatbot/send', methods=['POST'])
def chatbot_send():
    data    = request.get_json()
    msg     = data.get('message', '').strip()
    subject = data.get('subject', 'General')
    system  = f'You are an AI teaching assistant. Subject: {subject}. Under 200 words.'
    reply   = call_ai(msg, system=system) or 'Add Gemini API key'
    db.session.add(ChatMessage(user_id=current_user.id, role='user', content=msg, subject=subject))
    db.session.add(ChatMessage(user_id=current_user.id, role='assistant', content=reply, subject=subject))
    db.session.commit()
    return jsonify({'reply': reply})

@teacher_bp.route('/profile')
def profile():
    results  = get_user_quiz_results(current_user.id)
    scores   = analyze_skills(results) if results else {}
    level_name, conf, probs = predict_skill_level(scores)
    badges   = Badge.query.filter_by(user_id=current_user.id).all()
    path_row = LearningPath.query.filter_by(user_id=current_user.id).first()
    return render_template('teacher/teacher_profile.html', user=current_user,
        results=results, scores=scores, level_name=level_name,
        conf=round(conf*100), badges=badges,
        path=json.loads(path_row.path_data) if path_row else [])

app.register_blueprint(teacher_bp)

@app.route('/<role>/<path:filename>')
def legacy_html_redirect(role, filename):
    if role in ['student', 'teacher'] and filename.endswith('.html'):
        name = filename.replace(f"{role}_", "").replace('.html', '')
        if name == 'quiz': return redirect(url_for(f"{role}.quiz", subject='General'))
        try: return redirect(url_for(f"{role}.{name}"))
        except: pass
    return "Not Found", 404

# ── Init DB ───────────────────────────────────────────────
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True, port=5000)