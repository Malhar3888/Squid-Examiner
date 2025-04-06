from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import sqlite3
from onlinelogic import online_process_frames
from offlinelogic import offline_process_frames

app = Flask(__name__)

def initialize_database():
    conn = sqlite3.connect('squid_examiner.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, email TEXT, password TEXT, approved INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

# Initialize the database
initialize_database()

# Routes

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/admin')
def admin_panel():
    conn = sqlite3.connect('squid_examiner.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users")
    users = c.fetchall()
    conn.close()
    return render_template('admin_panel.html', users=users)

@app.route('/approve/<int:user_id>')
def approve_user(user_id):
    conn = sqlite3.connect('squid_examiner.db')
    c = conn.cursor()
    c.execute("UPDATE users SET approved=1 WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('admin_panel'))

@app.route('/decline/<int:user_id>')
def decline_user(user_id):
    conn = sqlite3.connect('squid_examiner.db')
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('admin_panel'))

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect('squid_examiner.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
        conn.commit()
        conn.close()
        return redirect(url_for('index'))

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect('squid_examiner.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
        user = c.fetchone()
        conn.close()
        if user:
            if user[4] == 1:
                return render_template('index.html', username=user[1])
            else:
                return "Your account is pending approval by the admin."
        else:
            return "Invalid credentials. Please try again or register."

@app.route('/squid_examiner/home')
def home():
    return render_template('index.html')

@app.route('/squid_examiner/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/squid_examiner/services')
def services():
    return render_template('services.html')

@app.route('/squid_examiner/online_exam')
def online_exam():
    return render_template('online_exam.html')

@app.route('/squid_examiner/online_exam/online_video')
def online_video():
    return render_template('online_video.html')

@app.route('/online_feed')
def online_feed():
    cap = cv2.VideoCapture(0)
    return Response(online_process_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/squid_examiner/offline_exam')
def offline_exam():
    return render_template('offline_exam.html')

@app.route('/squid_examiner/offline_exam/offline_video')
def offline_video():
    return render_template('offline_video.html')

@app.route('/offline_feed')
def offline_feed():
    cap = cv2.VideoCapture(0)
    return Response(offline_process_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/squid_examiner/team')
def team():
    return render_template('team.html')

if __name__ == "__main__":
    app.run(debug=True)