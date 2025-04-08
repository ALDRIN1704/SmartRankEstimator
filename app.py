from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# ------------------ Load Model & Data ------------------
model = load_model("saved_models/institution_lstm_model.h5")
feature_scaler = joblib.load("saved_models/feature_scaler.pkl")
target_scaler = joblib.load("saved_models/target_scaler.pkl")
df = pd.read_csv("Cleaned_NIRF_2022_2023_2024.csv")
df = df.sort_values(by=["Institute Name", "Year"])
features = ['TLR(100)', 'RPC(100)', 'GO(100)', 'OI(100)', 'Perception(100)']

# ------------------ Initialize DB ------------------
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE,
                    email TEXT,
                    password TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ------------------ Routes ------------------

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
            conn.commit()
            flash("Signup successful! Please log in.", "success")
            return redirect(url_for('login'))
        except:
            flash("Username already exists!", "danger")
        finally:
            conn.close()

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password_input = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[0], password_input):
            session['user'] = username
            return redirect(url_for('predict'))
        else:
            flash("Invalid credentials!", "danger")
    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    predicted_score = None
    estimated_rank = None
    college_selected = ""
    predict_year = ""
    entered_score = ""
    show_rank_input = False
    college_list = sorted(df["Institute Name"].unique().tolist())

    if request.method == 'POST':
        if request.form.get("action") == "predict_score":
            college_name = request.form['college_name']
            predict_year = request.form['predict_year']
            matched = df[df["Institute Name"] == college_name]

            if matched["Institute Name"].nunique() > 0:
                college = matched["Institute Name"].unique()[0]
                college_selected = college
                college_data = df[df["Institute Name"] == college].sort_values(by="Year")

                if len(college_data) >= 3:
                    X_input = college_data.tail(3)[features]
                    X_scaled = feature_scaler.transform(X_input)
                    X_input_seq = np.expand_dims(X_scaled, axis=0)
                    predicted_scaled = model.predict(X_input_seq)
                    predicted_score = float(target_scaler.inverse_transform([[predicted_scaled[0][0]]])[0][0])
                    show_rank_input = True

        elif request.form.get("action") == "predict_rank":
            entered_score = request.form['entered_score']
            predict_year = request.form['predict_year']
            college_selected = request.form['college_name']
            predicted_score = float(request.form['predicted_score'])  # Ensure it is float
            show_rank_input = True

            try:
                input_score = float(entered_score)
                hist_scores = df[df["Year"] == 2024][["Institute Name", "Score"]].sort_values(by="Score", ascending=False)
                temp_df = hist_scores.copy()
                temp_df.loc["User Prediction"] = ["Entered Score", input_score]
                temp_df = temp_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
                estimated_rank = temp_df[temp_df["Institute Name"] == "Entered Score"].index[0] + 1
            except:
                estimated_rank = None

    return render_template("predict.html",
                           predicted_score=predicted_score,
                           estimated_rank=estimated_rank,
                           college=college_selected,
                           predict_year=predict_year,
                           entered_score=entered_score,
                           show_rank_input=show_rank_input,
                           college_list=college_list)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('landing'))

# ------------------ Run Server ------------------

if __name__ == '__main__':
    app.run(debug=True)
