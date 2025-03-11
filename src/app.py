from flask import Flask, render_template, redirect, url_for, request,session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import io
from flask import jsonify
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask_mail import Mail, Message
from datetime import datetime, timedelta
from twilio.rest import Client
import requests
import yagmail
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
from flask import current_app
import openai  # If using OpenAI API
from flask import Flask, render_template, session
from dotenv import load_dotenv
import os
import google.generativeai as genai
import re
from werkzeug.utils import secure_filename
import sqlite3  # or MySQLdb / psycopg2 for MySQL/PostgreSQL

load_dotenv()  # Load environment variables from .env file

# Set up OpenAI API Key (If using OpenAI GPT)

client = openai.OpenAI(api_key=os.getenv("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"))



app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(BASE_DIR, 'users.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "your_secret_key"
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# Flask-Mail Configuration (Use your email service provider)
app.config["UPLOAD_FOLDER"] = "static/uploads/"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

os.environ["GOOGLE_API_KEY"] = "AIzaSyBQWxqjs3Fuy2htS2vyQzW5-XX7Ty0blaA"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

TWILIO_ACCOUNT_SID = "AC5628f7069cb431cb13d4da6b2f98dba7"

TWILIO_AUTH_TOKEN = "8fb90092a762d50746318f5e68052087"
TWILIO_PHONE_NUMBER = "+18125788331"

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

FAST2SMS_API_KEY = "imkgQAWzyrJj6IDlSdYfMO0cw75nGRHtuUZpsBPVT9CaXv2bhFamuSVchvYPx7psnok2wOW1XNrQEUz3"

model = load_model("model.h5")
print("Model loaded successfully!")

# Define class labels
class_labels = {
    0: "Gingivitis",
    1: "Hypodontia",
    2: "Tooth Discoloration",
    3: "Ulcers"
}

# Doctor Model
class Doctor(db.Model, UserMixin):
    __tablename__ = "doctor"
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(10), nullable=False)
    specialization = db.Column(db.String(100), nullable=False)
    city = db.Column(db.String(50), nullable=False)
    hospital_name = db.Column(db.String(100), nullable=False)

# Patient Model
class Patient(db.Model, UserMixin):
    __tablename__ = "patient"
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    doctor_id = db.Column(db.Integer, nullable=False)
    doctor_name=db.Column(db.String(100), nullable=False)
    patient_name = db.Column(db.String(100), nullable=False)
    patient_email = db.Column(db.String(100), nullable=False)
    patient_phone = db.Column(db.String(15), nullable=False)
    date = db.Column(db.String(50), nullable=False)
    problem_description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default="Pending")  # Appointment Status


    def __repr__(self):
        return f"<Appointment {self.id} - {self.patient_name} - {self.status}>"

@login_manager.user_loader
def load_user(user_id):
    doctor = db.session.get(Doctor, int(user_id))
    if doctor:
        return doctor
    return db.session.get(Patient, int(user_id))



@app.route('/')
def home():
    return render_template('home.html')  # Show home page when app opens



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        login_type = request.form['login_type']  # Identifies login type

        user = None
        if login_type == 'Patient':
            user = Patient.query.filter_by(username=username).first()
            if user and check_password_hash(user.password, password):  # Secure password check
                login_user(user)
                flash('Login successful!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Incorrect username or password', 'error_patient')

        elif login_type == 'Doctor':
            user = Doctor.query.filter_by(username=username).first()

            if user and check_password_hash(user.password, password):  # Secure password check
               login_user(user)
               flash('Login successful!', 'success')
               return redirect(url_for('doctor_home'))
            else:
                flash('Incorrect username or password', 'error_doctor')


    return render_template('login.html')

@app.route('/upload')
@login_required
def upload_page():
    return render_template('index.html')



@app.route('/index')
@login_required
def index():
    return render_template('index.html')



@app.route('/register_patient', methods=['GET', 'POST'])
def register_patient():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        fullname = request.form['fullname']
        age = request.form['age']
        gender = request.form['gender']

        if Patient.query.filter_by(username=username).first() or Patient.query.filter_by(email=email).first():
            flash("Username or Email already exists. Please use a different one.", "danger")
            return redirect(url_for('register_patient'))

        hashed_password = generate_password_hash(password)
        new_patient = Patient(username=username, email=email, phone=phone,password=hashed_password, fullname=fullname, age=int(age), gender=gender)
        db.session.add(new_patient)
        db.session.commit()

        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for('login'))

    return render_template('registration.html')

@app.route('/register_doctor', methods=['GET', 'POST'])
def register_doctor():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        phone = request.form['phone']
        specialization = request.form['specialization']
        city = request.form['city']
        hospital_name = request.form['hospital_name']

        if Doctor.query.filter_by(username=username).first() or Doctor.query.filter_by(email=email).first():
            flash("Username or Email already exists. Please use a different one.", "danger")
            return redirect(url_for('register_doctor'))

        hashed_password = generate_password_hash(password)
        new_doctor = Doctor(fullname=fullname, email=email, username=username, password=hashed_password, phone=phone, specialization=specialization, city=city, hospital_name=hospital_name)
        db.session.add(new_doctor)
        db.session.commit()

        flash("Doctor Registration Successful! You can now log in.", "success")
        return redirect(url_for('login'))

    return render_template('registration1.html')

@app.route('/doctor_home')
@login_required
def doctor_home():
    return render_template('button.html')

@app.route('/doctors')
def doctors():
    predicted_condition = session.get('predicted_condition')

    if not predicted_condition:
        return render_template('form3.html', recommended_doctors=[], other_doctors=Doctor.query.all())

    # ✅ Fix filtering: Ensure exact match by stripping and lowercasing
    recommended_doctors = [
        doctor for doctor in Doctor.query.all()
        if doctor.specialization.strip().lower() == predicted_condition.strip().lower()
    ]

    # ✅ Get other doctors (excluding recommended ones)
    other_doctors = [
        doctor for doctor in Doctor.query.all()
        if doctor not in recommended_doctors
    ]

    return render_template('form3.html', recommended_doctors=recommended_doctors, other_doctors=other_doctors)

# ✅ Send email with app context in a new thread
def send_email_with_context(app, patient_email, subject, body):
    with app.app_context():
        send_email(patient_email, subject, body)

import threading

@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    patient_id = request.form.get('patient_id')
    doctor_id = request.form.get('doctor_id')
    patient_name = request.form.get('patient_name')
    patient_email = request.form.get('patient_email')
    patient_phone = request.form.get('patient_phone')
    appointment_date = request.form.get('appointment_date')
    problem_description = request.form.get('problem_description')

    if not all([doctor_id, patient_name, patient_email, patient_phone, appointment_date, problem_description]):
        return "Missing fields", 400

    # ✅ Fetch the doctor's name
    doctor = Doctor.query.filter_by(id=doctor_id).first()

    if not doctor:
        return "Doctor not found", 404

    # ✅ Create the appointment instance
    appointment = Appointment(
        patient_id=current_user.id,
        doctor_id=doctor_id,
        doctor_name=doctor.fullname,
        patient_name=patient_name,
        patient_email=patient_email,
        patient_phone=patient_phone,
        date=appointment_date,
        problem_description=problem_description,
        status="Pending"
    )

    # ✅ Commit to the DB before starting the thread
    db.session.add(appointment)
    db.session.commit()

    # ✅ Send email asynchronously with app context
    subject = "Appointment Successfully Submitted - DentiPro"
    body = (
        f"Dear {patient_name},\n\n"
        f"Thank you for booking your appointment with Dr. {doctor.fullname} on {appointment_date}.\n"
        f"Your appointment has been successfully submitted and is currently on the waiting list.\n"
        f"We will contact you shortly regarding the confirmation status.\n\n"
        f"Best regards,\nDentiPro Team"
    )

    threading.Thread(
        target=send_email_with_context,
        args=(app, patient_email, subject, body)
    ).start()

    return redirect(url_for('appointment_page'))

# Route for appointment.html
@app.route('/appointment')
def appointment_page():
    return render_template('appointment.html')  # Make sure appointment.html exists in the templates folder


@app.route("/view_patients", methods=["GET"])
@login_required
def view_patients():
    if not isinstance(current_user, Doctor):
        flash("Unauthorized access.", "danger")
        return redirect(url_for("doctor_home"))  # Redirect to doctor's dashboard

    doctor_id = current_user.id  # Use Flask-Login's current_user

    appointments = Appointment.query.filter_by(doctor_id=doctor_id, status="Pending").all()

    return render_template("view_patients.html", appointments=appointments)

@app.route('/view_confirmed_patients')
def view_confirmed_patients():
    if not isinstance(current_user, Doctor):
        flash("Unauthorized access.", "danger")
        return redirect(url_for("doctor_home"))  # Redirect to doctor's dashboard

    doctor_id = current_user.id  # Use Flask-Login's current_user

    appointments = Appointment.query.filter_by(doctor_id=doctor_id, status="Confirmed").all()

    return render_template("view_patients.html", appointments=appointments)



@app.route('/view_cancelled_patients')
def view_cancelled_patients():
    if not isinstance(current_user, Doctor):
        flash("Unauthorized access.", "danger")
        return redirect(url_for("doctor_home"))  # Redirect to doctor's dashboard

    doctor_id = current_user.id  # Use Flask-Login's current_user

    appointments = Appointment.query.filter_by(doctor_id=doctor_id, status="Canceled").all()

    return render_template("view_patients.html", appointments=appointments)


@app.route("/doctor_login", methods=["GET", "POST"])
def doctor_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        doctor = Doctor.query.filter_by(username=username).first()

        if doctor and check_password_hash(doctor.password, password):  # Correct password check
            login_user(doctor)  # Use Flask-Login
            flash("Login successful!", "success")
            return redirect(url_for("doctor_home"))  # Redirect to doctor dashboard
        else:
            flash("Invalid credentials", "danger")

    return render_template("doctor_login.html")


def get_db_connection():
    conn = sqlite3.connect('users.db')  # Replace with your actual DB file
    conn.row_factory = sqlite3.Row
    return conn


def send_email(recipient, subject, body):
    sender_email = "dentiprooo@gmail.com"
    sender_password = "posp xouk vrnq nqyr"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    try:
        # Set up the MIME
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = recipient
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        # Connect to the server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient, message.as_string())
        server.quit()

        print(f"✅ Email sent successfully to {recipient}")
    except Exception as e:
        print(f"❌ Error sending email: {e}")


def send_email_with_context(app, to_email, subject, body):
    with app.app_context():
        send_email(to_email, subject, body)

@app.route('/update_status/<int:appointment_id>', methods=['POST'])
def update_status(appointment_id):
    data = request.get_json()
    action = data.get('action')

    if action not in ["confirm", "cancel"]:
        return jsonify({"success": False, "message": "Invalid action"}), 400

    new_status = "Confirmed" if action == "confirm" else "Canceled"

    # ✅ Fetch appointment from the database
    appointment = Appointment.query.get(appointment_id)
    if not appointment:
        return jsonify({"success": False, "message": "Appointment not found"}), 404

    # 🔄 Debug: Print before updating
    print(f"🔄 Updating appointment {appointment_id} to {new_status}")

    # ✅ Update appointment status
    appointment.status = new_status
    db.session.commit()

    # ✅ Debug: Print after commit
    print(f"✅ Updated appointment: {appointment.status}")

    # ✅ Prepare Email message
    email_subject = "Appointment Status Update"

    if new_status == "Confirmed":
        email_body = (
            f"Dear {appointment.patient_name},\n\n"
            f"Your appointment with Dr. {appointment.doctor_name} on {appointment.date} has been confirmed.\n\n"
            f"Please be on time.\n\n"
            f"Best regards,\nDentiPro Team"
        )
    else:
        email_body = (
            f"Dear {appointment.patient_name},\n\n"
            f"Your appointment with Dr. {appointment.doctor_name} on {appointment.date} has been canceled.\n"
            f"Please book the appointment again at your convenience.\n\n"
            f"Best regards,\nDentiPro Team"
        )

    # ✅ Send Email asynchronously using threading with Flask app context
    if appointment.patient_email:
        print(f"📧 Sending email to: {appointment.patient_email}")
        threading.Thread(
            target=send_email_with_context,
            args=(app, appointment.patient_email, email_subject, email_body)
        ).start()

    return jsonify({"success": True, "message": f"Appointment {new_status} and email sent"}), 200


@app.route('/appointment_status')
@login_required  # Ensures only logged-in users can access
def appointment_status():
    # Fetch only the appointments of the logged-in patient
    appointments = Appointment.query.filter_by(patient_id=current_user.id).all()

    return render_template('appointment_status.html', appointments=appointments)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        image = request.files['image']

        # Ensure an image is uploaded
        if image.filename == '':
            return jsonify({'error': 'No image uploaded'}), 400

        # Save the image to the uploads folder
        filename = secure_filename(image.filename)
        image_path = os.path.join('static/uploads', filename)
        image.save(image_path)

        # Process the image for prediction
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels.get(predicted_class_index, "Unknown")

        # Store prediction and image path in the session
        session['predicted_condition'] = predicted_class_label
        session['uploaded_image'] = filename  # Store only filename, not full path

        return jsonify({'predicted_class': predicted_class_label, 'image_url': filename})

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


# Food recommendation page
@app.route('/recommendation')
def recommendation():
    """Generates food recommendations and displays the uploaded image."""

    # 🔹 Get condition from session
    predicted_condition = session.get('predicted_condition', '')
    uploaded_image = session.get('uploaded_image', '')

    if not predicted_condition:
        return render_template('food.html', recommendation="No condition detected. Please upload an image.", image_url=None)

    try:
        # Generate food recommendations using Gemini AI
        prompt = f"Give food recommendations (do's and don'ts) for a person suffering from {predicted_condition}. Format the response with bullet points and headings for readability."
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)

        if not response or not hasattr(response, 'text') or not response.text.strip():
            return render_template('food.html', recommendation="No recommendation generated. Try again later.", image_url=uploaded_image)

        # Format response text
        recommendation = response.text.strip()
        recommendation = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", recommendation)
        recommendation = re.sub(r"\*(.*?)\*", r"<i>\1</i>", recommendation)
        recommendation = re.sub(r"(?m)^\s*[-•*]\s+(.+)", r"<li>\1</li>", recommendation)
        recommendation = f"<ul>{recommendation}</ul>"

    except Exception as e:
        recommendation = f"Error generating recommendation: {str(e)}"

    return render_template('food.html', recommendation=recommendation, image_url=uploaded_image)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
