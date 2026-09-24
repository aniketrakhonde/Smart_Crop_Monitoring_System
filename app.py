import os
import sqlite3

import numpy as np
import requests
import tensorflow as tf
from flask import Flask, redirect, render_template, request, send_file, session, url_for
from PIL import Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Weather API key ab code me nahi, Render ke Environment Variable se aayegi
API_KEY = os.environ.get("WEATHER_API_KEY", "")
CITY = "Jalgaon"
DB_PATH = "farmers.db"
MODEL_PATH = "models/plant_disease_model.tflite"

# ---------- Model (TFLite, ek baar load hota hai) ----------
try:
    model = tf.lite.Interpreter(model_path=MODEL_PATH)
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    print("TFLITE MODEL LOADED SUCCESSFULLY", flush=True)
except Exception as e:
    print("MODEL ERROR:", e, flush=True)
    model = None

# ---------- Plant classes, treatments, medicines, fertilizers ----------
classes = [
    "Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Apple Rust",
    "Apple Healthy",
    "Blueberry Healthy",
    "Cherry Powdery Mildew",
    "Cherry Healthy",
    "Corn Cercospora Leaf Spot",
    "Corn Common Rust",
    "Corn Northern Leaf Blight",
    "Corn Healthy",
    "Grape Black Rot",
    "Grape Esca (Black Measles)",
    "Grape Leaf Blight",
    "Grape Healthy",
    "Orange Huanglongbing (Citrus Greening)",
    "Peach Bacterial Spot",
    "Peach Healthy",
    "Bell Pepper Bacterial Spot",
    "Bell Pepper Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Raspberry Healthy",
    "Soybean Healthy",
    "Squash Powdery Mildew",
    "Strawberry Leaf Scorch",
    "Strawberry Healthy",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]

treatments = {
    "Apple Scab": "Spray Mancozeb or Captan fungicide. Remove infected leaves.",
    "Apple Black Rot": "Prune infected branches and spray Copper fungicide.",
    "Apple Cedar Apple Rust": "Use Myclobutanil fungicide and remove nearby cedar trees.",
    "Apple Healthy": "Healthy crop. No treatment required.",
    "Blueberry Healthy": "Healthy crop. No treatment required.",
    "Cherry Powdery Mildew": "Apply Sulfur fungicide and improve air circulation.",
    "Cherry Healthy": "Healthy crop. No treatment required.",
    "Corn Cercospora Leaf Spot": "Use resistant varieties and spray fungicide.",
    "Corn Common Rust": "Apply fungicide if infection is severe.",
    "Corn Northern Leaf Blight": "Spray Azoxystrobin fungicide.",
    "Corn Healthy": "Healthy crop. No treatment required.",
    "Grape Black Rot": "Spray Mancozeb or Myclobutanil. Remove infected fruits.",
    "Grape Esca (Black Measles)": "Prune infected vines and maintain vineyard hygiene.",
    "Grape Leaf Blight": "Use Copper fungicide and remove infected leaves.",
    "Grape Healthy": "Healthy crop. No treatment required.",
    "Orange Huanglongbing (Citrus Greening)": "Control psyllids and remove infected trees.",
    "Peach Bacterial Spot": "Spray Copper bactericide.",
    "Peach Healthy": "Healthy crop. No treatment required.",
    "Bell Pepper Bacterial Spot": "Use Copper spray and disease-free seeds.",
    "Bell Pepper Healthy": "Healthy crop. No treatment required.",
    "Potato Early Blight": "Spray Mancozeb or Chlorothalonil.",
    "Potato Late Blight": "Spray Metalaxyl fungicide immediately.",
    "Potato Healthy": "Healthy crop. No treatment required.",
    "Raspberry Healthy": "Healthy crop. No treatment required.",
    "Soybean Healthy": "Healthy crop. No treatment required.",
    "Squash Powdery Mildew": "Spray Sulfur fungicide.",
    "Strawberry Leaf Scorch": "Remove infected leaves and spray fungicide.",
    "Strawberry Healthy": "Healthy crop. No treatment required.",
    "Tomato Bacterial Spot": "Apply Copper bactericide.",
    "Tomato Early Blight": "Spray Chlorothalonil or Mancozeb.",
    "Tomato Late Blight": "Spray Metalaxyl fungicide.",
    "Tomato Leaf Mold": "Improve ventilation and spray fungicide.",
    "Tomato Septoria Leaf Spot": "Spray Copper fungicide.",
    "Tomato Spider Mites": "Use Neem oil or miticide.",
    "Tomato Target Spot": "Apply suitable fungicide.",
    "Tomato Yellow Leaf Curl Virus": "Control whiteflies and remove infected plants.",
    "Tomato Mosaic Virus": "Remove infected plants immediately.",
    "Tomato Healthy": "Healthy crop. No treatment required."
}

medicines = {
    "Apple Scab": "Mancozeb 75% WP",
    "Apple Black Rot": "Copper Oxychloride",
    "Apple Cedar Apple Rust": "Myclobutanil",
    "Apple Healthy": "No medicine required",

    "Blueberry Healthy": "No medicine required",

    "Cherry Powdery Mildew": "Sulfur Fungicide",
    "Cherry Healthy": "No medicine required",

    "Corn Cercospora Leaf Spot": "Azoxystrobin",
    "Corn Common Rust": "Propiconazole",
    "Corn Northern Leaf Blight": "Azoxystrobin",
    "Corn Healthy": "No medicine required",

    "Grape Black Rot": "Mancozeb",
    "Grape Esca (Black Measles)": "Copper Fungicide",
    "Grape Leaf Blight": "Copper Oxychloride",
    "Grape Healthy": "No medicine required",

    "Orange Huanglongbing (Citrus Greening)": "No effective medicine",

    "Peach Bacterial Spot": "Copper Bactericide",
    "Peach Healthy": "No medicine required",

    "Bell Pepper Bacterial Spot": "Copper Spray",
    "Bell Pepper Healthy": "No medicine required",

    "Potato Early Blight": "Chlorothalonil",
    "Potato Late Blight": "Metalaxyl + Mancozeb",
    "Potato Healthy": "No medicine required",

    "Raspberry Healthy": "No medicine required",

    "Soybean Healthy": "No medicine required",

    "Squash Powdery Mildew": "Sulfur Fungicide",

    "Strawberry Leaf Scorch": "Copper Fungicide",
    "Strawberry Healthy": "No medicine required",

    "Tomato Bacterial Spot": "Copper Oxychloride",
    "Tomato Early Blight": "Mancozeb",
    "Tomato Late Blight": "Metalaxyl",
    "Tomato Leaf Mold": "Chlorothalonil",
    "Tomato Septoria Leaf Spot": "Copper Fungicide",
    "Tomato Spider Mites": "Neem Oil",
    "Tomato Target Spot": "Azoxystrobin",
    "Tomato Yellow Leaf Curl Virus": "No medicine available",
    "Tomato Mosaic Virus": "Remove infected plants",
    "Tomato Healthy": "No medicine required"
}
fertilizers = {
    "Apple Scab": "NPK 19:19:19 + Organic Compost",
    "Apple Black Rot": "Potash + Organic Compost",
    "Apple Cedar Apple Rust": "NPK 20:20:20",
    "Apple Healthy": "Vermicompost",

    "Blueberry Healthy": "Organic Compost",

    "Cherry Powdery Mildew": "NPK 19:19:19",
    "Cherry Healthy": "Farmyard Manure",

    "Corn Cercospora Leaf Spot": "Urea + Potash",
    "Corn Common Rust": "NPK 20:20:20",
    "Corn Northern Leaf Blight": "Potash",
    "Corn Healthy": "Compost",

    "Grape Black Rot": "Organic Compost",
    "Grape Esca (Black Measles)": "NPK 19:19:19",
    "Grape Leaf Blight": "Potash",
    "Grape Healthy": "Vermicompost",

    "Orange Huanglongbing (Citrus Greening)": "Micronutrient Mix",

    "Peach Bacterial Spot": "Organic Compost",
    "Peach Healthy": "Farmyard Manure",

    "Bell Pepper Bacterial Spot": "NPK 20:20:20",
    "Bell Pepper Healthy": "Compost",

    "Potato Early Blight": "Potash",
    "Potato Late Blight": "NPK 19:19:19",
    "Potato Healthy": "Organic Compost",

    "Raspberry Healthy": "Compost",
    "Soybean Healthy": "Rhizobium Biofertilizer",

    "Squash Powdery Mildew": "NPK 20:20:20",

    "Strawberry Leaf Scorch": "Organic Compost",
    "Strawberry Healthy": "Vermicompost",

    "Tomato Bacterial Spot": "Potash",
    "Tomato Early Blight": "NPK 19:19:19",
    "Tomato Late Blight": "Potash",
    "Tomato Leaf Mold": "Organic Compost",
    "Tomato Septoria Leaf Spot": "NPK 20:20:20",
    "Tomato Spider Mites": "Seaweed Fertilizer",
    "Tomato Target Spot": "Potash",
    "Tomato Yellow Leaf Curl Virus": "Micronutrient Mix",
    "Tomato Mosaic Virus": "Organic Compost",
    "Tomato Healthy": "Vermicompost"
}


# ---------- Database ----------
def get_db():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS farmers(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            mobile TEXT,
            email TEXT,
            password TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS disease_reports(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer TEXT,
            disease TEXT,
            confidence TEXT,
            treatment TEXT,
            image TEXT,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS crop_monitoring(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer TEXT,
            crop_name TEXT,
            crop_date TEXT,
            soil_type TEXT,
            location TEXT,
            image TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ---------- Auth ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        conn = get_db()
        conn.execute(
            "INSERT INTO farmers(name, mobile, email, password) VALUES (?, ?, ?, ?)",
            (
                request.form["name"],
                request.form["mobile"],
                request.form["email"],
                request.form["password"],
            ),
        )
        conn.commit()
        conn.close()
        return "Registration Successful!"
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM farmers WHERE email=? AND password=?", (email, password)
        ).fetchone()
        conn.close()

        if user:
            session["user"] = email
            return redirect(url_for("dashboard"))
        return "Invalid Email or Password!"
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ---------- Dashboard ----------
def get_weather():
    """Weather laata hai. API na chale to site band nahi hoti, N/A dikhata hai."""
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": CITY, "appid": API_KEY, "units": "metric"}
        data = requests.get(url, params=params, timeout=5).json()
        return (
            data["main"]["temp"],
            data["weather"][0]["main"],
            data["main"]["humidity"],
            data["wind"]["speed"],
        )
    except Exception as e:
        print("WEATHER ERROR:", e, flush=True)
        return "N/A", "N/A", "N/A", "N/A"


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    user = conn.execute(
        "SELECT name, email, mobile FROM farmers WHERE email=?", (session["user"],)
    ).fetchone()
    conn.close()

    if user is None:
        return "User data not found"

    temperature, condition, humidity, wind = get_weather()

    return render_template(
        "dashboard.html",
        name=user[0],
        email=user[1],
        mobile=user[2],
        temperature=temperature,
        condition=condition,
        humidity=humidity,
        wind=wind,
    )


# ---------- Reports ----------
@app.route("/report")
def report():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    reports = conn.execute(
        "SELECT * FROM disease_reports WHERE farmer=?", (session["user"],)
    ).fetchall()
    conn.close()

    return render_template("report.html", reports=reports)


@app.route("/view_report/<int:id>")
def view_report(id):
    conn = get_db()
    report = conn.execute(
        "SELECT * FROM disease_reports WHERE id=?", (id,)
    ).fetchone()
    conn.close()

    return render_template("view_report.html", report=report)


@app.route("/download_report")
def download_report():
    conn = get_db()
    reports = conn.execute("SELECT * FROM disease_reports").fetchall()
    conn.close()

    file_path = "Disease_Report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    elements = [
        Paragraph("Smart Crop Monitoring System", styles["Title"]),
        Paragraph("Disease Detection Report", styles["Heading2"]),
        Spacer(1, 20),
    ]

    data = [["ID", "Disease", "Confidence", "Treatment", "Date"]]
    for r in reports:
        data.append([str(r[0]), str(r[2]), str(r[3]), str(r[4]), str(r[5])])

    table = Table(data, colWidths=[30, 120, 80, 200, 100], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), "grey"),
        ("TEXTCOLOR", (0, 0), (-1, 0), "white"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 1, "black"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("WORDWRAP", (0, 0), (-1, -1), True),
    ]))
    elements.append(table)

    doc.build(elements)
    return send_file(file_path, as_attachment=True)


# ---------- Crop information ----------
@app.route("/crop", methods=["GET", "POST"])
def crop():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    cursor = conn.cursor()

    if request.method == "POST":
        file = request.files.get("image")
        filename = ""
        if file and file.filename:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        cursor.execute(
            """
            INSERT INTO crop_monitoring
            (farmer, crop_name, crop_date, soil_type, location, image)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session["user"],
                request.form["crop"],
                request.form["date"],
                request.form["soil"],
                request.form["location"],
                filename,
            ),
        )
        conn.commit()

    crops = cursor.execute(
        "SELECT * FROM crop_monitoring WHERE farmer=?", (session["user"],)
    ).fetchall()
    conn.close()

    return render_template(
        "crop.html",
        crops=crops,
        success="Crop information saved successfully!" if request.method == "POST" else "",
    )


# ---------- Disease detection ----------
def predict_disease(filepath):
    """Image se (disease_name, confidence_percent_text) return karta hai."""
    img = Image.open(filepath).convert("RGB").resize((224, 224), Image.NEAREST)
    x = np.expand_dims(np.asarray(img, dtype=np.float32) / 255.0, axis=0)

    model.set_tensor(input_details[0]["index"], x)
    model.invoke()
    probabilities = model.get_tensor(output_details[0]["index"])[0]

    index = int(np.argmax(probabilities))
    confidence = f"{np.max(probabilities) * 100:.2f}%"
    disease_name = classes[index] if index < len(classes) else "Unknown Disease"
    return disease_name, confidence


@app.route("/disease", methods=["GET", "POST"])
def disease():
    disease_name = "Not Detected"
    confidence = "0%"
    treatment = "No treatment available."

    file = request.files.get("image") if request.method == "POST" else None

    if file and file.filename and model is not None:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            disease_name, confidence = predict_disease(filepath)
            treatment = treatments.get(disease_name, "No treatment available.")

            conn = get_db()
            conn.execute(
                """
                INSERT INTO disease_reports (farmer, disease, confidence, treatment, image)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session.get("user", "Unknown"), disease_name, confidence, treatment, filename),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print("PREDICTION ERROR:", e, flush=True)
            disease_name = "Analysis Error"
            treatment = "Please try uploading a clearer crop image."

    return render_template(
        "disease.html",
        disease=disease_name,
        confidence=confidence,
        treatment=treatment,
    )


# ---------- Treatment info ----------
@app.route("/treatment")
def treatment():
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "treatment.html",
        treatments=treatments,
        medicines=medicines,
        fertilizers=fertilizers,
    )


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")