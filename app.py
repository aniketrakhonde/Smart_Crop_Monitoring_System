from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import os
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.secret_key = "smartcrop123"

API_KEY = "243e5860b6443601e31a1423665b8f43" 
CITY = "Jalgaon"

from tensorflow.keras.models import load_model

try:
    model = load_model(
    "models/plant_disease_model.h5",compile=False)
   
    print("MODEL LOADED SUCCESSFULLY")
except Exception as e:
    print("MODEL ERROR:", e)
    model = None

# Humare specific plant classes aur unke treatments
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



app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def init_db():
    conn = sqlite3.connect("farmers.db")
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
    date TIMESTAMP DEFAULT 
CURRENT_TIMESTAMP
)
""")
    
    cur.execute("""
CREATE TABLE IF NOT EXISTS crop_information(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    farmer TEXT,
    crop_name TEXT,
    crop_date TEXT,
    soil_type TEXT,
    location TEXT,
    image TEXT,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
    cur.execute("""
CREATE TABLE IF NOT EXISTS crop_monitoring (
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        mobile = request.form['mobile']
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect("farmers.db")
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO farmers(name, mobile, email, password) VALUES (?, ?, ?, ?)",
            (name, mobile, email, password)
        )
        conn.commit()
        conn.close()
        return "Registration Successful!"
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect("farmers.db")
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM farmers WHERE email=? AND password=?",
            (email, password)
        )
        user = cur.fetchone()
        conn.close()

        if user:
            session["user"] = email
            return redirect(url_for("dashboard"))
        else:
            return "Invalid Email or Password!"
    return render_template("login.html")

@app.route('/dashboard')
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect("farmers.db")
    cur = conn.cursor()

    cur.execute(
        "SELECT name, email, mobile FROM farmers WHERE email=?",
        (session["user"],)
    )
    print("Login Email:",session["user"])

    user = cur.fetchone()
    print(user)

    print("Database user:", user)

    if user is None:
        return "User data not found"
    conn.close()

    url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()
    humidity = data["main"]["humidity"]
    wind =data["wind"]["speed"]
    print(data)

    temperature = data["main"]["temp"]
    condition = data["weather"][0]["main"]

    return render_template(
        "dashboard.html",
        name=user[0],
        email=user[1],
        mobile=user[2],
        temperature=temperature,
        condition=condition,
        humidity=humidity,
        wind=wind
    )

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))
@app.route("/report")
def report():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect("farmers.db")
    cur = conn.cursor()

    cur.execute(
    "SELECT * FROM disease_reports WHERE farmer=?",
    (session["user"],)
)
              
    reports = cur.fetchall()
    conn.close()

    return render_template("report.html", reports=reports)
@app.route('/view_report/<int:id>')
def view_report(id):

    conn = sqlite3.connect("farmers.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM disease_reports WHERE id=?",
        (id,)
    )

    report = cursor.fetchone()

    conn.close()

    return render_template("view_report.html",
    report=report)

@app.route("/crop", methods=["GET", "POST"])
def crop():
    conn = sqlite3.connect("farmers.db")
    cursor = conn.cursor()

    if request.method == "POST":
        crop_name = request.form["crop"]
        crop_date = request.form["date"]
        soil_type = request.form["soil"]
        location = request.form["location"]

        file = request.files.get("image")
        filename = ""

        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        cursor.execute("""
        INSERT INTO crop_monitoring
        (farmer, crop_name, crop_date, soil_type, location, image)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (session["user"], crop_name, crop_date, soil_type, location, filename))

        conn.commit()

    cursor.execute("SELECT * FROM crop_monitoring WHERE farmer=?", (session["user"],))
    crops = cursor.fetchall()
    conn.close()

    return render_template(
        "crop.html",
        crops=crops,
        success="Crop information saved successfully!" if request.method == "POST" else ""
    )
@app.route("/disease", methods=["GET", "POST"])
def disease():
    disease_name = "Not Detected"
    confidence = "0%"
    treatment = "No treatment available."
    
    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            if model is not None:
                try:
                    # Model ke liye image format set karna (224x224)
                    img = image.load_img(filepath, target_size=(224, 224))
                    img.save("test_image.jpg")
                    x = image.img_to_array(img)
                    x=x / 255.0
                    x = np.expand_dims(x, axis=0)
                    print("Image Shape:", x.shape)
                    print("Image min:", np.min(x))
                    print("Image max:", np.max(x))
                    

                    # Asli AI Hub Model Prediction
                    preds = model(x)
                    predictions = preds.numpy()[0]

                    top5 = predictions.argsort()[-5:][::-1]

                    print("Top 5 Predictions:")
                    for i in top5:
                      print(i, classes[i], predictions[i])
                    predictions = preds.numpy()[0]

                    print("MAX PROBABILITY:",np.max(predictions))
                    print("Top 5 Predictions:")
                    top5 = predictions.argsort()[-5:][::-1]

                    for i in top5:
                        
                        print(classes[i], predictions[i])
                    probabilities = predictions
                    index = np.argmax(probabilities)
                    print("PREDICTED INDEX:", index)
                    print("PROBABILITIES:", probabilities)

                    
                    # Result calculate karna
                    confidence = f"{np.max(probabilities)*100:.2f}%"
                    
                    if index < len(classes):
                       disease_name = classes[index]
                       treatment = treatments.get(disease_name, "No treatment available.")
                       print("Predicted Class:",classes[index])
                    else:
                       disease_name = "Unknown Disease"

                    
                        # Save disease report
                    conn = sqlite3.connect("farmers.db")
                    cur = conn.cursor()

                    cur.execute("""
                    INSERT INTO disease_reports
                    (farmer, disease, confidence, treatment,image) 
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                      session.get("user", "Unknown"),
                      disease_name,
                      confidence,
                      treatment,
                      filename
                    ))

                    conn.commit()
                    conn.close()


                except Exception as e:
                 print("PREDICTION ERROR:", str(e))
                 import traceback
                 traceback.print_exc()

                 disease_name = "Analysis Error"
                 treatment = "Please try uploading a clearer crop image."

    return render_template(
        "disease.html",
        disease=disease_name,
        confidence=confidence,
        treatment=treatment
    )
from flask import send_file
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import sqlite3
import os

@app.route("/treatment")
def treatment():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "treatment.html",
        treatments=treatments,
        medicines=medicines,
        fertilizers=fertilizers
    )
@app.route('/download_report')
def download_report():

    conn = sqlite3.connect("farmers.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM disease_reports")
    reports = cursor.fetchall()

    conn.close()

    file_path = "Disease_Report.pdf"

    doc = SimpleDocTemplate(file_path)

    elements = []

    styles = getSampleStyleSheet()

    # Main Title
    title = Paragraph(
        "Smart Crop Monitoring System",
        styles["Title"]
    )
    elements.append(title)

    subtitle = Paragraph(
        "Disease Detection Report",
        styles["Heading2"]
    )
    elements.append(subtitle)

    elements.append(Spacer(1,20))


    data = [
        ["ID","Disease","Confidence","Treatment","Date"]
    ]


    for r in reports:
        data.append([
            str(r[0]),
            str(r[2]),
            str(r[3]),
            str(r[4]),
            str(r[5])
        ])


    table = Table(
        data,
        colWidths=[30,120,80,200,100],
        repeatRows=1
    )


    table.setStyle(TableStyle([

        ('BACKGROUND',(0,0),(-1,0),'grey'),

        ('TEXTCOLOR',(0,0),(-1,0),'white'),

        ('ALIGN',(0,0),(-1,-1),'CENTER'),

        ('GRID',(0,0),(-1,-1),1,'black'),

        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),

        ('FONTSIZE',(0,0),(-1,-1),7),

        ('WORDWRAP',(0,0),(-1,-1),True),

    ]))


    elements.append(table)

    doc.build(elements)


    return send_file(
        file_path,
        as_attachment=True
    )


if __name__ == '__main__':
    app.run(debug=True)