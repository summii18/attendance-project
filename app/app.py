from flask import Flask, render_template, request
from PIL import Image
import io
from detect import recognize_student

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    student, prob = recognize_student(image)
    
    return {
        "student": student,
        "probability": prob
    }

if __name__ == "__main__":
    app.run(debug=True)