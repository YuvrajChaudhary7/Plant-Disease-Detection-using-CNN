from flask import Flask, render_template, request, send_from_directory
import os
import webbrowser
from threading import Timer
from predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Treatment suggestions
treatments = {

"Tomato Early blight":
"Remove infected leaves and apply fungicides such as chlorothalonil.",

"Tomato Late blight":
"Use copper-based fungicides and avoid overhead watering.",

"Tomato healthy":
"The plant is healthy. Maintain proper watering and fertilization.",

"Unknown / Not a Tomato Leaf":
"The uploaded image does not appear to be a tomato leaf."
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files['file']

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    file.save(filepath)

    disease, confidence = predict_image(filepath)

    treatment = treatments.get(disease,"No treatment available")

    return render_template(
        "result.html",
        result=disease,
        confidence=confidence,
        treatment=treatment,
        image=file.filename
    )


# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Auto open browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == "__main__":

    Timer(1, open_browser).start()

    app.run(debug=True, use_reloader=False)
    