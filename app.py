from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load trained machine learning model
MODEL_FILE = "spam_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    raise FileNotFoundError(
        "Model files not found. Please run 'python train_model.py' first."
    )

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

with open(VECTORIZER_FILE, "rb") as file:
    vectorizer = pickle.load(file)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Email prediction
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    message = data.get("message", "").strip()

    if not message:
        return jsonify({
            "error": "Please enter an email message."
        }), 400

    # Convert email text into numerical features
    email_vector = vectorizer.transform([message])

    # Predict spam or ham
    prediction = model.predict(email_vector)[0]

    # Calculate confidence
    confidence = model.predict_proba(email_vector).max() * 100

    if prediction == "spam":
        result = "SPAM EMAIL"
        status = "spam"
    else:
        result = "NOT SPAM"
        status = "ham"

    return jsonify({
        "result": result,
        "status": status,
        "confidence": round(confidence, 2)
    })


# Start Flask application
if __name__ == "__main__":
    app.run(debug=True)