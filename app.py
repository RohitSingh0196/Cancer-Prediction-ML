from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
features = [
'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
'compactness_mean','concavity_mean','concave points_mean','symmetry_mean',
'fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se',
'smoothness_se','compactness_se','concavity_se','concave points_se',
'symmetry_se','fractal_dimension_se','radius_worst','texture_worst',
'perimeter_worst','area_worst','smoothness_worst','compactness_worst',
'concavity_worst','concave points_worst','symmetry_worst',
'fractal_dimension_worst'
]

# Load model and scaler
model = joblib.load("models/cancer_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html", features=features)

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = [float(x) for x in request.form.values()]

        data = np.array(data).reshape(1, -1)

        data = scaler.transform(data)

        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "✅ Benign (Non-Cancerous)"
        else:
            result = "❌ Malignant (Cancerous)"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)