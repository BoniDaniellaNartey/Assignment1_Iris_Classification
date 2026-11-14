
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


# ==========================================
# 1. APPLICATION SETUP
# ==========================================

app = FastAPI(
    title="Iris Flower Classification",
    description="Machine Learning API for Iris flower prediction",
    version="1.0.0"
)


# ==========================================
# 2. LOAD THE TRAINED MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "notebooks" / "iris_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("Iris classification model loaded successfully!")


# ==========================================
# 3. FLOWER SPECIES
# ==========================================

species_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}


# ==========================================
# 4. INPUT VALIDATION
# ==========================================

class IrisInput(BaseModel):

    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


# ==========================================
# 5. HOME PAGE - WEB INTERFACE
# ==========================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Iris Classification | ML Application</title>

    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: Arial, sans-serif;
        }

        body {
            background: #f4f7fb;
            color: #1e293b;
            min-height: 100vh;
        }

        header {
            background: #102c3c;
            color: white;
            padding: 22px 8%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }

        .brand {
            font-size: 21px;
            font-weight: bold;
        }

        .brand span {
            color: #54d6b0;
        }

        .status {
            background: #1d4b4b;
            color: #8ef0d1;
            padding: 8px 14px;
            border-radius: 20px;
            font-size: 12px;
        }

        .container {
            max-width: 1100px;
            margin: 55px auto;
            padding: 0 20px;
        }

        .intro {
            margin-bottom: 30px;
        }

        .intro h1 {
            font-size: 32px;
            margin-bottom: 12px;
            color: #102c3c;
        }

        .intro p {
            color: #64748b;
            line-height: 1.7;
        }

        .dashboard {
            display: grid;
            grid-template-columns: 1.3fr 0.7fr;
            gap: 25px;
            align-items: start;
        }

        .card {
            background: white;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 5px 25px rgba(15, 40, 60, 0.06);
        }

        .card h2 {
            font-size: 20px;
            margin-bottom: 10px;
            color: #102c3c;
        }

        .description {
            color: #64748b;
            font-size: 14px;
            margin-bottom: 25px;
            line-height: 1.6;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .form-group label {
            display: block;
            font-size: 13px;
            font-weight: bold;
            margin-bottom: 9px;
            color: #334155;
        }

        .form-group input {
            width: 100%;
            padding: 14px;
            border: 1px solid #dbe3eb;
            border-radius: 9px;
            font-size: 15px;
            outline: none;
        }

        .form-group input:focus {
            border-color: #159c83;
            box-shadow: 0 0 0 3px rgba(21, 156, 131, 0.10);
        }

        .unit {
            font-size: 11px;
            color: #94a3b8;
            margin-top: 6px;
        }

        .buttons {
            display: flex;
            gap: 12px;
            margin-top: 28px;
        }

        button {
            border: none;
            border-radius: 9px;
            padding: 14px 20px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
        }

        .predict-btn {
            background: #087f70;
            color: white;
            flex: 1;
        }

        .predict-btn:hover {
            background: #06685c;
        }

        .reset-btn {
            background: #edf2f7;
            color: #334155;
        }

        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .result-card {
            background: #102c3c;
            color: white;
            min-height: 310px;
        }

        .result-card h2 {
            color: white;
        }

        .result-placeholder {
            margin-top: 45px;
            text-align: center;
            color: #bdcbd4;
            line-height: 1.7;
        }

        .flower-icon {
            font-size: 55px;
            margin-bottom: 15px;
        }

        .result-label {
            color: #9cb1be;
            font-size: 13px;
            margin-top: 30px;
        }

        .result-name {
            color: #65e0ba;
            font-size: 30px;
            font-weight: bold;
            margin: 10px 0;
        }

        .confidence {
            color: #d3e3e9;
            font-size: 14px;
            margin-top: 12px;
        }

        .error {
            color: #ffb5b5;
            margin-top: 18px;
            font-size: 14px;
        }

        .info-section {
            margin-top: 30px;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 18px;
        }

        .info-box {
            background: white;
            padding: 22px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 3px 15px rgba(15, 40, 60, 0.04);
        }

        .info-box h3 {
            color: #087f70;
            font-size: 23px;
            margin-bottom: 7px;
        }

        .info-box p {
            color: #64748b;
            font-size: 13px;
        }

        footer {
            text-align: center;
            padding: 30px 15px;
            color: #94a3b8;
            font-size: 12px;
        }

        @media(max-width: 750px) {

            .dashboard {
                grid-template-columns: 1fr;
            }

            .form-grid {
                grid-template-columns: 1fr;
            }

            .info-section {
                grid-template-columns: 1fr;
            }

            .intro h1 {
                font-size: 26px;
            }

            .container {
                margin-top: 35px;
            }

        }

    </style>

</head>

<body>

<header>

    <div class="brand">
        IRIS<span>CLASSIFY</span>
    </div>

    <div class="status">
        ● Machine Learning Model Active
    </div>

</header>


<main class="container">

    <section class="intro">

        <h1>Iris Flower Classification</h1>

        <p>
            Welcome to the Iris Flower Classification System.
            Enter the measurements of an Iris flower below
            to predict its species using a trained
            Machine Learning model.
        </p>

    </section>


    <section class="dashboard">

        <div class="card">

            <h2>Flower Measurements</h2>

            <p class="description">
                Enter the four measurements in centimetres
                to classify your Iris flower.
            </p>


            <form id="predictionForm">

                <div class="form-grid">

                    <div class="form-group">

                        <label for="sepal_length">
                            Sepal Length
                        </label>

                        <input
                            type="number"
                            id="sepal_length"
                            step="any"
                            min="0.1"
                            placeholder="Enter sepal length"
                            required
                        >

                        <div class="unit">Centimetres (cm)</div>

                    </div>


                    <div class="form-group">

                        <label for="sepal_width">
                            Sepal Width
                        </label>

                        <input
                            type="number"
                            id="sepal_width"
                            step="any"
                            min="0.1"
                            placeholder="Enter sepal widths"
                            required
                        >

                        <div class="unit">Centimetres (cm)</div>

                    </div>


                    <div class="form-group">

                        <label for="petal_length">
                            Petal Length
                        </label>

                        <input
                            type="number"
                            id="petal_length"
                            step="any"
                            min="0.1"
                            placeholder="Enter petal length"
                            required
                        >

                        <div class="unit">Centimetres (cm)</div>

                    </div>


                    <div class="form-group">

                        <label for="petal_width">
                            Petal Width
                        </label>

                        <input
                            type="number"
                            id="petal_width"
                            step="any"
                            min="0.1"
                            placeholder="Enter petal width"
                            required
                        >

                        <div class="unit">Centimetres (cm)</div>

                    </div>

                </div>


                <div class="buttons">

                    <button
                        type="submit"
                        class="predict-btn"
                        id="predictButton"
                    >
                        Predict Species
                    </button>

                    <button
                        type="reset"
                        class="reset-btn"
                    >
                        Reset
                    </button>

                </div>

            </form>

        </div>


        <div class="card result-card">

            <h2>Prediction Result</h2>

            <div id="resultArea">

                <div class="result-placeholder">

                    <div class="flower-icon">🌸</div>

                    <p>
                        Your prediction will appear here
                        after you submit the flower measurements.
                    </p>

                </div>

            </div>

        </div>

    </section>


    <section class="info-section">

        <div class="info-box">

            <h3>3</h3>
            <p>Flower Species</p>

        </div>

        <div class="info-box">

            <h3>4</h3>
            <p>Input Measurements</p>

        </div>

        <div class="info-box">

            <h3>ML</h3>
            <p>Logistic Regression Model</p>

        </div>

    </section>

</main>


<footer>

    Iris Flower Classification System |
    Powered by Python, FastAPI and Machine Learning

</footer>


<script>

    const form = document.getElementById("predictionForm");

    const resultArea = document.getElementById("resultArea");

    const predictButton = document.getElementById("predictButton");


    form.addEventListener("submit", async function(event) {

        event.preventDefault();

        predictButton.disabled = true;
        predictButton.textContent = "Processing...";

        resultArea.innerHTML = `
            <div class="result-placeholder">
                <div class="flower-icon">⏳</div>
                <p>Analysing flower measurements...</p>
            </div>
        `;

        const inputData = {

            sepal_length: Number(document.getElementById("sepal_length").value),

            sepal_width: Number(document.getElementById("sepal_width").value),

            petal_length: Number(document.getElementById("petal_length").value),

            petal_width: Number(document.getElementById("petal_width").value)

        };


        try {

            const response = await fetch("/predict", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(inputData)

            });


            const data = await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail || "Prediction failed. Please try again."
                );

            }


            let confidenceText = "Confidence score unavailable";

            if (data.confidence !== null) {

                confidenceText =
                    "Model confidence: " +
                    data.confidence.toFixed(2) + "%";

            }


            resultArea.innerHTML = `

                <div class="result-label">
                    PREDICTED FLOWER SPECIES
                </div>

                <div class="flower-icon">🌼</div>

                <div class="result-name">
                    ${data.prediction}
                </div>

                <div class="confidence">
                    ${confidenceText}
                </div>

            `;

        }

        catch(error) {

            resultArea.innerHTML = `
                <div class="error">
                    ${error.message}
                </div>
            `;

        }

        finally {

            predictButton.disabled = false;
            predictButton.textContent = "Predict Species";

        }

    });


    form.addEventListener("reset", function() {

        resultArea.innerHTML = `
            <div class="result-placeholder">
                <div class="flower-icon">🌸</div>
                <p>Your prediction will appear here after you submit the flower measurements.</p>
            </div>
        `;

    });

</script>

</body>
</html>
"""


# ==========================================
# 6. PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(data: IrisInput):

    # Arrange the input using the original
    # Iris dataset feature names.

    features = pd.DataFrame([{

        "sepal length (cm)": data.sepal_length,

        "sepal width (cm)": data.sepal_width,

        "petal length (cm)": data.petal_length,

        "petal width (cm)": data.petal_width

    }])

    # Make prediction

    prediction = model.predict(features)

    predicted_label = int(prediction[0])

    predicted_species = species_names.get(
        predicted_label,
        "Unknown"
    )

    # Calculate confidence if the model supports it

    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(features)[0]

        confidence = float(max(probabilities) * 100)

    return {

        "prediction": predicted_species,

        "class_label": predicted_label,

        "confidence": confidence

    }


# ==========================================
# 7. HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "model_loaded": True,

        "model": "Iris Logistic Regression"

    }
