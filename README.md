# Cat vs Dog Classification (Transfer Learning + Streamlit + FastAPI)

## Overview

Binary image classification system that detects **Cat vs Dog** using a trained deep learning model.
Includes:

* Streamlit app (UI)
* FastAPI backend (API inference)
* Pretrained model (`.keras`)

---

## Repository Structure

```
.
├── app.py              # Streamlit frontend
├── main.py             # FastAPI backend
├── my_model.keras      # Trained model
├── uploads/            # Temporary uploaded images
├── requirements.txt    # Dependencies
├── runtime.txt         # Deployment runtime (for cloud)
└── README.md
```

---

## Features

* Upload image and classify (Streamlit UI)
* REST API for predictions (FastAPI)
* Transfer learning-based model
* Image preprocessing pipeline (224x224, normalized)

---

## Installation

### 1. Clone repository

```
git clone https://github.com/joshibibek91/transferlearning_dogvscat_classification.git
cd transferlearning_dogvscat_classification
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## Run Applications

### Run Streamlit App (Frontend)

```
streamlit run app.py
```

### Run FastAPI Server (Backend)

```
uvicorn main:app --reload
```

FastAPI will run at:

```
http://127.0.0.1:8000
```

---

## API Usage

### Endpoint

```
POST /predict
```

### Input

* Image file (multipart/form-data)

### Output

```
{
  "prediction": "Cat"
}
```

---

## Model Details

* Architecture: Transfer Learning (CNN-based)
* Input Size: 224 × 224 × 3
* Output:

  * 0 → Cat
  * 1 → Dog
* Format: Keras `.keras`

---

## Workflow

1. Image uploaded (UI or API)
2. Resized to 224×224
3. Normalized (0–1 scale)
4. Passed to model
5. Argmax → class label

---

## Deployment Ready

* `requirements.txt` for dependencies
* `runtime.txt` for cloud environments (Heroku/Streamlit Cloud)

---

## Improvements (Next Steps)

* Add confidence score
* Add model versioning
* Dockerize (FastAPI + Streamlit)
* Add logging + error handling
* Extend to multi-class classification

---

## Author

Bibek Joshi
