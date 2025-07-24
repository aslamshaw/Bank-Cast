# Bank-Cast 📊

This project builds a **complete machine learning pipeline** to predict whether a customer will subscribe to a term deposit based on their interaction history in a marketing campaign.

---

## 🚀 Features

- **Full ML Pipeline**:
  - Custom imputers and encoders (`SimpleImputer`, `LeaveOneOut`, `OneHot`)
  - Feature engineering with custom ratios and binning
  - Mutual Information-based feature selection
  - Automated scaling & encoding
- **Model Ensemble**:
  - Random Forest, Gradient Boosting, XGBoost, SVM
  - Stacked with Logistic Regression as meta-learner
- **FastAPI Backend**:
  - `/predict` endpoint for live inference
  - `/health` endpoint for service status
- **Serialized Stack**:
  - Trained model + label encoder saved via Pickle
- **Model Explainability**:
  - **SHAP values** for local/global insights
  - Feature importance visualizations
- **Deployment Ready**:
  - Trained ensemble + encoders serialized with Pickle
  - Modular codebase for easy experimentation

---

# 📁 Project Structure

Use this format inside the markdown file (README.md), not inside a comment block.
But if you insist on keeping it here, use indentation:

    Trait-Smith/
    ├── ml_pipeline.py                   # Full data preprocessing pipeline
    ├── train_and_save_model.py          # Training logic and model serialization
    ├── app.py                           # FastAPI app with /predict route
    ├── stacking_model.pkl               # Trained ensemble model (after training)
    ├── label_encoder.pkl                # Label encoder for target class decoding
    ├── feature_importance.png           # Bar chart of feature importances
    ├── shap_summary_all_classes.png     # SHAP summary plot (all classes)
    ├── shap_summary_all_classes.png     # SHAP subscribed class plot
    └── bank_marketing.csv               # Raw marketing dataset

---

## 🧪 Getting Started

### 1. Clone the repo

    git clone https://github.com/yourusername/Bank-Cast.git
    cd Bank-Cast

### 2. Install requirements

    pip install -r requirements.txt

### 3. Train the model

Ensure `marketing_dataset.csv` is available at the path hardcoded in `train_and_save_model.py`:

    os.path.join(os.path.dirname(__file__), 'marketing_dataset.csv')

Adjust the path if needed. Then run:

    python train_and_save_model.py

This will generate `stacking_model.pkl` and `label_encoder.pkl`.

### 4. Run the API

    uvicorn app:app --reload

---

## 🎯 Usage

### POST /predict

Send JSON like:

    {
      "age": 45,
      "job": "blue-collar",
      "marital": "married",
      "education": "secondary",
      "default": "no",
      "balance": 1200,
      "housing": "yes",
      "loan": "no",
      "contact": "cellular",
      "day": 5,
      "month": "may",
      "duration": 230,
      "campaign": 2,
      "pdays": -1,
      "previous": 0,
      "poutcome": "unknown"
    }

Response:

    {
      "prediction": "no"
    }

### GET /health

    { "status": "ok" }

---

## 📌 Notes

- Leave-One-Out Encoding applied only to high-cardinality categorical features
- Mutual Information feature selector uses threshold tuning during CV
- Input format is strict — any schema drift will cause prediction to fail

---

## 🧠 Future Enhancements

- Add input schema validation
- Extend Swagger UI docs
- Add model version tracking and experiment logging
- Dockerize + CI/CD for production deployment

---

## 🛡️ License

MIT License — free to use, modify responsibly.
