# CardioGuard AI — HistGradientBoosting + Streamlit

## Files

- `app.py` — Streamlit web application.
- `train_model.py` — trains/tunes HistGradientBoosting and creates `model.pkl` + `metrics.json`.
- `cardio_train.csv` — training dataset.
- `model.pkl` — trained model bundle.
- `metrics.json` — saved evaluation metrics.
- `requirements.txt` — Streamlit Cloud dependencies.

## Local setup

```bash
pip install -r requirements.txt
python train_model.py cardio_train.csv
streamlit run app.py
```

## Streamlit Cloud

Upload all files in this folder to your GitHub repository and set the main file to:

`app.py`

The repository should contain `model.pkl`, so Streamlit Cloud can load the already-trained model without retraining every time the app starts.

## Notes

- HistGradientBoosting is used instead of Logistic Regression.
- StandardScaler and LabelEncoder are not needed because the dataset's categorical variables are already numeric and tree-based boosting does not require feature scaling.
- No noise score, noise calculation, or IQR-based outlier filtering is performed.
- Missing numeric values are handled with median imputation inside the training pipeline.
- `id` is excluded from the model.
- The original dataset stores age in days; the model uses age in years.
