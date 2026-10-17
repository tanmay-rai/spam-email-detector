# Email Spam & Phishing Detector

A machine learning application that classifies emails as **Benign** or **Spam/Phishing** using TF-IDF and Multinomial Naive Bayes. The project includes an interactive Streamlit dashboard, SHAP-based model explanations, dataset analysis, and optional AI-powered email rewriting using Ollama.

## Features

- **Email Classification:** Predict whether an email is benign or spam/phishing.
- **Prediction Probabilities:** View the model's estimated probabilities for both classes.
- **Model Explainability:** Use SHAP to explore which words influence a prediction.
- **Dataset Analysis:** Visualize class distribution, email lengths, and basic data-quality statistics.
- **AI Email Rewriting:** Optionally rewrite email text using a locally running Ollama model and classify the rewritten text.
- **Machine Learning Pipeline:** Combine TF-IDF text vectorization and Multinomial Naive Bayes in a single scikit-learn pipeline.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- SHAP
- Joblib
- PyArrow
- Ollama (optional)

## Project Structure

```text
spam-email-detector/
├── data/
│   ├── train.parquet       # Download separately
│   └── test.parquet        # Download separately
├── app.py                  # Streamlit application
├── main.py                 # Model training and evaluation
├── spam_pipeline.joblib    # Generated after training
├── requirements.txt
├── README.md
└── .gitignore
```

The dataset files and trained model are not included in this repository. Follow the setup instructions below to obtain the data and generate the model.

## Dataset

This project uses the [Seven Phishing Email Datasets collection on Hugging Face](https://huggingface.co/datasets/mamtakumar/seven-phishing-email-datasets).

The dataset contains email text, labels, and additional metadata. This project uses the `text` and `label` columns.

| Label | Meaning |
|---|---|
| `0` | Benign |
| `1` | Spam/Phishing |

### Download the dataset

1. Open the [dataset page](https://huggingface.co/datasets/mamtakumar/seven-phishing-email-datasets).
2
