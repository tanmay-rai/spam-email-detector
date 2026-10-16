# Email Spam & Phishing Detector

A machine learning project that classifies emails as **Benign** or **Spam/Phishing** using TF-IDF and Multinomial Naive Bayes. It includes an interactive Streamlit interface, SHAP-based explanations, and an optional local AI email rewriting feature.

## Features

- **Email Classification:** Predicts whether an email is benign or spam/phishing.
- **Probability Scores:** Displays the model's estimated probabilities for both classes.
- **Model Explainability:** Uses SHAP to visualize which words influence the prediction.
- **Dataset Analysis:** Explores class distribution, email lengths, and data quality.
- **AI Email Rewriting:** Uses a locally running Ollama model to rewrite an email and classify the rewritten version.
- **Machine Learning Pipeline:** Combines TF-IDF text vectorization with Multinomial Naive Bayes.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- SHAP
- Joblib
- PyArrow
- Ollama

## Project Structure

```text
SpamMsg/
├── data/
│   ├── train.parquet
│   └── test.parquet
├── app.py
├── main.py
├── spam_pipeline.joblib
├── requirements.txt
├── README.md
└── .gitignore
```

The dataset files and trained model are excluded from Git to avoid committing large files.

## Dataset

The project uses a collection of phishing email datasets available through Hugging Face:

[Seven Phishing Email Datasets — mamtakumar](https://huggingface.co/datasets/mamtakumar/seven-phishing-email-datasets)

The dataset contains email text and labels, among other fields. The classifier uses the email `text` column.

- `0` — Benign
- `1` — Spam/Phishing

Place the required `train.parquet` and `test.parquet` files inside the `data/` directory before running the project.

## Machine Learning Approach

1. Load and clean the email dataset.
2. Separate email text and labels.
3. Split the training data into training and validation sets.
4. Convert email text into numerical features using **TF-IDF**.
5. Train a **Multinomial Naive Bayes** classifier.
6. Evaluate the model on validation data and a separate test dataset.
7. Save the trained pipeline using Joblib.

The text vectorizer and classifier are combined into a Scikit-learn pipeline.

### Model Performance

| Evaluation | Accuracy |
|---|---:|
| Validation | 94.65% |
| Test | 94.85% |

These are the results obtained during this project's evaluation. Accuracy alone does not capture every type of classification error.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/spam-email-detector.git
cd spam-email-detector
```

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place `train.parquet` and `test.parquet` inside the `data/` directory.

### 5. Train the model

Run the training script:

```bash
python main.py
```

The script should generate `spam_pipeline.joblib` when training and saving complete successfully.

### 6. Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## SHAP Explainability

The application uses SHAP to explain individual email predictions by visualizing word-level contributions. The chart helps show which words increase or decrease the model's spam score.

These explanations describe the model's behavior; they do not establish whether an email is genuinely malicious.

## Local AI Integration

The optional rewriting feature uses the Ollama Python client and the local `qwen3:4b` model.

To use this feature, install Ollama separately, ensure the model is available locally, and run the Ollama service. The rewriting feature is separate from model training and does not retrain the classifier.

## Limitations

- Predictions depend on the dataset and patterns learned during training.
- Accuracy does not guarantee reliable detection of every phishing email.
- Rewriting an email may change its classification, but it does not establish that the original email is safe.
- The local AI rewriting feature requires additional setup and computational resources.

## Future Improvements

- Evaluate precision, recall, and F1-score in more detail.
- Test the classifier on additional real-world email datasets.
- Improve detection of previously unseen phishing patterns.
- Explore additional models and feature-engineering approaches.

## Author

**Tanmay Rai**

Machine Learning Project — Email Spam & Phishing Detection
