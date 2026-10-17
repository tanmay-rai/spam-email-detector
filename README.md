# Email Spam & Phishing Detector

A machine learning application that classifies emails as **Benign** or **Spam/Phishing** using TF-IDF and Multinomial Naive Bayes. It includes an interactive Streamlit dashboard, SHAP-based explanations, dataset analysis, and optional AI-powered email rewriting using Ollama.

## Features

- **Email Classification:** Classify email text as benign or spam/phishing.
- **Prediction Probabilities:** View the model's estimated probabilities for each class.
- **Model Explainability:** Use SHAP to explore word-level contributions to predictions.
- **Dataset Analysis:** Visualize email class distribution, email lengths, and basic data-quality statistics.
- **AI Email Rewriting:** Optionally rewrite email text using a local Ollama model and classify the rewritten version.
- **Machine Learning Pipeline:** Combine TF-IDF vectorization and Multinomial Naive Bayes in a scikit-learn pipeline.

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
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

The dataset and generated model are not stored in this repository because of their file sizes. The trained model is generated when you run the training script.

## Dataset

This project uses the [Seven Phishing Email Datasets collection on Hugging Face](https://huggingface.co/datasets/mamtakumar/seven-phishing-email-datasets).

The dataset contains email text, labels, and additional metadata. This project uses the `text` and `label` columns.

### Download the required files

Download both files using the links below:

- **[Download train.parquet](https://huggingface.co/datasets/mamtakumar/seven-phishing-email-datasets/resolve/main/train.parquet?download=true)**
- **[Download test.parquet](https://huggingface.co/datasets/mamtakumar/seven-phishing-email-datasets/resolve/main/test.parquet?download=true)**

If a direct download does not start, open the [dataset files page](https://huggingface.co/datasets/mamtakumar/seven-phishing-email-datasets/tree/main) and select the required files.

After downloading, create a folder named `data` in the project root and place both files inside it.

Your folder should look like this:

```text
spam-email-detector/
├── data/
│   ├── train.parquet
│   └── test.parquet
├── app.py
├── main.py
└── requirements.txt
```

Keep the original filenames unchanged.

**Label mapping used by this project:**

| Label | Meaning |
|---|---|
| `0` | Benign |
| `1` | Spam/Phishing |

Check the dataset's documentation and license information before redistributing its files.

## Machine Learning Workflow

1. Load and clean the email dataset.
2. Separate email text and labels.
3. Split the training data into training and validation sets.
4. Convert email text into numerical features using TF-IDF.
5. Train a Multinomial Naive Bayes classifier.
6. Evaluate the classifier using validation and test data.
7. Save the fitted pipeline using Joblib.
8. Load the saved pipeline in the Streamlit application to classify new messages.

The pipeline combines the fitted TF-IDF vectorizer and the trained classifier so the application can reuse the same preprocessing and model.

## Model Performance

The following results were recorded during the project's evaluation:

| Evaluation | Accuracy |
|---|---:|
| Validation | 94.65% |
| Test | 94.85% |

Accuracy alone does not capture every type of classification error. Precision, recall, F1-score, and the confusion matrix provide additional information about false positives and missed phishing emails.

These figures are results from this project's evaluation, not a guarantee of performance on unseen real-world emails.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/tanmay-rai/spam-email-detector.git
cd spam-email-detector
```

### 2. Create and activate a virtual environment

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the dataset

Follow the [dataset download instructions](#dataset) and make sure both files are inside the `data/` folder.

### 5. Train the model

Run:

```bash
python main.py
```

This script loads the dataset, trains and evaluates the classifier, and saves the trained pipeline as `spam_pipeline.joblib` when execution completes successfully.

Wait for training to finish without errors before continuing.

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

The application loads the saved model and the training dataset for its prediction and analysis features.

## SHAP Explainability

SHAP is used to explain individual predictions by showing which words contribute to the model's output.

These explanations help interpret the classifier's behavior; they do not independently prove whether an email is malicious.

## Optional: AI Email Rewriting with Ollama

The optional rewriting feature uses the Ollama Python client and the local `qwen3:4b` model.

1. Install [Ollama](https://ollama.com/).
2. Download the model:

   ```bash
   ollama pull qwen3:4b
   ```

3. Ensure the Ollama service is running.
4. Start the Streamlit application and use the rewriting feature.

Ollama is optional for the core classification workflow, but the rewriting feature requires the local service and model to be available.

## Limitations

- Predictions depend on the dataset and patterns learned during training.
- Accuracy does not guarantee reliable detection of every phishing email.
- False positives and false negatives can occur.
- Rewriting an email may change its predicted class without proving that either version is safe.
- The optional Ollama feature requires additional setup and local computational resources.

## Future Improvements

- Compare additional classification algorithms.
- Evaluate precision, recall, and F1-score in more detail.
- Test the classifier on additional real-world email datasets.
- Improve detection of previously unseen phishing patterns.
- Explore additional feature-engineering techniques.

## Author

**Tanmay Rai**

Machine Learning Project — Email Spam & Phishing Detection
