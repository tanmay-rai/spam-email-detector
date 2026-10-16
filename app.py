
from pathlib import Path
import ollama
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import shap



# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.parquet"
MODEL_PATH = BASE_DIR / "spam_pipeline.joblib"


# Page configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📩",
    layout="wide"
)


# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Load the training dataset
@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


# Calculate word counts for email-length analysis
@st.cache_data
def prepare_email_lengths():
    df = load_data()
    df["word_count"] = df["text"].str.findall(r"\b\w+\b").str.len()
    return df[["label", "word_count"]]


# Load the SHAP explainer
@st.cache_resource
def load_explainer():
    model = load_model()
    masker = shap.maskers.Text()
    return shap.Explainer(model.predict_proba, masker)


# Page 1: Dataset Analysis
def dataset_page():
    st.title("📊 Dataset Analysis")
    st.write(
        "Explore class balance, email lengths, and basic "
        "data-quality checks in the training dataset."
    )

    df = load_data()

    # Display dataset statistics
    total_emails = len(df)
    benign_count = int((df["label"] == 0).sum())
    spam_count = int((df["label"] == 1).sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Emails", f"{total_emails:,}")
    col2.metric("Legitimate Emails", f"{benign_count:,}")
    col3.metric("Spam / Phishing Emails", f"{spam_count:,}")

    st.divider()

    # 1. Class distribution
    st.subheader("1. Email Class Distribution")

    distribution = df["label"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(
        ["Legitimate (0)", "Spam / Phishing (1)"],
        distribution.values
    )

    ax.set_ylabel("Number of emails")
    ax.set_title("Distribution of Training Labels")

    ax.bar_label(bars, fmt="%.0f")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Label 0 represents benign emails; label 1 represents "
        "phishing/spam, according to this dataset's labels."
    )

    st.divider()

    # 2. Email length distribution
    st.subheader("2. Email Length Distribution")
    st.write(
        "Compare email lengths by the number of words in each message. "
        "The x-axis is capped at the 99th percentile so a few extremely "
        "long emails do not make the main distribution hard to read."
    )

    lengths_df = prepare_email_lengths()
    length_cap = max(1, int(lengths_df["word_count"].quantile(0.99)))
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, name, color in [
        (0, "Legitimate", "#2E86C1"),
        (1, "Spam / phishing", "#D9534F"),
    ]:
        values = lengths_df.loc[
            lengths_df["label"] == label, "word_count"
        ].clip(upper=length_cap)
        ax.hist(values, bins=50, alpha=0.55, label=name, color=color)
    ax.set_xlabel(f"Words per email (values above {length_cap} capped)")
    ax.set_ylabel("Number of emails")
    ax.set_title("Email Length Distribution by Class")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # 3. Basic dataset quality checks
    st.subheader("3. Dataset Quality")
    raw_df = pd.read_parquet(DATA_PATH)
    missing_text = int(raw_df["text"].isna().sum())
    missing_label = int(raw_df["label"].isna().sum())
    empty_text = int(
        raw_df["text"].fillna("").astype(str).str.strip().eq("").sum()
    )
    duplicate_text = int(
        raw_df["text"].dropna().astype(str).duplicated().sum()
    )

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Missing text", f"{missing_text:,}")
    q2.metric("Missing labels", f"{missing_label:,}")
    q3.metric("Empty messages", f"{empty_text:,}")
    q4.metric("Repeated text rows", f"{duplicate_text:,}")

    st.caption(
        "These are checks on the raw training file. Repeated text rows are "
        "flagged for inspection; they are not automatically removed."
    )


# Apply a configurable decision threshold to the spam probability
THRESHOLD = 0.55

def classify_email(message, threshold=THRESHOLD):
    model = load_model()
    probabilities = model.predict_proba([message])[0]
    classes = list(model.classes_)

    benign_index = classes.index(0)
    spam_index = classes.index(1)
    spam_probability = probabilities[spam_index]

    prediction = 1 if spam_probability >= threshold else 0
    return prediction, probabilities, benign_index, spam_index


# Page 2: Email Detector
def detector_page():
    st.title("📩 Email Spam Detector")
    st.write(
        "Enter an email to classify it and explore the words "
        "that influenced the prediction."
    )

    model = load_model()

    message = st.text_area(
        "Enter your email",
        placeholder=(
            "Example: Congratulations! You won a free prize..."
        ),
        height=180
    )
    if st.button("✨ Rewrite & Check Again"):
        if not message.strip():
            st.warning("Please enter an email first.")
        else:
            try:
                with st.spinner("AI is rewriting the email..."):
                    rewritten_email = convert_email(message)

                st.subheader("AI-Rewritten Email")
                st.write(rewritten_email)

                new_prediction, new_probabilities, benign_index, spam_index = (
                    classify_email(rewritten_email)
                )

                st.subheader("Prediction After Rewriting")

                if new_prediction == 1:
                    st.error("⚠️ Still classified as SPAM / PHISHING.")
                else:
                    st.success("✅ Classified as LEGITIMATE.")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Legitimate Probability",
                    f"{new_probabilities[benign_index] * 100:.2f}%"
                )

                col2.metric(
                    "Spam Probability",
                    f"{new_probabilities[spam_index] * 100:.2f}%"
                )

            except Exception as error:
                st.error(f"Could not process the email: {error}")

    if st.button("Check Email", type="primary"):
        if not message.strip():
            st.warning("Please enter an email first.")
            return

        # Make the prediction using the 30% spam threshold
        prediction, probabilities, benign_index, spam_index = (
            classify_email(message)
        )

        st.subheader("Prediction")

        if prediction == 1:
            st.error("⚠️ This email is classified as SPAM / PHISHING.")
        else:
            st.success("✅ This email is classified as LEGITIMATE.")

        # Display prediction probabilities
        col1, col2 = st.columns(2)

        col1.metric(
            "Legitimate Probability",
            f"{probabilities[benign_index] * 100:.2f}%"
        )

        col2.metric(
            "Spam Probability",
            f"{probabilities[spam_index] * 100:.2f}%"
        )

        st.caption(
            "These are model-estimated probabilities, not guarantees."
        )

        st.divider()

        # Explain the prediction using SHAP
        st.subheader("Why did the model make this prediction?")

        try:
            with st.spinner("Analyzing the email's words..."):
                explainer = load_explainer()
                explanation = explainer([message])

                values = np.asarray(explanation.values)
                data = explanation.data

                # Multi-class output: (emails, tokens, classes)
                if values.ndim == 3:
                    contributions = values[0, :, spam_index]
                    words = np.asarray(data[0]).astype(str)

                # Single-output fallback
                elif values.ndim == 2:
                    contributions = values[0]
                    words = np.asarray(data[0]).astype(str)

                else:
                    raise ValueError(
                        f"Unexpected SHAP output shape: {values.shape}"
                    )

                if len(words) != len(contributions):
                    raise ValueError(
                        "SHAP token and contribution counts do not match."
                    )

                valid = np.array([
                    bool(word.strip()) for word in words
                ])

                words = words[valid]
                contributions = contributions[valid]

                if len(words) == 0:
                    st.info("No words were available to explain.")

                else:
                    # Select the most influential words
                    top_indices = np.argsort(
                        np.abs(contributions)
                    )[-15:]

                    top_words = words[top_indices]
                    top_values = contributions[top_indices]

                    order = np.argsort(top_values)
                    top_words = top_words[order]
                    top_values = top_values[order]

                    fig, ax = plt.subplots(figsize=(10, 7))

                    colors = [
                        "#E45756" if value > 0 else "#4C78A8"
                        for value in top_values
                    ]

                    ax.barh(top_words, top_values, color=colors, height=0.7)
                    ax.axvline(0, color="black", linewidth=0.8)

                    ax.set_xlabel("SHAP contribution to spam output")
                    ax.set_ylabel("Words")
                    ax.set_title("Words Influencing the Spam Prediction")

                    ax.tick_params(axis="y", labelsize=10)
                    ax.grid(axis="x", linestyle="--", alpha=0.25)
                    ax.set_axisbelow(True)

                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.caption(
                        "Red bars push the model's spam output upward; "
                        "blue bars push it downward. These are model "
                        "contributions, not proof that a word is malicious."
                    )

        except Exception as error:
            st.error(
                f"Could not generate the SHAP explanation: {error}"
            )


def convert_email(message):
    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the supplied email in clear, neutral, "
                    "professional language. "
                    "Preserve the original meaning and all factual details. "
                    "Do not invent offers, deadlines, discounts, links, "
                    "or other claims. "
                    "Remove manipulative urgency, misleading claims, "
                    "and deceptive requests. "
                    "Do not turn the email into a marketing message. "
                    "If the original email is promotional, keep it "
                    "transparently promotional rather than disguising it "
                    "as a personal or legitimate message. "
                    "Return only the rewritten email."
                )
            },
            {
                "role": "user",
                "content": message
            }
        ]
    )

    return response["message"]["content"]


# Check required files
if not MODEL_PATH.exists():
    st.error("Model not found. Run main.py first to train it.")
    st.stop()

if not DATA_PATH.exists():
    st.error("Training dataset not found at data/train.parquet.")
    st.stop()


# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Dataset Analysis", "Email Detector"]
)

if page == "Dataset Analysis":
    dataset_page()
else:
    detector_page()
