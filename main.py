
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.parquet"
TEST_PATH = BASE_DIR / "data" / "test.parquet"
MODEL_PATH = BASE_DIR / "spam_pipeline.joblib"


def load_data():
    """Loading the Training Set

    Returns:
        _type_: DataFrame Object
    """
    df = pd.read_parquet(DATA_PATH)

    df = df.dropna(subset=["text", "label"])

    # Ensure the email text is stored as strings
    df["text"] = df["text"].astype(str)

    # Printing/Showing Basic info about the Training Set
    print("Dataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    return df


def prepare_data(df):
    """Split the Data using SciKit-Learn

    Args:
        df (DataFrame Object): Takes the DataFrame Object for training

    Returns:
        Pandas Series: Returns the traning and testing series
    """
    X = df["text"]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining emails:", len(X_train))
    print("Validation emails:", len(X_val))

    return X_train, X_val, y_train, y_val


def train_model(X_train, y_train):
    """Train the Model based on the training data received

    Args:
        X_train (Pandas Series): Training Messages
        y_train (Pandas Series): Tells wheter message is spam or not (0's,1's)

    Returns:
        Pipleline Object: Returns the trained model
    """
    
    #Pipeline Object is created that has tfidf and classifier
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ])

    #Model is trained
    model.fit(X_train, y_train)

    print("\nModel training completed!")

    return model


def evaluate_model(model, X_val, y_val):
    """Evaluates the model accuracy over the testing data

    Args:
        model (Pipeline Object):  The Trained Model
        X_val (Pandas Series:  Testing Messages (The remaining 20%)
        y_val (Pandas Series):  Testing label (The remaining 20%)
    """
    
    #Predicts the output for the 20% of data
    y_pred = model.predict(X_val)

    print("\nAccuracy:", accuracy_score(y_val, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Phishing/Spam"],
        zero_division=0
    ))

  


def test_model(model):
    """Tests the model over other traning data

    Args:
        model (Pipeline Object): Accepts the trained model
    """
    
    df_test = pd.read_parquet(TEST_PATH)

    # Removes the empy fields
    df_test = df_test.dropna(subset=["text", "label"])

    # Ensure the email text is stored as strings
    df_test["text"] = df_test["text"].astype(str)

    # Printing/Showing Basic info about the Testing Set
    print("\n\nTesting Dataset shape:", df_test.shape)
    print("\nTesting Label distribution:")
    print(df_test["label"].value_counts())

    X_test = df_test["text"]
    y_test = df_test["label"]

    print("\nTesting emails:", len(X_test))

    # Evaluate the model on the Testing Set
    y_pred = model.predict(X_test)

    print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))

    print("\nFinal Test Classification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Phishing/Spam"],
        zero_division=0
    ))



def save_model(model):
    """Saves the Model in file

    Args:
        model (Pipeline Object): Requires the trained model
    """
    joblib.dump(model, MODEL_PATH)
    print("\nModel saved to:", MODEL_PATH)


# 7. Run the complete workflow
def main():
    df = load_data()

    X_train, X_val, y_train, y_val = prepare_data(df)

    model = train_model(X_train, y_train)

    # Evaluate the model using the validation set
    evaluate_model(model, X_val, y_val)

    # Test the model using test.parquet
    test_model(model)

    save_model(model)


if __name__ == "__main__":
    main()
