import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
import re
import os

def load_data(filename, is_train=True):
  """Loads training or test data from a text file.

  Args:
    filename (str): Path to the text file.
    is_train (bool, optional): Whether loading training data (default: True).

  Returns:
    pd.DataFrame: DataFrame containing movie data with 'description' and 'genre' columns (training data only).
  """
  script_dir = os.getcwd()

  # Join the directory with the filename (optional, but clarifies the path)
  file_path = os.path.join(script_dir, filename)

  # Open the file using pandas
  data = pd.read_csv(file_path,sep=":::", engine="python") 
  if is_train:
    data.columns = ["ID", "TITLE", "GENRE", "DESCRIPTION"]
    return data.drop("TITLE", axis=1)  # Training data with genre labels
  else:
    data.columns = ["ID", "TITLE", "DESCRIPTION"]
    return data.drop(["ID", "TITLE"], axis=1)  # Test data without genre labels

def preprocess_text(text):
  """Preprocesses text data for machine learning.

  Args:
    text (str): Text to be preprocessed.

  Returns:
    str: Preprocessed text (lowercase, alphanumeric characters only).
  """

  text = text.lower()
  text = re.sub(r"[^a-z0-9\s]", "", text)
  return text

def train_model(train_data, test_data):
  """Trains a machine learning model for genre prediction.

  Args:
    train_data (pd.DataFrame): Training data DataFrame.
    test_data (pd.DataFrame): Test data DataFrame.

  Returns:
    tuple: Tuple containing the trained model and the vectorizer.
  """

  X_train = train_data["DESCRIPTION"].apply(preprocess_text)
  y_train = train_data["GENRE"]

  vectorizer = TfidfVectorizer(max_features=2000)
  X_train_vec = vectorizer.fit_transform(X_train)

  model = MultinomialNB()
  model.fit(X_train_vec, y_train)

  X_test = test_data["DESCRIPTION"].apply(preprocess_text)
  X_test_vec = vectorizer.transform(X_test)

  return model, vectorizer

def predict_genre(model, vectorizer, text):
  """Predicts the genre of a new movie based on its description.

  Args:
    model: Trained machine learning model.
    vectorizer: Fitted vectorizer used during training.
    text (str): Description of the movie.

  Returns:
    str: Predicted genre of the movie.
  """

  preprocessed_text = preprocess_text(text)
  text_vec = vectorizer.transform([preprocessed_text])
  predicted_genre = model.predict(text_vec)[0]
  return predicted_genre


# Replace with paths to your training and test data files
train_data_path = "train_data.txt"
test_data_path = "test_data.txt"

# Load training and test data
train_data = load_data(train_data_path)
test_data = load_data(test_data_path, is_train=False)  # Specify test data

# Train the model
model, vectorizer = train_model(train_data, test_data)

# Example usage: Predict genre for a new movie description
new_movie_description = "A group of astronauts become stranded on Mars after their mission goes wrong."
predicted_genre = predict_genre(model, vectorizer, new_movie_description)
print(f"Predicted genre for '{new_movie_description}': {predicted_genre}")
