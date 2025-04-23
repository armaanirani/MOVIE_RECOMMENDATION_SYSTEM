import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('preprocess.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logging.info("Starting preprocessing...")

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Cleaning the text
stop_words = set(stopwords.words('english'))

# Load the dataset
try:
    df = pd.read_csv('movies_dataset.csv')
    logging.info("Dataset loaded successfully.")
except Exception as e:
    logging.error("Dataset file not found. Please check the file path.")
    raise e

def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Filter the required columns for recommendation
required_columns = ['title', 'overview', 'genres', 'keywords', 'cast', 'director']

df = df[required_columns]

df = df.dropna().reset_index(drop=True)

df['combined_features'] = df['overview'] + ' ' + df['genres'] + ' ' + df['keywords'] + ' ' + df['cast'] + ' ' + df['director']

logging.info("Cleaning text...")
df['cleaned_text'] = df['combined_features'].apply(preprocess_text)
logging.info("Text cleaned successfully.")

# Vectorize the combined features using TF-IDF
logging.info("Vectorizing combined features...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])
logging.info("Combined features vectorized successfully.")

# Computing cosine similarity matrix
logging.info("Computing cosine similarity matrix...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("Cosine similarity matrix computed successfully.")

# Save the cosine similarity matrix and other necessary objects for later use
joblib.dump(df, 'df_cleaned.pkl')
joblib.dump(cosine_sim, 'cosine_sim.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
logging.info("Objects saved successfully.")

logging.info("Preprocessing completed successfully.")