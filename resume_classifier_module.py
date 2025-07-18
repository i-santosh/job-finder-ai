import re
import os
import pickle
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources if not already downloaded
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

download_nltk_resources()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define file paths for saved models
MODEL_PATH = 'models'
CLASSIFIER_FILE = os.path.join(MODEL_PATH, 'resume_classifier.pkl')
VECTORIZER_FILE = os.path.join(MODEL_PATH, 'tfidf_vectorizer.pkl')
ENCODER_FILE = os.path.join(MODEL_PATH, 'label_encoder.pkl')

def clean_text(text):
    cleaned_words = []
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    words = word_tokenize(text)

    for word in words:
        if word not in stop_words and len(word) > 2:
            cleaned_words.append(lemmatizer.lemmatize(word))
    
    return ' '.join(cleaned_words)

def train_model():
    """Train the model and save it to disk"""
    print("Loading and preparing dataset...")
    # Step 1: Load dataset
    data = pd.read_csv('ResumeDataSet/UpdatedResumeDataSet.csv')
    print(f"Loaded {len(data)} resumes")
    
    # Step 2: Clean dataset
    print("Cleaning resume text...")
    data['cleaned_resume'] = data['Resume'].apply(clean_text)
    
    # Step 3: Convert text labels to numbers
    print("Encoding categories...")
    label_encoder = LabelEncoder()
    data['category_encoded'] = label_encoder.fit_transform(data['Category'])
    
    # Step 4: Train model
    print("Training the model...")
    X_text = data['cleaned_resume']
    y = data['category_encoded']
    
    # Step 5: Convert text to numbers using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(X_text)
    
    # Step 6: Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Step 7: Model Training
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    print("Training Complete!")
    
    # Step 8: Model Evaluation
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Save the trained model, vectorizer, and encoder
    with open(CLASSIFIER_FILE, 'wb') as f:
        pickle.dump(classifier, f)
    
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(ENCODER_FILE, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Model saved to {MODEL_PATH} directory")
    
    return classifier, vectorizer, label_encoder

def load_model():
    """Load the trained model from disk"""
    # Check if model files exist
    if not (os.path.exists(CLASSIFIER_FILE) and 
            os.path.exists(VECTORIZER_FILE) and 
            os.path.exists(ENCODER_FILE)):
        print("Trained model not found. Training new model...")
        return train_model()
    
    # Load the trained model, vectorizer, and encoder
    try:
        with open(CLASSIFIER_FILE, 'rb') as f:
            classifier = pickle.load(f)
        
        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open(ENCODER_FILE, 'rb') as f:
            label_encoder = pickle.load(f)
        
        return classifier, vectorizer, label_encoder
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training new model...")
        return train_model()

# Global variables for model, vectorizer, and encoder
classifier = None
vectorizer = None
label_encoder = None

# Load the model when the module is imported
classifier, vectorizer, label_encoder = load_model()

def predict_resume_category(resume_text):
    """Predict the category of a resume"""
    global classifier, vectorizer, label_encoder
    
    # Ensure model is loaded
    if classifier is None or vectorizer is None or label_encoder is None:
        classifier, vectorizer, label_encoder = load_model()
    
    cleaned_text = clean_text(resume_text)
    text_vector = vectorizer.transform([cleaned_text])
    
    prediction = classifier.predict(text_vector)[0]
    category = label_encoder.inverse_transform([prediction])[0]
    confidence = classifier.predict_proba(text_vector)[0][prediction]

    return category, confidence

def run_interactive_prediction():
    print("\n=== Try Your Own Resume ===")
    print("Enter a resume text (or 'quit' to exit):")
    
    while True:
        user_input = input("\nResume text: ")
        if user_input.lower() == 'quit':
            break
        
        if user_input.strip():
            try:
                category, confidence = predict_resume_category(user_input)
                print(f"Predicted Category: {category} with confidence: {confidence:.2f}")
            except ValueError as e:
                print(f"Error: {e}")
                break
        else:
            print("Please enter some text!")

if __name__ == "__main__":
    # Example prediction
    print("\n=== Example Prediction ===")
    sample_resume = """
            Software Engineer with 3 years of experience in Python, Django, and React.
            Skilled in machine learning, data analysis, and web development.
            Experience with SQL databases and cloud computing.
            """
    category, confidence = predict_resume_category(sample_resume)
    print(f"\n=== Predicted Category: {category} with confidence: {confidence:.2f} ===")

    run_interactive_prediction()
