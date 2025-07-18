import re

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

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Step 1: Load dataset
data = pd.read_csv('ResumeDataSet/UpdatedResumeDataSet.csv')
print(f"\n=== Loaded {len(data)} resumes ===")
print(f"\n=== Categories: {data['Category'].unique()} ===")


def clean_text(text):
    global stop_words, lemmatizer
    cleaned_words = []
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    words = word_tokenize(text)

    for word in words:
        if word not in stop_words and len(word) > 2:
            cleaned_words.append(lemmatizer.lemmatize(word))

    
    return ' '.join(cleaned_words)

# Step 2: Clean dataset
print("\n=== Cleaning resume text... ===")
data['cleaned_resume'] = data['Resume'].apply(clean_text)


# Step 3: Convert text labels to numbers
print("\n=== Preparing data... ===")
label_encoder = LabelEncoder()
data['category_encoded'] = label_encoder.fit_transform(data['Category'])

# for i, category in enumerate(label_encoder.classes_):
#     print(f"{i}: {category}")


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
print(f"\n=== Training Complete! ===")


# Step 8: Model Evaluation
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n=== Model Accuracy: {accuracy:.2f} ===")


# Step 9: Function to Predict Resume Category
def predict_resume_category(resume_text):
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
