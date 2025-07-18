# Resume Based Job Finder

This application uses machine learning to classify resumes into job categories and list available jobs.

## Features

- Upload PDF resumes for classification
- Paste resume text directly
- View prediction results with probability scores
- Visualize top category matches

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install streamlit PyPDF2 pandas scikit-learn nltk
```

3. Download NLTK resources (this will happen automatically when you run the app)

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

This will open the app in your default web browser.

## How to Use

1. Choose one of the input methods:
   - Upload a PDF resume file
   - Paste resume text directly

2. Click "Classify Resume" to analyze the content

3. View the results:
   - The predicted job category
   - Probability scores for top matches
   - Bar chart visualization of category probabilities

## Model Information

The classifier uses a Random Forest model trained on a dataset of labeled resumes. Text is preprocessed using NLTK for tokenization, lemmatization, and stop word removal, then vectorized using TF-IDF before classification. 