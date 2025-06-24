# Restaurant Review Sentiment Analysis

This project performs binary sentiment analysis on restaurant reviews using Natural Language Processing (NLP) in R. It uses TF-IDF vectorization and logistic regression to classify reviews as positive or negative.

## 📂 Dataset

The dataset contains:
- `Review`: Text content of a customer review
- `Liked`: Binary value (1 = positive sentiment, 0 = negative)

## 🔧 Tools & Packages

- `dplyr` for data wrangling
- `tm` for text mining
- `caret` for evaluation
- `ggplot2` for visualizations
- `wordcloud2` for word cloud visualization

## 🧠 Model

- Preprocessing: Text is cleaned (lowercased, punctuation removed, stopwords filtered)
- Features: Term Frequency-Inverse Document Frequency (TF-IDF)
- Model: Logistic Regression
- Evaluation: Confusion Matrix

## 📊 Output

- Accuracy and confusion matrix of predictions
- Bar plot of most frequent terms
- Word cloud of commonly used words
