import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import movie_reviews

# step-1 setup - Install necessary libraries (if not installed)

#Step 2: Data Preparation

nltk.download("movie_review")

#load the dataset
