from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import nltk
import re
import string
from nltk.corpus import stopwords

# Initialize Flask application
app = Flask(__name__)

# Load data and model
df = pd.read_csv("C:/Users/91639/Desktop/mahi's clg/6th sem/Hate_Speech_detection/Hate_Speech_detection/twitter_data.csv")
df['labels'] = df['class'].map({0: "Hate speech detected", 1: "Offensive language detected", 2: "No hate and offensive speech"})
df = df[['tweet', 'labels']]

stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words("english"))

def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

df["tweet"] = df["tweet"].apply(clean)

x = np.array(df["tweet"])
y = np.array(df["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [clean(message)]
        vect = cv.transform(data).toarray()
        prediction = clf.predict(vect)
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
