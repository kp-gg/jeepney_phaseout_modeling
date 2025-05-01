import pandas as pd
from googletrans import Translator
import time
from scipy import stats
from scipy.stats import ttest_ind
from scipy.sparse import hstack
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#KATHERINE COMMENT: Nice use of translation processing where both language and emotion need analysis, it shows thoughtful handling of the multilingual and nuanced text inputs.
analyzer = SentimentIntensityAnalyzer()
translator = Translator()

# Read your CSV (make sure it has sentiment scores, not just 'positive'/'negative' words)
df_filtered = pd.read_csv('/Users/gpasion/Documents/INST414/sentiments.csv')

# 👇 Instead of mapping "positive"/"negative", use the raw sentiment scores
# Assume your CSV has a column called 'sentiment_score' (the numbers you showed)

def binary_sentiment(score):
    if score > 0:
        return 1
    else:
        return 0

#KATHERINE COMMENT: I appreciate the clear binary sentiment mapping, it aligns well with logistic regression's binary classification that you use later on.
# Apply binary mapping
df_filtered['sentiment_binary'] = df_filtered['sentiment_score'].apply(binary_sentiment)

# (Optional) Filter out NaNs if any
df_model = df_filtered.dropna(subset=['sentiment_binary']).copy()

# Text cleaning
def clean_text(text):
    text = re.sub(r'http\S+', '', text)           # remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)         # remove mentions/hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)       # remove punctuation
    return text.lower().strip()

df_model['clean_text'] = df_model['translated'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_text = vectorizer.fit_transform(df_model['clean_text'])

# Numeric features
X_numeric = df_model[['likeCount', 'phaseout']].copy()
X_numeric['phaseout'] = X_numeric['phaseout'].astype(int)

# Combine features
X = hstack([X_text, X_numeric])
y = df_model['sentiment_binary']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Model
#KATHERINE COMMENT: Logistic Regression is a great way to handle the text data you are using, it's a great interpretable model choice.
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Check your final labels
#print(df_model['sentiment_binary'])


# Get the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display it nicely
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')  # 'Blues' colormap makes it look clean

plt.title('Confusion Matrix')
plt.show()


# two sample t test
#KATHERINE COMMENT: Interesting inclusion of statistical testing, the t-test adds a layer of analytical depth that complements the ML model.
from scipy.stats import ttest_ind

# Assume you have two groups
group_phaseout = df_model[df_model['phaseout'] == 1]['sentiment_score']
group_non_phaseout = df_model[df_model['phaseout'] == 0]['sentiment_score']

t_stat, p_val = ttest_ind(group_phaseout, group_non_phaseout)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")
