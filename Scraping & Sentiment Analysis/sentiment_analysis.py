
import pandas as pd
import re
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

df = pd.read_csv('Sentiment_IDX.csv')
print(df)

def clean_text(text):
  if isinstance(text, float):
    text = str(text)

  text = text.lower()
  text = re.sub(r'http\S+', '', text)                   # Hapus URL
  text = re.sub(r'@\w+', '', text)                      # Hapus mention
  text = re.sub(r'#\w+', '', text)                      # Hapus hashtag
  text = re.sub(r'[^\w\s]', '', text)                   # Hapus tanda baca
  text = re.sub(r'\s+', ' ', text).strip()              # Hapus spasi berlebih
  stop_words = set(stopwords.words('indonesian'))
  words = text.split()
  words = [word for word in words if word not in stop_words and len(word) > 2]
  return ' '.join(words)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

from transformers import pipeline

sentiment_analyzer = pipeline(
    "text-classification",
    model=model_name,
    tokenizer=tokenizer,
)

def enhanced_sentiment_analysis(text):
    try:
        # Analisis with BERT
        result = sentiment_analyzer(text)[0]
        bert_label = result['label']
        bert_score = result['score']

        return bert_label, bert_score

    except Exception as e:
        print(f"Error analyzing text: {text}, Error: {e}")
        return 'neutral', 0.5

# Terapkan analisis sentimen
df['Cleaned_Title'] = df['Title'].apply(clean_text)
results = df['Cleaned_Title'].head(50).apply(enhanced_sentiment_analysis)
df[['sentiment_label', 'bert_confidence']] = pd.DataFrame(results.tolist(), index=df.head(50).index)

