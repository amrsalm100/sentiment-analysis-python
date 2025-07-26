import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')

# تحميل بيانات التغريدات من CSV
df = pd.read_csv(r'C:\Users\LENOVO\Downloads\sample_tweets.csv')  # غيّر المسار لو لزم

# تحليل المشاعر
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Tweet'].apply(get_sentiment)
print(df[['Tweet', 'Sentiment']])

# رسم النتائج
sentiment_counts = df['Sentiment'].value_counts()
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Analysis of Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.grid(True)
plt.tight_layout()
plt.show()