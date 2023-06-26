import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

def format_dataset(df):
    """
    Lemmatize dataset and convert to lowercase
    Change sentiment, positive->1, negaive->0
    """
    df['review'] = df['review'].apply(lambda x: x.lower())
    df['review'] = df['review'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df
    
def clean_dataset(df):
    """
    Remove html tags and punctuation from a dataframe
    """
    HTML_TAGS = re.compile('<.*?>')
    PUNCT_AND_CHARS = re.compile('[^A-Za-z ]+')
    
    df['review'] = df['review'].apply(lambda x: re.sub(HTML_TAGS, ' ', x))
    df['review'] = df['review'].apply(lambda x: re.sub(PUNCT_AND_CHARS, ' ', x))
    return df
    

def remove_stop_words(df):
    """
    Removes stopwords from a dataframe
    """
    stop_words = set(stopwords.words('english'))
    df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    return df


if __name__ == "__main__":
    df = pd.read_csv('IMDB_Dataset.csv')
    print(df.head())
    
    df = clean_dataset(df)
    print(df.head())
    
    df = format_dataset(df)
    print(df.head())
    
    df = remove_stop_words(df)
    print(df.head())
    
    df.to_csv('cleaned_dataset.csv', index=False)
       

