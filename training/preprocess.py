import pandas as pd
import re
from sklearn.model_selection import train_test_split
# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


def create_train_test_valid_dataset(df):
    """
    Takes a dataframe and splits it into train, test and validation sets
    Initially dataset was supposed to be 75k dataset with 25k test and 50k training data
    This 50k training data was supposed to be splitted into (80:20) 40k train and 10k validation data
    Since I couldnot find 75k dataset, I am using 50k dataset.
    
    I am following the same split ratio as mentioned above.
    """
    
    train,test = train_test_split(df, test_size=0.3333, random_state=42)
    train,valid = train_test_split(train, test_size=0.2, random_state=42)
    
    return train, test, valid



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
    with open('stop_words.txt', 'r') as f:
        stop_words = f.read().split('\n')
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

    # print(df.head())
    
    train, test, valid = create_train_test_valid_dataset(df)

    
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)
    valid.to_csv('valid.csv', index=False)
    
       

