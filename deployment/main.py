import numpy as np
import re
import pickle
import argparse

from keras import optimizers
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import pad_sequences

parser = argparse.ArgumentParser()

parser.add_argument("--vocab_path", type=str, default="../training/vocab.pkl")
parser.add_argument("--model_path", type=str, default="../training/model.h5")
parser.add_argument("--input_text", type=str, default="It is a good movie")

opt = parser.parse_args()

def load_model_and_vocab():
    vocab_path = opt.vocab_path
    model_path = opt.model_path
    model_json_path = model_path.replace('.h5','.json')
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    
    with open(model_json_path, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(model_path)
    return model, vocab

   
def clean_dataset(text):
    HTML_TAGS = re.compile('<.*?>')
    PUNCT_AND_CHARS = re.compile('[^A-Za-z ]+')
    text = re.sub(HTML_TAGS, ' ', text)
    text = re.sub(PUNCT_AND_CHARS, ' ', text)
    return text
    

def remove_stop_words(text):
    with open('./../training/stop_words.txt', 'r') as f:
        stop_words = f.read().split('\n')
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


def encode_text(text,vocab):
    encoded_text = []
    # labels = []       
    encoded_text.append([vocab[word] if word in vocab else 0 for word in text.split()])
    # labels.append(item['sentiment'])
    return np.array(encoded_text) #, np.array(labels)


def infer_the_model(text):
    
    text = clean_dataset(text)
    text = remove_stop_words(text)    
    
    model, vocab = load_model_and_vocab()
    model.summary()
    maxlen = model.layers[1].get_output_at(0).get_shape().as_list()[1]
    encoded_text = encode_text(text,vocab)
    
    encoded_text = pad_sequences(encoded_text, maxlen=maxlen)
    output = model.predict(encoded_text)
        
    print(output)
    return round(output[0][0]*100,2)
    
    
if __name__ == "__main__":
    print(infer_the_model(opt.input_text))
    
    