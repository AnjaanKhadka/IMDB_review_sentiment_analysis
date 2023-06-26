import numpy as np
import pandas as pd
import pickle
import argparse

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import pad_sequences

parser = argparse.ArgumentParser()

parser.add_argument("--vocab_path", type=str, default="../training/vocab.pkl")
parser.add_argument("--model_path", type=str, default="../training/model.pkl")
parser.add_argument("--input_text", type=str, default="It is a good movie")

opt = parser.parse_args()

def load_model_and_vocab():
    vocab_path = opt.vocab_path
    model_path = opt.model_path
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model, vocab

def encode_text(text,vocab):
    encoded_text = []
    labels = []       
    encoded_text.append([vocab[word] if word in vocab else 0 for word in text.split()])
    # labels.append(item['sentiment'])
    return np.array(encoded_text) #, np.array(labels)

if __name__ == "__main__":
    
    model, vocab = load_model_and_vocab()
    
    encoded_text = encode_text(opt['input_text'],vocab)
    
    print(model.predict(encoded_text))
    
    
    
    
    
    