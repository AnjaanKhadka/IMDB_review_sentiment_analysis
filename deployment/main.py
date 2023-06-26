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

def infer_the_model(text):
    model, vocab = load_model_and_vocab()
    maxlen = model.layers[1].get_output_at(0).get_shape().as_list()[1]
    encoded_text = encode_text(text,vocab)
    
    encoded_text = pad_sequences(encoded_text, maxlen=maxlen)
    output = model.predict(encoded_text)
    return output[0][0]
    
    
if __name__ == "__main__":
    print(infer_the_model(opt.input_text))
    
    