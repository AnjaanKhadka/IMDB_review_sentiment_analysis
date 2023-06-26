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
parser.add_argument("--top_words", type=int, default=5000)
parser.add_argument("--max_words", type=int, default=200)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.00001)

args = parser.parse_args()
opt = vars(args)
print(opt)



def create_vocab(df,top_words=5000):
    vocab = {}
    for _,item in df.iterrows():
        for word in item['review'].split():
            if word not in vocab:
                vocab[word] = 0
            else:
                vocab[word] += 1

    # sort the doctionery by value
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    new_vocab = {}
    for i,(word,_) in enumerate(vocab[:top_words-1]):
        # i+1 becasue 0 is reserved for padding
        new_vocab[word] = i+1
    
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(new_vocab, f)
        
    return new_vocab

def encode_text(df,vocab):
    encoded_text = []
    labels = []
    for _,item in df.iterrows():
        
        encoded_text.append([vocab[word] if word in vocab else 0 for word in item['review'].split()])
        labels.append([item['sentiment'],1-int(item['sentiment'])])
    return np.array(encoded_text), np.array(labels)





if __name__ == "__main__":
    top_words = opt['top_words']
    max_words = opt['max_words']
    vocab = create_vocab(pd.read_csv("cleaned_dataset.csv"),top_words=top_words)
    X_train,Y_train = encode_text(pd.read_csv("train.csv"),vocab)
    X_valid,Y_valid = encode_text(pd.read_csv("valid.csv"),vocab)


    X_train = pad_sequences(X_train, maxlen=max_words)
    X_valid = pad_sequences(X_valid, maxlen=max_words)

    optm = optimizers.Adam(learning_rate=opt['learning_rate'])
    

    model = Sequential()      

    model.add(Embedding(top_words, 64, input_length=max_words))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','Precision','Recall'])
    model.summary()

    history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=opt['epochs'], batch_size = opt['batch_size'], verbose=2)
    
    print(history)
    
    
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    













# import numpy as np
# import pandas as pd
# from datasets import load_metric, load_dataset

# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


 
# def compute_metrics(eval_pred):
#    load_accuracy = load_metric("accuracy")
#    load_f1 = load_metric("f1")
#    load_precision = load_metric("precision")
#    load_recall = load_metric("recall")
  
#    logits, labels = eval_pred
#    predictions = np.argmax(logits, axis=-1)
#    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
#    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
#    precision = load_precision.compute(predictions=predictions, references=labels)["precision"]
#    recall = load_recall.compute(predictions=predictions, references=labels)["recall"]
#    return {"accuracy": accuracy,"precision":precision, "recall":recall, "f1": f1}


# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
# model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)


# def create_train_test_valid_dataset(df):
#     """
#     Takes a dataframe and splits it into train, test and validation sets
#     Initially dataset was supposed to be 75k dataset with 25k test and 50k training data
#     This 50k training data was supposed to be splitted into (80:20) 40k train and 10k validation data
#     Since I couldnot find 75k dataset, I am using 50k dataset.
    
#     I am following the same split ratio as mentioned above.
#     """
    
#     train,test = train_test_split(df, test_size=0.3333, random_state=42)
#     train,valid = train_test_split(train, test_size=0.2, random_state=42)
    
#     return train, test, valid


# def tokenize_dataset(df):
#     data = []
#     labels = []
#     for _,item in df.iterrows():
#         data.append(tokenizer(item['review'], padding='max_length', truncation=True, max_length=200))
#         labels.append(item['sentiment'])
#     return data,labels


# def preprocess_function(examples):
#    return tokenizer(examples["review"], truncation=True)
 




# if __name__ == "__main__":

    
#     print("data_split complete!!")
#     # train_x, train_y = tokenize_dataset(train)
#     # valid_x, valid_y = tokenize_dataset(valid)
    
#     train = load_dataset('csv', data_files='train.csv')
#     valid = load_dataset('csv', data_files='valid.csv')
    
#     tokenized_train = train.map(preprocess_function, batched=True)
#     tokenized_valid = valid.map(preprocess_function, batched=True)
    
#     print("tokenization complete!!")
    
#     training_args = TrainingArguments(
#         output_dir="Anjaan-Khadka/DistilbertForSentimentAnalysis",
#         learning_rate=1e-3,
#         per_device_train_batch_size=1,
#         per_device_eval_batch_size=1,
#         num_train_epochs=2,
#         weight_decay=0.01,
#         save_strategy="epoch",
#         push_to_hub=True,
#     )
 
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_train['train'],
#         eval_dataset=tokenized_valid['train'],
#         compute_metrics=compute_metrics,
#         tokenizer=tokenizer,
#     )
    
#     trainer.train()
    
    
    
    
    
    
    
    