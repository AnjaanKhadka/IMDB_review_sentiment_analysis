# Sentiment analysis on IMDB movie review dataset

I have trained a sequential CNN model for the task. I initially started training a transformer model (distilbert) which should have yeild better results. Byt it took very long time to train. Thus I opted for small CNN model to achieve. This has resulted in over-fit model.

## Dataset

I used a 50k IMDB movie review dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

This dataset was split into 3ways: Train, Test, and Validation.

![image](https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis/assets/43941329/f64e1bd0-4a3a-41fb-9b86-0f073281784a)

## Training

This model is trained on a simple sequential CNN architecture

![image](https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis/assets/43941329/d373d54f-3874-477d-97e3-c5084724dc69)

Dataset vocabulary size: 50_000

Hyperparameters used for training:

1. Epochs: 50
2. Batch size: 32
3. Adam optimizer
4. Fixed learning rate of 1e-6

## Results

### Training Results

![image](https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis/assets/43941329/e577cb0f-99e0-4c0b-af26-40ed8e6c4319)
![image](https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis/assets/43941329/bf39fc0b-d085-4de3-b52a-14787bec4afc)
![image](https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis/assets/43941329/96454898-67d7-4670-bfa4-93fbdf8a312a)
![image](https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis/assets/43941329/882a32cd-d1df-4735-a6dc-a3ad7e3d1a2f)

### Testing Results

Accuracy: 77.23%

Precision: 71.07%

Recall: 76.34%

f1-score: 73.61%

## Conclusions

1. This is a very small model thus is heavily overfit on training dataset.

2. There are many things that can be done to improve on this model, including increasing model width and depth, increasing dataset. Incorporating multiple sentiments (like, dislike, neutral, confused, ...)

3. This model cannot be used to handle ambiguios reviews. It also fails to properly understand the context of negation of the sentences or certain phrases.

4. Better models can be used for better understanding of the context. These models include transformer model or even other sequence2sequence models. This allows model to be trained to first understand meaning of various words, then that language model can be repurposed and finetuned for sentiment analysis.

## Deployment

I have implemented a simple web UI to access this model using Streamlit.

Output Screenshots:

![image](https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis/assets/43941329/ccee1292-a35d-40ee-956d-4212bf467a74)
![image](https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis/assets/43941329/1cced2db-35ac-4d3f-9880-9136414827cf)
![image](https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis/assets/43941329/a32862bd-2a71-4eb8-a317-a47248168b7c)
![image](https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis/assets/43941329/4879857c-bd6c-446b-bc57-725206ca81ab)
![image](https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis/assets/43941329/33aa563d-f986-4254-a618-dc00891e2ee2)

## If You want to try Training

Clone my repository as

    git clone https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis.git

Install requirements as

    pip install -r requirements_train.txt

Then start by downloading dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Then by executing preprocess.py as

    python preprocess.py

Then we obtain train, test and validation csv files.

These files are used for the training and testing of the model.

To train you can execute trainer.py

    python trainer.py

You can use various arguments for better training. These arguments are:

1. --cnn_blocks: This is an integer input to define number of CNN blocks in the system.

2. --cnn_depth: This is an integer input to define number of CNN layers per CNN block.

3. --top_words: Selecting most frequent n words in our vocabulary. This defines how rare words our model accepts. Low value of top_words will result in common words missing. Higher value results in consideration of unlikely words and makes harder for the model to see pattern within the textual data. I found optimum value to be around 40k for the dataset.

4. --max_length: It defines maximum number of words allowed per review that our model can accept. It defines the model input size. Smaller value limits the review length. Larger value increases model size and may account for longer reviews and longer training time. Larger value may also result in sparcity of input data for smaller sentences. I found optimum value to be around 200-600. max_length must be divisible by 2<sup>cnn_blocks</sup>

5. --epochs: No of epochs to train train.

6. --batch_size: Batch size defines how many sample texts are cycles in each operation. use batch_size=1 for cpu.

7. --learning_rate: It defines how quickly we proceed towards minimum state. small learning rate are prefered. larger learning rate may not converge to global minimum.

8. --model_name: This defins the model name. At last model is saved as model_name.json and model_name.h5 file, and if you want the history dictionery, model_name.history is also saved

Various other hyperparameters can be tuned by manually editing the code.

## If You want to try Testing

Clone my repository as

    git clone https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis.git

Install requirements as

    pip install -r requirements_test.txt

Then change to deployment directory and execute interface.py with streamlit as:

    streamlit run interface.py
