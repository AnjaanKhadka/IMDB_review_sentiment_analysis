# Sentiment analysis on IMDB movie review dataset

I have trained a sequential CNN model for the task.

## Dataset

I used 50k IMDB movie review dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

This dataset was splitted in 3ways. Train, Test, Validation.



## Training

This model is trained on a simple sequential CNN architecture



Dataset vocabulary size: 50_000

Hyperparameters used for training:

1. Epochs: 50
2. Batch size: 32
3. Adam optimizer
4. Fixed learning rate of 1e-6

Results:




## Deployment

I have implemented a simple webui to access this model using streamlit.

Output Screenshots:



## If You want to try Training

Clone my repository as

    git clone https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis.git

Install requiremensts as

    pip install -r requirements_train.txt


### Testing-requirements

    pip install -r requirements_test.txt

