# Sentiment analysis on IMDB movie review dataset

I have trained a sequential CNN model for the task.

## Dataset

I used a 50k IMDB movie review dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

This dataset was split into 3ways: Train, Test, and Validation.

![image](https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis/assets/43941329/13ea3184-450d-47d9-91b3-7509070e0d1d)

## Training

This model is trained on a simple sequential CNN architecture

![image](https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis/assets/43941329/d373d54f-3874-477d-97e3-c5084724dc69)

Dataset vocabulary size: 50_000

Hyperparameters used for training:

1. Epochs: 50
2. Batch size: 32
3. Adam optimizer
4. Fixed learning rate of 1e-6

Results:


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

## If You want to try Testing

Clone my repository as

    git clone https://github.com/AnjaanKhadka/IMDB_review_sentiment_analysis.git

Install requirements as

    pip install -r requirements_test.txt



