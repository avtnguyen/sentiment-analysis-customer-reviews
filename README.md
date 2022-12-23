# Sentiment Analysis on Customer Reviews from Yelp

#### -- Project Status: [Completed]

## Project Motivations:
The restaurant industry is a competitive market, and it is important for businesses to understand and meet the needs and expectations of their customers. 
One way to do this is by analyzing customer sentiment, which can provide valuable insights into their experiences and preferences. 
This project aims to use natural language processing (NLP) techniques and the NLTK library to conduct sentiment analysis on reviews of various restaurants, with the goal of identifying trends and areas for improvement.

By analyzing customer sentiment, restaurants can better understand what their customers like and dislike about their products and services, and use this information to make informed decisions about how to improve their business. 
In addition, by identifying trends across different restaurants, this project can provide a broader understanding of the current state of the industry and how it is perceived by customers.

Overall, the goal of this project is to use NLP and sentiment analysis to gain a deeper understanding of customer experiences in the restaurant industry, 
and to provide actionable insights that can help businesses improve and succeed in this competitive market.


## Project Description

This project aims to use natural language processing (NLP) techniques and machine learning to conduct sentiment analysis on reviews of restaurants in the Yelp database. The project includes the following tasks:

* Data exploration: Using the BigQuery API in a Jupyter Notebook provided in the Vertex AI workbench, the project will explore multiple datasets to understand the overall businesses in the Yelp database.

* Unsupervised classification: An unsupervised model will be built to classify businesses based on their keywords descriptions.

* Text preprocessing and machine learning model building: The project will preprocess text data and build machine learning models using Gaussian Naive Bayes, Support Vector Machines, and XGBoost to predict the sentiment of reviews.

* Sentiment analysis: The project will conduct sentiment analysis on the most popular businesses and pizza restaurants in Philadelphia, with a focus on cheese steak as a key topic.

* Insights and recommendations: The project will draw insights from the analysis to determine the aspects that lead to good or bad reviews from customers in the restaurant industry, particularly for pizza restaurants. 
These insights and recommendations can be used by restaurants to improve their products and services and better meet the needs and expectations of their customers.

**About the dataset** The dataset was provided by Yelp with the following specifications:
The Yelp dataset is a subset of businesses, reviews, and user data for use in personal, educational, and academic purposes. 
Available as JSON files, use it to teach students about databases, to learn NLP, or for sample production data while you learn how to make mobile apps.

* 6,990,280 reviews 
* 150,346 businesses
* 11 metropolitans areas in USA
908,915 tips by 1,987,897 users
Over 1.2 million business attributes like hours, parking, availability, and ambience
Aggregated check-ins over time for each of the 131,930 businesses
[Source](https://www.yelp.com/dataset)

### Project Pipeline :
**1. Data processing and exploration: In this stage, the project will use the BigQuery API in Python with Jupyter Notebook in the Vertex AI workbench to perform queries on the large (~5 GB) Yelp datasets. 
Specifically, the business and reviews datasets will be used to draw insights on businesses with reviews on Yelp and answer the following questions:

* Which ten cities have the most opened businesses on Yelp?
* What are the businesses with the highest number of reviews in Philadelphia?
* What businesses are located in Philadelphia?
* Can any insights be drawn from the open and closed businesses?
* What types of businesses have 5-star reviews on Yelp?
* What keywords can be used to distinguish between good and bad reviews for the top businesses in Philadelphia?
* How do the reviews for pizza restaurants compare?

Data exploration code and analysis can be found [here](https://github.com/avtnguyen/sentiment-analysis-customer-reviews/blob/main/Yelp_Data_Exploration.html)

**2. Unsupervised classification** An unsupervised model will be built to classify businesses based on their keywords descriptions 
In this part of the project, an unsupervised machine learning model will be built to classify businesses based on their keywords descriptions. 
The model will use the k-means clustering algorithm and the Natural Language Toolkit (NLTK) library for text processing.

K-means clustering is a popular unsupervised learning method that groups similar data points together into clusters. 
It works by initially selecting a predetermined number of "centroids" or starting points, and then iteratively assigning data points to the cluster with the closest centroid. 
The centroids are then updated to the mean of the data points in the cluster, and the process is repeated until convergence. Hyperparameters tuning will also be performed to obtain the optimum solutions

NLTK is a widely-used library for natural language processing tasks such as tokenization, stemming, and lemmatization. 
It will be used to preprocess the text data in the keywords descriptions, such as by removing stop words and punctuation, and converting the words to their base form (e.g., running -> run). 
This will allow the model to focus on the meaningful content of the text and improve the accuracy of the classification.

Overall, the goal of the unsupervised model is to classify businesses into different groups based on their keywords descriptions, 
which can provide valuable insights into the types of businesses that exist in the Yelp database and how they are related to each other.

**3. Machine learning model for sentiment analysis** Text preprocessing and machine learning model building: The project will preprocess text data and build machine learning models using Gaussian Naive Bayes, Support Vector Machines, and XGBoost to predict the sentiment of reviews.
Here, I will use the Natural Language Toolkit (NLTK) library to perform text preprocessing, which will include the following steps:

* Tokenization: This process involves splitting the text into smaller units called tokens, which can be words, phrases, or symbols. 
Tokenization helps to break down the text into its individual components, which can be useful for further analysis.

* Stopwords removal: Stopwords are common words that carry little meaning and are often removed from text data to reduce noise and improve the accuracy of machine learning models. 
Examples of stopwords include "a," "an," and "the."

* Punctuation removal: Removing punctuation from the text data can help to reduce noise and simplify the data for machine learning models.

* Lemmatization: This process involves reducing words to their base form (e.g., running -> run) to help the model focus on the meaning of the text rather than the specific word forms.

After text preprocessing, I will build machine learning models using Gaussian Naive Bayes, Support Vector Machines, and XGBoost to predict the sentiment of the reviews. 

Gaussian Naive Bayes is a probabilistic classifier that makes assumptions about the independence of features and uses Bayes' theorem to calculate the probability of a data point belonging to a certain class. 

Support Vector Machines are linear models that find the hyperplane in a high-dimensional space that maximally separates different classes. 

XGBoost is a gradient boosting algorithm that creates an ensemble of decision trees and uses them to make predictions.

Overall, the goal of this phase of the project is to preprocess the text data and build machine learning models that can accurately predict the sentiment of the reviews, 
which can provide valuable insights into customer experiences and preferences in the restaurant industry.

**4. Sentiment analysis**: Sentiment analysis is the process of using natural language processing techniques to automatically identify and extract subjective information from text data. 
In this project, sentiment analysis will be conducted on the most popular businesses and pizza restaurants in Philadelphia.

The project will use monograms and bigrams analysis to identify unique keywords for bad and good reviews. 
Monograms are single words, while bigrams are pairs of words that occur together. 
By analyzing the frequency of these keywords in the reviews, the project can identify trends and patterns that can help to understand the sentiment of the reviews.

Data visualization with word clouds can also be used to gain insights from the sentiment analysis. 
A word cloud is a visual representation of the most frequently occurring words in a dataset, where the size of the word reflects its frequency. 
By creating word clouds for the good and bad reviews, the project can quickly identify the most commonly mentioned words and themes, which can provide valuable insights into the sentiment of the reviews.

Overall, the goal of the sentiment analysis is to understand the sentiment of the reviews of the most popular businesses and pizza restaurants in Philadelphia.
The results of the analysis can be used to gain insights into customer experiences and identify areas for improvement in the restaurant industry.

### Evaluation metrics
For imbalance dataset and classification model, the following metrics are used to evaluate the model performance:
* Precision
* Recall
* f1 score

### Results:
- Both the precision, recall and f1 scores are low in all models. This could be due to the small imbalance dataset that we have.
- Given the dataset without any resampling technique, XGBoost outperformed other algorithms.
- When data augmentation technique is implemented, the performance of Random Forrest model is improved significantly using SMOTE+TOMEK technique as shown in table below.
- More data is needed to improve the model accuracy for oil spill detection

<img src="https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/f1_scores.png" width="700" align = "center">

| model       | resample                     | precision  | recall | f1   |
| ------------|:----------------------------:| ----------:|-------:|-----:|
| DNN         | SMOTE                        |   0.267    |0.8     |0.4   |
| DNN         | SMOTE+TOMEK                  |   0.385    |0.5     |0.435 |
| RF          | SMOTE+TOMEK                  |  0.625     |0.5     |0.555 |
| RF          | SMOTE+ENN                    |   0.461    |0.6     |0.522 |
| XGBoost     | ADASYN                       |   0.5      |0.5     |0.5   |
| XGBoost     | SMOTE                        |   0.454    |0.5     |0.476 |
| DNN         | No resample                  |   0.114    |0.9     |0.2   |
| RF          | No resample                  |   0.2      |0.1     |0.133 |
| XGBoost     | No resample                  |   0.357    |0.5     |0.417 |


