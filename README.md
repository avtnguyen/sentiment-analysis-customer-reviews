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
The Yelp dataset is a subset of our businesses, reviews, and user data for use in personal, educational, and academic purposes. 
Available as JSON files, use it to teach students about databases, to learn NLP, or for sample production data while you learn how to make mobile apps.

* 6,990,280 reviews 
* 150,346 businesses
* 11 metropolitans areas in USA
908,915 tips by 1,987,897 users
Over 1.2 million business attributes like hours, parking, availability, and ambience
Aggregated check-ins over time for each of the 131,930 businesses
[Source](https://www.yelp.com/dataset)
