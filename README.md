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

**2. Feature selection:** I performed feature selection based on the univariate statistical tests by computing the ANOVA F-value betwen the numerical features (e.g., f_1, f_2...) and the label target. The new dataset includes the most 25 features and f_46 because it is a categorical feature. 

**3. Splitting the dataset** to train test sets based on the following specifications: Train size: 75%, test size: 25%, stratifying based on the y label  to ensure that both the train and test sets have the same class proportion similar to the original dataset. After that, I normalized both train and test datasets using the StandardScaller() to remove the mean and scaling to unit variance. 

**4. Data augmentation**: Since the dataset is highly imbalanced, i implemented multiple data augmentation techniques to improve the quality of the dataset based on the following algorithms:

* Synthetic Minority Oversampling Technique(SMOTE): The sample in minority class is first selected randomly and its k nearest minority class neighbors are found based on the K-nearest neighbors algorithm. The synthetic data is generated between two instances in feature space. 
* Adaptive Synthetic Sampling (ADASYN): The synthetic data for minority class is generated based on the desnity distribution of the minority class. Specifically, more data is created in area with low density of minority class and less data is generated in area with high density of minority example
* SMOTE-TOMEK: Combine SMOTE and TOMEK techniqes where the oversampling technique for minority class and the cleaning using Tomek links.  
* SMOTE- ENN: Combine SMOTE and Edited Nearest Neighbours (ENN) techniques where the oversampling technique for minority class and the cleaning using ENN

Source: [Imbalanced learn](https://imbalanced-learn.org/stable/references/over_sampling.html)
 
**5. Build a simple deep learning network** and combine with multiple data augmentation techniques [See code here](https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/oil_spill_detection_deepLearningModel.ipynb)
<img src="https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/DNN_summary.png">

**6. Implement ensemble learning algorithms**, which include Random Forest, and XGBoost, and compare the model performance given the unbalanced dataset for oil spill detection [See code here](https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/oil_spill_detection_model.ipynb)

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


