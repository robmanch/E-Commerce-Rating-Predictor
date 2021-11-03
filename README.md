# E-Commerce-Rating-Predictor

## Introduction
Capturing ratings from the reviews is important in these times where a few words have the power to start a revolution. A review, once posted, can make, or break a product – it is important to learn from previous reviews and thus direct a customer’s attention accordingly. Evaluating ratings from customer’s reviews can benefit the organization in the following ways:
• Become more competitive
• Attract new customers
• Retain present customers
• Sell more products and services
• Reduce customer servicing
• Make customers more profitable
• Improve marketing messages and campaigns
In this project, we will sift through the reviews in our data set and attempt to gauge what sentiment the customer was expressing by trying to predict the rating a customer would give that product.

## The Problem Statement:

Our project will generate results/insights that may be feed into an e-commerce recommendation system. We are trying to predict Customer Ratings by analyzing customer data and reviews. Recommendation engines then can use this data along with other strategies to make product recommendations to customers. Negative Reviews are to be predicted as matches to the 1-star category and positive reviews as 5-stars.
The Dataset

To explore the subject further, we chose an interesting dataset with from Kaggle which provided us with real-world data scrapped from a Women’s Clothing E-Commerce Website:
https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews

The data set consists of the following columns:
1.	Clothing ID: Integer variable that refers to the specific piece being reviewed. 
2.	Age: Positive Integer variable of the reviewer's age.
3.	Title: String variable for the title of the review.
4.	Review Text: String variable for the review body.
5.	Rating: Positive Ordinal Integer variable for the product score by the customer [1 Worst, to 5 Best]

## Conclusion

•	Random Forest classifier with TF-IDF vectorizer outperforms the other classifiers.
•	Stratified split provides better results than non-stratified.
•	Feature extraction with PCA didn’t provide good results.

## References

1.	The data is acquired from the following source: Kaggle: https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews
