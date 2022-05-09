# cs5293sp22-project3

### Text Analytics Project 3
### Author: Vudumula Kranthi Kumar Reddy


# About

The task for this project is to train a model that learns from the training set and be evaluated on the validation set. One of the most important aspects of this project is creating the appropriate set of features. Features may include, n-grams in the document, number of letters in the redacted word, the number of spaces in the redacted word, the sentiment score of the review, previous word, next word. The key to this task is to generate a precision, recall, and f1-score of the code for the dataset.


# Project Structure

<img width="400" alt="Screen Shot 2022-05-08 at 6 57 06 PM" src="https://user-images.githubusercontent.com/98420519/167321097-6a24aba7-d877-4cd2-8882-efe733eeac6d.png">


# Table of Contents

**[Required Packages](#required-packages)**<br>
**[Function Description](#function-description)**<br>
**[How to run the project](#how-to-run-the-project)**<br>
**[Bugs and Assumptions](#bugs-and-assumptions)**<br>
**[References](#references)**<br>


# Required Packages

The below mentioned are the packages used for this Project:
* pandas
* csv
* nltk
* re
* sklearn
* warnings


# Function Description

1. Unredactor.py :

This File contains all the functions that has been used for predicting the Precision, Recall and F-1 Scores.

- [x] Readataurl() :
  - This function is used to read the data from the the url. In this function we extract the unredactor.tsv raw data for the url by specifying the delimiters.

- [x] preprocess() : 
  -  This function is used to clean the read data using normalaization techniques like stemming, lemataization etc. In this function firstly we append all the sentences data into a list from a data frame and then we convert all the characters to lower. Then we check if there are any digits and replace them as empty and then using nltk we tokenize the data. Finally, we check if there are any stop words and then remove them by not appending them into cleanlist. Then we rewrite the sentences column with the cleanlist data.

- [x] Sentimentscore():
  - This function is used to find the Sentiment score of the reviews. In this function we first create an empty positivity list and we create a SentimentIntensityAnalyzer() model. Using this model we find the polarity and append all the positive data polarity into positivity list. Then finally we append the positivity list data into a datafram.

- [x] fngrams():
  - This function is used to find the ngrams. In this function we are using 4 gram for the sentences column in the dataframe. Then finally, we append all then 4 grams of the sentences into a empty list n_grams and add this n_grams to the data set.

- [x] Vectorize():
  - This function is used to vectorize all the redacted sentences. In this function we are using a CountVectorizer() model to vectorize the data. After vectorizing the data we append this data into a new dataframe. Then we concatenate the two dataframes using pd.concat() method. Finally we keep the required columns and drop the unwanted columns.

- [x] Prediction():
  - This function is used to predict the required precision, recall and f1-scores. In this function we first filter all the testing, training and validation data into different dataframes that is df4 and df5. Then we loc the required fields into the x, y variables that is vectorized sentences into the x variable and names into the y variable. Then we add the testing sentences into the z variables and testing names into the n variable. Then we create a SVM Model  and fit the data into that model for training the data purpose. Then we predict the model with the testing data and store it in y_predi. Then finally, we find the precision, recall and f1-score using the testing names and y_predi for getting the desired result.

2. test_data.py :

This file contains all the test cases for the above mentioned functions, to check whether every function is running correctly or not.

- [x] test_readdata():
  - This function is used to check whether the Readdata() is functioning properly or not. In this function I am trying to assert that the resulting output is not None.

- [x] test_preprocessing():
  - This function is used to check whether the preprocess() is functioning properly or not. In this function I am trying to assert that the resulting output is greater than 0 and is not None.

- [x] test_sentimentscore():
  - This function is used to check whether the Sentimentscore() is functioning properly or not. In this function I am trying to assert that the resulting output is not None.

- [x] test_fngrams():
  - This function is used to check whether the fngrams() is functioning properly or not. Here In this function I am trying to assert that the resulting output is equal to the output in the fngrams().

- [x] test_vectorize();
  - This function is used to check whether the Vectorize() is functioning properly or not. Here In this function I am trying to assert that the resulting output is equal to the output in the Vectorize().

- [x] test_prediction():
  - This function is used to check whether the prediction() is functioning properly or not. In this function I am trying to assert that the resulting output is not None.

3. pytest.ini :

- [x] Aditionally, I have created a *pytest.ini* file and added some code to eliminate some of the warnings that are being displayed while running the test cases.


# How to run the project

- First we need to clone the git repository with the repository url.
- Then we need to change to the cloned repository.
- Then we need to run the below command:
```
pipenv install
```

```
pipenv shell
```

- By running the above commands we are creating a new virtul environment by installing the all the dependencies from the Pipfile.
- Then for running the test cases that is the test_data.py file we need to run the below command:
```
pipenv run python -m pytest
```

- Then for running the unredactor.py we need to run the below command:

```
pipenv run python unredactor.py
```

# Bugs and Assumptions

* When checked for this model accuracy rate it was 5.405% (approx.). By this we can say that one may/may not get the correct prediction all the times.
* One might face problem when the passed url is not active or not functioning. And when extracting the data they have specify proper delimiters by removing the bad lines.
* One might get errors when the normalization (cleaning of data) is not done correctly.
* One might have trouble when the normalized list that is the cleanlist is not vectorized properly.
* one may get errors when the packages are not installed properly. For example, In this project, I have downloaded the stopwords for which they have to use nltk.download('stopwords') and I have used SentimentIntensityAnalyzer model for which they have to downlaod nltk.download('vader_lexicon').
* one might face problem when the passsed x and y values for training vary with respect to their shapes or row, column size.

# References

* [What are n grams](https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/)
* [Pandas Parse Error](https://stackoverflow.com/questions/18016037/pandas-parsererror-eof-character-when-reading-multiple-csv-files-to-hdf5)
* [Merging Two Dataframes](https://pandas.pydata.org/docs/user_guide/merging.html)
* [To Remove Warnings](https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings)
* [File Reading](https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781783551668/1/ch01lvl1sec10/reading-and-writing-csv-tsv-files-with-python)




