# -*- coding: utf-8 -*-
"""
@author: dpankin
"""
import pandas as pd #pandas library to work with large data set in a table format
import matplotlib.pyplot as plt # plot for crateing charts and graphs
import string #will need for functions that turn the text into list of words
import numpy as np

'''importing statistical methods'''
import statsmodels.api as sm #will use for OLS - linear regression
from scipy import stats # need this for spearman correlation

'''importing natural language processing tools'''
import nltk #nltk library to use for natural language processing (will be needed for key words analysis)
from nltk.corpus import stopwords #import english stopwords from nltk - will use to get rid of most common english words in the reviews' texts
from nltk import pos_tag #will use this to tag the words as the parts of speech (will be needed to lemmatizze text properly)
from nltk.tokenize import word_tokenize #will use to split strings into words
from nltk.stem import WordNetLemmatizer #will use to lemmatize words from a string

'''importing machine learning tools'''
from sklearn.feature_extraction.text import CountVectorizer # need this to transform the list of words from reviews into vectors
from sklearn.model_selection import train_test_split #need this to split dataset into training and testing data sets - for NLP predictive model
from sklearn.naive_bayes import MultinomialNB #nThe multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification).The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

import operator


''' this is required to transform POS tags allocated by pos_tag into something that'''

part = {
    'N' : 'n',
    'V' : 'v',
    'J' : 'a',
    'S' : 's',
    'R' : 'r'
}

lemmatizer = nltk.stem.WordNetLemmatizer()
wnl = WordNetLemmatizer()


#this function will transform POS tags to the format that lemmatizer WordNetLemmatizer function can understand
def convert_tag(penn_tag):
    '''
    convert_tag() accepts the **first letter** of a Penn part-of-speech tag,
    then uses a dict lookup to convert it to the appropriate WordNet tag.
    '''
    if penn_tag in part.keys():
        return part[penn_tag]
    else:
        # other parts of speech will be tagged as nouns
        return 'n';

#this function will assign POS tags to words
def tag_and_lem(element):
    '''
    tag_and_lem() accepts a string, tokenizes, tags, converts tags,
    lemmatizes, and returns a string
     list of tuples [('token', 'tag'), ('token2', 'tag2')...]
    '''
    sent = pos_tag(word_tokenize(element)) # must tag in context
    return ' '.join([wnl.lemmatize(sent[k][0], convert_tag(sent[k][1][0]))
                    for k in range(len(sent))]);
    
    
#this function will lemmatize text - transform words into their 'standard' form
def lemmatize_text(text):   
    return [lemmatizer.lemmatize(w) for w in text];

# this function will remove punctuation and english stop words
def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. get rid of punctuation
    2. get rid of stopwords
    3. Return the text as a list of words
    '''
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')];

# this function will remove punctuation signs
def remove_punctuation(text):
    '''
    Takes in a string of text, then performs the following:
    1. get rid of all punctuation
    2. Return the cleaned text as a list of words
    '''
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split()];


''' This part is about data import, transformation and prepration '''
#Import the original json file
amazon_reviews_TV = pd.read_json('/Users/panki/Desktop/reviews_Movies_and_TV_5/Movies_and_TV_5.json', lines = 'true') 
#split the helpfulnes score in 2 columns - overall number of votes and number of positive votes
amazon_reviews_TV[['positive_helpfulness_number','overall_helpfulness_number']] = pd.DataFrame(amazon_reviews_TV.helpful.values.tolist(), index= amazon_reviews_TV.index) 
# calculate the helpfulness ratio - postive helpfulness votes divided by overall helpfulness votes
amazon_reviews_TV['helpfulness_ratio'] = amazon_reviews_TV['positive_helpfulness_number']/amazon_reviews_TV['overall_helpfulness_number'] 
#make a new dataframe that will be used to gather statistics on products, add a column with mean star rating per product
Statistics_Per_Product_Full_List = amazon_reviews_TV.groupby('asin',as_index=False)['overall'].mean() #make a new dataframe that will be used to gather statistics on products, add a column with mean star rating per product
#rename the column with mean star rating per product
Statistics_Per_Product_Full_List.rename(columns={'overall': 'star_rating_mean'}, inplace=True) 
#bring the mean number back into the original data frame
amazon_reviews_TV = pd.merge(amazon_reviews_TV,Statistics_Per_Product_Full_List) 
#calculate star rating of the review to the distance to star rating mean value for this product
amazon_reviews_TV['star_rating_distance_to_mean'] = abs(amazon_reviews_TV['overall'] - amazon_reviews_TV['star_rating_mean']) 
# calculate helpfulness index with the following logic - if helpfulness ratio is >0.5, then review is helpful, otherwise - not helpful.
amazon_reviews_TV['helpfulness_index_50percent_cutoff'] = [1 if x > 0.5 else 0 for x in amazon_reviews_TV['helpfulness_ratio']]
# generate a new dataset which includes only reviews that got at least 5 votes on the review helpfulness
amazon_reviews_TV_nonzero_5plus = amazon_reviews_TV[amazon_reviews_TV['overall_helpfulness_number']>4]
# take 10 percent sample from the data set
sample10percent = amazon_reviews_TV_nonzero_5plus.sample(frac=0.1)


''' This part is splitting the text into words and counting them'''
# generate column with words (text) from the review text in the lemmatized form
sample10percent["text_lemmas"] = sample10percent['reviewText'].apply(tag_and_lem)
# generate column with words (text) from the review text in the lemmatized form and in lower case
sample10percent["text_lemmas_lower"] = sample10percent['text_lemmas'].apply(str.lower)
# generate column with the list of words from the review text in the lemmatized form, in lower case and witout english stop words
sample10percent["list_of_words_lemmas"] = sample10percent['text_lemmas_lower'].apply(text_process)
# generate column with the list of words from the review text in the lemmatized form, in lower case including stop words
sample10percent["complete_list_of_words_lemmas"] = sample10percent['text_lemmas_lower'].apply(remove_punctuation)
#get he number of words in the review
amazon_reviews_TV['Review_Length_Words'] = amazon_reviews_TV['list_of_words_lemmas'].str.len() #Calculate the lenght of the review in words, add this number as a column to the dataset




''' this part is to check correlations between reviews length and the helpfulness '''
# produce histogram to show the number of reviews with different length in words across the dataset. Limit the X acis at 1500 words for readability
sample10percent['Review_Length_Words'].plot.hist(bins=150, xlim=(0,1500), color ='orange', figsize=(20,10))
# produce scatter chart to show how reviews of different length are spread across helpfulness ratio. Use a subset of 1000 reviews to generate readable chart (otherwise there are too many dots)
sample1000.plot.scatter(['helpfulness_ratio']
# run spearman correlation between helpfulness ratio and the length of the review in words
stats.spearmanr(sample10percent['Review_Length_Words'], sample10percent['helpfulness_ratio'])
# add columns that would only count number of words after a certain threshold - after 100, 150, 200, 300
sample10percent['word_count_min_200']= sample10percent['Review_Length_Words']
sample10percent.loc[sample10percent['Review_Length_Words'] <200, 'word_count_min_200'] = 200
sample10percent['word_count_min300']= sample10percent['Review_Length_Words']
sample10percent.loc[sample10percent['Review_Length_Words'] <300, 'word_count_min_300'] = 300
sample10percent['word_count_min100']= sample10percent['Review_Length_Words']
sample10percent.loc[sample10percent['Review_Length_Words'] <100, 'word_count_min_100'] = 100
sample10percent['word_count_min150']= sample10percent['Review_Length_Words']
sample10percent.loc[sample10percent['Review_Length_Words'] <150, 'word_count_min_150'] = 150

# run spearman correlation between helpfulness ratio and those columns
stats.spearmanr(sample10percent['word_count_min_200'], sample10percent['helpfulness_ratio'])
stats.spearmanr(sample10percent['word_count_min_300'], sample10percent['helpfulness_ratio'])
stats.spearmanr(sample10percent['word_count_min_150'], sample10percent['helpfulness_ratio'])
stats.spearmanr(sample10percent['word_count_min_100'], sample10percent['helpfulness_ratio'])

# create separate data frames with reviews shorter than 20, 30 and 50 words, calculate proportion of the unhelpful reviews
samplelessthan30words = sample10percent[sample10percent['Review_Length_Words']<30]
samplelessthan30words['helpfulness_index_50percent_cutoff'].value_counts(normalize)
samplelessthan20words = sample10percent[sample10percent['Review_Length_Words']<20]
samplelessthan20words['helpfulness_index_50percent_cutoff'].value_counts(normalize)
samplelessthan50words = sample10percent[sample10percent['Review_Length_Words']<50]
samplelessthan50words['helpfulness_index_50percent_cutoff'].value_counts(normalize)


''' this part is to check correlations between star rating and the helpfulness '''
#investigate the data set - plot histagram for star rating
sample10percent['overall'].plot.hist(bins=5, xlim=(1,5), color ='orange', figsize=(30,15), fontsize=30 )
#investigate the data set - count different star rating in the dataset
sample10percent['overall'].value_counts()
# make a scatter chart from a subset of the data
sample1000.plot.scatter(['helpfulness_ratio'],['overall'],color ='red', figsize=(20,10), fontsize=30)
# run spearman correlation between star rating and the helpfulness of the reivew
stats.spearmanr(sample10percent['overall'], sample10percent['helpfulness_ratio'])


''' This part is to test regression to identify most useful variables '''
# get the linear regression on some of the variables

testingStarRating = sample10percent['overall']
model=sm.OLS(sample10percent["helpfulness_ratio"],testingStarRating).fit()
model.summary()

testingReviewLength = sample10percent['Review_Length_Words']
model=sm.OLS(sample10percent["helpfulness_ratio"],testingReviewLength).fit()
model.summary()

testingReviewLength = sample10percent['star_rating_distance_to_mean']
model=sm.OLS(sample10percent["helpfulness_ratio"],testingReviewLength).fit()
model.summary()

# need to get a smaller sample for performance reasons
sample4000 = sample10percent.sample(n=4000)

''' This part trains the model based on the words in the review text '''
# transform list of words (from the reviews) into vectors, use 4000 reviews sample and top 250 words for performance reasons
transformed_to_vector = CountVectorizer(max_features=250,analyzer=text_process ).fit(sample4000['text_lemmas_lower'])
# transform vectors into a matrix
transformed_to_matrix = transformed_to_vector.transform(sample4000['text_lemmas_lower'])
#create variable X to contain the matrix
x= transformed_to_matrix
#create variable Y to contain info helpfuleness indicator info
y=sample4000['helpfulness_index_50percent_cutoff']
#split the data set into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# rub Multinominal Naive Bias function on the data (assume that all words in the text are independent from each other )
nb = MultinomialNB()
nb.fit(x_train, y_train)
# test nb model against the test data set and print the results
predictions = nb.predict(x_test)
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))



''''This part is to evaluate effect of the simplified version of the Flesch-Kincaid Reading Ease Indicator - Readability'''
sample10percent = sample10percent[sample10percent['Review_Length_Words']>0]
sample10percent['vowels'] = sample10percent['reviewText'].str.lower().str.count(r'[aeiou]')
sample10percent['sentences'] = sample10percent['reviewText'].str.lower().str.count(r'[.!?]')
sample10percent['Simplified_FK'] = (0.39*(sample10percent['Review_Length_Words']/sample10percent['sentences'])+11.8*(sample10percent['vowels']/sample10percent['Review_Length_Words']) - 15.59)
