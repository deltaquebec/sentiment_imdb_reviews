import csv
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import string
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import requests
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from pickle import load
from tensorflow.keras.models import Model
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.utils import get_file
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from statistics import mean
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from statistics import median
from wordcloud import WordCloud, STOPWORDS
from imp import reload

#########################################################################
# get data
#########################################################################
# map words to indices
indx = imdb.get_word_index()
indxword = {v: k for k, v in indx.items()}

# keep top 5000 words
vocabulary_size = 5000

# split dataset into two equal parts: training set and test set
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)
    
#########################################################################
# move data to csv 
#########################################################################
# move training data to csv
with open('train.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    for i in range(0, len(X_train)):
        label = y_train[i]
        review = ' '.join([indxword[o] for o in X_train[i]])
        writer.writerow([review, label])

# move testing data to csv
with open('test.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    for i in range(0, len(X_test)):
        label = y_test[i]
        review = ' '.join([indxword[o] for o in X_test[i]])
        writer.writerow([review, label])

# move positive training data to csv
with open('train_pos.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    for i in range(0, len(X_train)):
        label = y_train[i]
        if label == 1:
            review = ' '.join([indxword[o] for o in X_train[i]])
            writer.writerow([review, label])
        elif label == 0:
            continue

# move negative training data to csv
with open('train_neg.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    for i in range(0, len(X_train)):
        label = y_train[i]
        if label == 0:
            review = ' '.join([indxword[o] for o in X_train[i]])
            writer.writerow([review, label])
        elif label == 1:
            continue

#########################################################################
# move data from csv to pandas dataframes 
#########################################################################
# save training and test data as respective dataframe
train_data = pd.read_csv('train.csv', header=None)
test_data = pd.read_csv('test.csv', header=None)
# save training and test positive and negative data as respective dataframe
train_data_pos = pd.read_csv('train_pos.csv', header=None)
train_data_neg = pd.read_csv('train_neg.csv', header=None)

data = [train_data,train_data_pos,train_data_neg]
fulldata = []

#########################################################################
# define stopwords
#########################################################################
# stop words
stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stopwords = set(stopwords_list.decode().splitlines())

#########################################################################
# expand data frame such that each review has numerical analyses of some kind
#########################################################################
for i in data:
    # char count
    i[2]=i[0].str.len()
    # word count
    i[3]=i[0].apply(lambda x: len(str(x).split()))
    # unique count
    i[4]=i[0].apply(lambda x: len(set(str(x).split())))
    # uppercase count
    i[5]=i[0].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    # stopwords count
    i[6]=i[0].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))
    # average word length
    i[7]=i[0].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    # punctuation count
    i[8]=i[0].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    # sentences count
    i[9]=i[0].apply(lambda x: len(re.findall("/n",str(x)))+1)
    # average sentence length
    i[10]=i[0].apply(lambda x: np.mean([len(re.findall("/n",str(x)))+1]))
    
    fulldata.append(i)

#########################################################################
# clean data
#########################################################################

# map common contracted words to periphrasis; this code follows from https://github.com/072arushi/Movie_review_analysis
mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot",
           "'cause": "because", "could've": "could have", "couldn't": "could not",
           "didn't": "did not",  "doesn't": "does not", "don't": "do not",
           "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did",
           "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
           "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
           "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have",
           "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
           "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
           "might've": "might have","mightn't": "might not","mightn't've": "might not have",
           "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
           "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't":"shall not",
           "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
           "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
           "she's": "she is", "should've": "should have", "shouldn't": "should not",
           "shouldn't've": "should not have", "so've": "so have","so's": "so as",
           "this's": "this is","that'd": "that would", "that'd've": "that would have",
           "that's": "that is", "there'd": "there would", "there'd've": "there would have",
           "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
           "they'll": "they will", "they'll've": "they will have", "they're": "they are",
           "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
           "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
           "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
           "what'll've": "what will have", "what're": "what are",  "what's": "what is",
           "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
           "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
           "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
           "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
           "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are",
           "y'all've": "you all have","you'd": "you would", "you'd've": "you would have",
           "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
punct = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', punct))
def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stopwords])
def word_replace(text):
    return text.replace('<br />','')
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)
def preprocess(text):
    text=clean_contractions(text,mapping)
    text=text.lower()
    text=word_replace(text)
    text=remove_urls(text)
    text=remove_html(text)
    text=remove_stopwords(text)
    text=remove_punctuation(text)
    text=lemmatize_words(text)
    return text

clean=[]
for i in fulldata:
    i[11] = i[0].apply(lambda text: preprocess(text))
    clean.append(i)

#########################################################################
# join total, positive, and negative into continuous textblocks (use index [11] for cleaned)
#########################################################################
# extract total and pos/neg from total reviews as continuous text from reviews with stopwords removed
tot=' '.join(train_data[11])                         # join all reviews in training set
pos=' '.join(train_data_pos[11])                     # join all positive reviews in training set
neg=' '.join(train_data_neg[11])                     # join all negative reviews in training set

#########################################################################
# distinct words
#########################################################################

# delineate word in each of total, positive, and negative reviews
stringtot = tot.split(" ")
stringpos = pos.split(" ")
stringneg = neg.split(" ")

#########################################################################
# wordcloud for cleaned texts
#########################################################################
def cloud():
    # limit word count
    wordcount = 20000

    # setup wordcloud; generate wordcloud
    wordcloudtot = WordCloud(scale=3, background_color ='black', max_words=wordcount, stopwords=stopwords).generate(tot)
    wordcloudpos = WordCloud(scale=3, background_color ='black', max_words=wordcount, stopwords=stopwords).generate(pos)
    wordcloudneg = WordCloud(scale=3, background_color ='black', max_words=wordcount, stopwords=stopwords).generate(neg)

    # store to file
    #wordcloudtot.to_file("wordcloudtot.png")
    #wordcloudpos.to_file("wordcloudpos.png")
    #wordcloudneg.to_file("wordcloudneg.png")

    # show wordclouds together
    f = plt.figure()
    
    # total reviews wordcloud
    f.add_subplot(211)
    plt.imshow(wordcloudtot,interpolation='bilinear')
    plt.title('Total Sentiment')
    plt.axis('off')
    
    # positive reviews wordcloud
    f.add_subplot(223)
    plt.imshow(wordcloudpos,interpolation='bilinear')
    plt.title('Positive Sentiment')
    plt.axis('off')
    
    # negative reviews wordcloud
    f.add_subplot(224)
    plt.imshow(wordcloudneg,interpolation='bilinear')
    plt.title('Negative Sentiment')
    plt.axis('off')
    
    # plot
    f.suptitle('Sentiment Wordclouds')
    plt.savefig("vis_data_cloud.png", dpi=300)
    plt.show(block=True)

#########################################################################
# n-gram definition for cleaned texts
#########################################################################
def n_gram():
    # items to loop through
    gram = [1,2,3]
    string = [stringtot,stringpos,stringneg]

    # save each n-gram for plotting
    save = []

    # loop through each of tot, pos, and neg dataframes for each type of 1, 2, and 3 grams
    for i in gram:
        for j in string:
            # look for top 15 used items
            n_gram = (pd.Series(nltk.ngrams(j, i)).value_counts())[:15]
            # save as dataframe
            n_gram_df=pd.DataFrame(n_gram)
            n_gram_df = n_gram_df.reset_index()
            # aquire index, word, count
            n_gram_df = n_gram_df.rename(columns={"index": "word", 0: "count"})
            # append data to save
            save.append(n_gram_df)

    #set seaborn plotting aesthetics as default
    sns.set()

    # define plotting region (3 rows, 3 columns)
    fig, axes = plt.subplots(3, 3)

    # adjust space between each subplot
    plt.subplots_adjust(hspace = 0.7)
    plt.subplots_adjust(wspace = 0.9)

    # create barplot for each data in save
    sns.barplot(data=save[0], x='count', y='word', ax=axes[0,0]).set(title="1-gram for total")
    sns.barplot(data=save[1], x='count', y='word', ax=axes[0,1]).set(title="1-gram for positive")
    sns.barplot(data=save[2], x='count', y='word', ax=axes[0,2]).set(title="1-gram for negative")
    sns.barplot(data=save[3], x='count', y='word', ax=axes[1,0]).set(title="2-gram for total")
    sns.barplot(data=save[4], x='count', y='word', ax=axes[1,1]).set(title="2-gram for positive")
    sns.barplot(data=save[5], x='count', y='word', ax=axes[1,2]).set(title="2-gram for negative")
    sns.barplot(data=save[6], x='count', y='word', ax=axes[2,0]).set(title="3-gram for total")
    sns.barplot(data=save[7], x='count', y='word', ax=axes[2,1]).set(title="3-gram for positive")
    sns.barplot(data=save[8], x='count', y='word', ax=axes[2,2]).set(title="3-gram for negative")

    # plot
    plt.savefig("vis_data_ngram.png", dpi=300)
    plt.show()
    
#########################################################################
# number of characters in cleaned texts
#########################################################################
def char():
    # show character distributions together
    f = plt.figure(figsize=(12,8))
    
    # total reviews
    text_len=train_data[11].str.len()
    f.add_subplot(211)
    plt.hist(text_len,color='blue')
    plt.title("Total Reviews")
    plt.xlabel("number of characters")
    plt.ylabel("number of reviews")
    
    # positive reviews
    text_len=train_data_pos[11].str.len()
    f.add_subplot(223)
    plt.hist(text_len,color='green')
    plt.title("Text with Good Reviews")
    plt.xlabel("number of characters")
    plt.ylabel("number of reviews")
    
    # negative reviews
    text_len=train_data_neg[11].str.len()
    f.add_subplot(224)
    plt.hist(text_len,color='red')
    plt.title("Text with Bad Reviews")
    plt.xlabel("number of characters")
    plt.ylabel("number of reviews")
    
    # plot
    f.suptitle("Characters in Texts")
    plt.savefig("vis_data_char.png", dpi=300)
    plt.show(block=True)  

#########################################################################
# number of words in cleaned texts
#########################################################################
def words():
    # number of words together
    f = plt.figure(figsize=(12,8))

    # total reviews
    text_len=train_data[11].str.split().map(lambda x: len(x))
    #text_len=train_data[3]
    f.add_subplot(211)
    plt.hist(text_len,color='blue')
    plt.title("Total Reviews")
    plt.xlabel("number of words")
    plt.ylabel("number of reviews")

    # positive reviews
    text_len=train_data_pos[11].str.split().map(lambda x: len(x))
    #text_len=train_data_pos[3]
    f.add_subplot(223)
    plt.hist(text_len,color='green')
    plt.title("Text with Good Reviews")
    plt.xlabel("number of words")
    plt.ylabel("number of reviews")

    # negative reviews
    text_len=train_data_neg[11].str.split().map(lambda x: len(x))
    #text_len=train_data_neg[3]
    f.add_subplot(224)
    plt.hist(text_len,color='red')
    plt.title("Text with Bad Reviews")
    plt.xlabel("number of words")
    plt.ylabel("number of reviews")

    # plot
    f.suptitle("Words in texts")
    plt.savefig("vis_data_words.png", dpi=300)
    plt.show()

#########################################################################
# average word length in cleaned texts
#The y-axis in a density plot is the probability density function for the kernel density estimation.
#However, we need to be careful to specify this is a probability density and not a probability. The difference is the probability
#density is the probability per unit on the x-axis. To convert to an actual probability, we need to find the area under the curve
#for a specific interval on the x-axis. Somewhat confusingly, because this is a probability density and not a probability,
#the y-axis can take values greater than one. The only requirement of the density plot is that the total area under the curve
#integrates to one. I generally tend to think of the y-axis on a density plot as a value only for relative comparisons between different categories.
#https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
#########################################################################
def leng():
    # average word length together
    f = plt.figure(figsize=(20,10))

    # total reviews
    word=train_data[11].str.split().apply(lambda x : [len(i) for i in x])
    #text_len=train_data[7]
    f.add_subplot(131)
    sns.histplot(word.map(lambda x: np.mean(x)),stat='density',kde=True,color='blue')
    plt.title("Total Reviews")
    plt.xlabel("average word length")
    plt.ylabel("probability density")

    # positive reviews
    word=train_data_pos[11].str.split().apply(lambda x : [len(i) for i in x])
    #text_len=train_data_pos[7]
    f.add_subplot(132)
    sns.histplot(word.map(lambda x: np.mean(x)),stat='density',kde=True,color='green')
    plt.title("Text with Good Reviews")
    plt.xlabel("average word length")
    plt.ylabel("probability density")

    # negative reviews
    word=train_data_neg[11].str.split().apply(lambda x : [len(i) for i in x])
    #text_len=train_data_neg[7]
    f.add_subplot(133)
    sns.histplot(word.map(lambda x: np.mean(x)),stat='density',kde=True,color='red')
    plt.title("Text with Bad Reviews")
    plt.xlabel("average word length")
    plt.ylabel("probability density")

    # plot
    f.suptitle("Average word length in texts")
    plt.savefig("vis_data_leng.png", dpi=300)
    plt.show()
    
cloud()
n_gram()
char()
words()
leng()






    
    
