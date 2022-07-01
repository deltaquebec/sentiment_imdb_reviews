# Data Visualization and Sentiment Analysis of Movie Reviews across Four Neural Network NLP Models

This goal of this project is twofold: 1) **practice with data exploration and visualization** 2) **classify sentiment polarity of IMBD reviews as "positive" or "negative"**. The latter is achieved through four neural network models, whose performances are compared. This project serves as a learning tool for practicing with data visualization and building neural networks.

The project is arranged as follows:

1. **Visualization**
- Data preparation and data cleaning
- Wordcloud representations
- n-gram (mono-,bi-,tri-gram)
- Number of characters in text
- Number of words in text
- Average word length represented as probability density
2. **Neural Network Models**
- CNN
- RNN
- RCNN
- LSTM

# 1. Visualization
## Data preparation and data cleaning

The [Large Movie Review Dataset v1.0](http://ai.stanford.edu/~amaas/data/sentiment/) is used, which consists of movie reviews classified as positive or negative. 25,000 movie reviews represent the training dataset and 25,000 for the testing dataset. The review dataset can be imported from keras.

```
from tensorflow.keras.datasets import imdb
```

Appropriate functions and libraries are imported.

```
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
```

Mapping an index of words identifies words with some label or index such that each word (or token) has some identifier. This will be useful when we assign the training and test data to their own respective CSV.

```
indx = imdb.get_word_index()
indxword = {v: k for k, v in idx.items()}
```

We define the vocabulary size as the top 5000 from the dataset, and simultaneously assign the training and test data to two tuples.

```
vocabulary_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)
```

In part to practice with CSV data and to be able to individually manipulate the training and test data, we transform the data to CSV. For the purpose of data visualization, we take the (X_train, y_train) tuple to 'train.csv' and the (X_test, y_test) tuple to 'test.csv'. The positive examples in the training data are sent to their own CSV as 'train_pos.csv'; the negative examples in the trianing data are sent to their own CSV as 'train_neg.csv'.

```
with open('train.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    for i in range(0, len(X_train)):
        label = y_train[i]
        review = ' '.join([indxword[o] for o in X_train[i]])
        writer.writerow([review, label])

with open('test.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    for i in range(0, len(X_test)):
        label = y_test[i]
        review = ' '.join([indxword[o] for o in X_test[i]])
        writer.writerow([review, label])

with open('train_pos.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    for i in range(0, len(X_train)):
        label = y_train[i]
        if label == 1:
            review = ' '.join([indxword[o] for o in X_train[i]])
            writer.writerow([review, label])
        elif label == 0:
            continue

with open('train_neg.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    for i in range(0, len(X_train)):
        label = y_train[i]
        if label == 0:
            review = ' '.join([indxword[o] for o in X_train[i]])
            writer.writerow([review, label])
        elif label == 1:
            continue
```

The CSV are changed into a pandas dataframe.

```
train_data = pd.read_csv('train.csv', header=None)
test_data = pd.read_csv('test.csv', header=None)
train_data_pos = pd.read_csv('train_pos.csv', header=None)
train_data_neg = pd.read_csv('train_neg.csv', header=None)
```

For now, the dataframes we have only contain the reviews themselves and whether they are positive (1) or negative (0). Dependning on what kind of information we are interested in, we can append to this dataframe additional information such as character count, word count, unique word count, uppercase count,  stopwords count, etc.. We loop through each of the training dataframes and append to them such information.

```
data = [train_data,train_data_pos,train_data_neg]
fulldata = []

stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stopwords = set(stopwords_list.decode().splitlines())

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
```

Each dataframe, however, is still working with raw data; we need to clean the data so we can have good information free of personal idiosyncrasies and can meaningfully compare the data. We map common contractions to theyr pariphrastic forms such as "isn't" to "is not" and "they're" to "they are". Other noise in the data likewise cleaned (this particular cleaning follows from [this work](https://github.com/072arushi/Movie_review_analysis)).

```
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
```

Some visualization tasks ask for data in _toto_, so the total training dataframe, the positive dataframe, and the negative dataframe get joined into respective continuous texts, which are delineated so that tokens can be accessed and analyzed.

```
tot=' '.join(train_data[11])
pos=' '.join(train_data_pos[11]) 
neg=' '.join(train_data_neg[11]) 

stringtot = tot.split(" ")
stringpos = pos.split(" ")
stringneg = neg.split(" ")
```

## Wordcloud representations

Wordclouds are visualizations of (text) data in which the size of a word represents its frequency or importance in that data. Wordclouds are handy for visualization-at-a-glance, and have the enjoyable consequence of making a report more lively. 

Generating wordclouds for each of the total training data and the positive and negative data follow from defining the numerical limiter for how many words will be considered after sifting through stopwords.

```
wordcount = 20000

wordcloudtot = WordCloud(scale=3, background_color ='black', max_words=wordcount, stopwords=stopwords).generate(tot)
wordcloudpos = WordCloud(scale=3, background_color ='black', max_words=wordcount, stopwords=stopwords).generate(pos)
wordcloudneg = WordCloud(scale=3, background_color ='black', max_words=wordcount, stopwords=stopwords).generate(neg)
```

Each of these may then be plotted accordingly.

```
f = plt.figure()
    
f.add_subplot(211)
plt.imshow(wordcloudtot,interpolation='bilinear')
plt.title('Total Sentiment')
plt.axis('off')

f.add_subplot(223)
plt.imshow(wordcloudpos,interpolation='bilinear')
plt.title('Positive Sentiment')
plt.axis('off')

f.add_subplot(224)
plt.imshow(wordcloudneg,interpolation='bilinear')
plt.title('Negative Sentiment')
plt.axis('off')
    
f.suptitle('Sentiment Wordclouds')
plt.show(block=True)
```


## n-gram (mono-,bi-,tri-gram)

[n-grams](https://web.stanford.edu/~jurafsky/slp3/3.pdf) track the frquency in which (word) tokens appear. 1-grams (monograms) refer to the frequency in which single word tokens appear; 2-grams (bigrams) refer to the frequency in which two word tokens appear together; 3-grams (trigrams) refer to the frequency in which three word tokens appear together. Roughly, such frequencies will follow a [Zipf-like distribution](https://web.archive.org/web/20021010193014/http://linkage.rockefeller.edu/wli/zipf/).

We loop through 1-, 2-, and 3-gram analyses for each of the total training data, positive training data, and negative training data, and save the top fifteen of each n-grams saved as dataframes.

```
gram = [1,2,3]
string = [stringtot,stringpos,stringneg]

save = []

for i in gram:
    for j in string:
        n_gram = (pd.Series(nltk.ngrams(j, i)).value_counts())[:15]
        n_gram_df=pd.DataFrame(n_gram)
        n_gram_df = n_gram_df.reset_index()
        n_gram_df = n_gram_df.rename(columns={"index": "word", 0: "count"})
        save.append(n_gram_df)
```

The n-gram distributions are plotted accordingly.

```
sns.set()

fig, axes = plt.subplots(3, 3)

plt.subplots_adjust(hspace = 0.7)
plt.subplots_adjust(wspace = 0.9)

sns.barplot(data=save[0], x='count', y='word', ax=axes[0,0]).set(title="1-gram for total")
sns.barplot(data=save[1], x='count', y='word', ax=axes[0,1]).set(title="1-gram for positive")
sns.barplot(data=save[2], x='count', y='word', ax=axes[0,2]).set(title="1-gram for negative")
sns.barplot(data=save[3], x='count', y='word', ax=axes[1,0]).set(title="2-gram for total")
sns.barplot(data=save[4], x='count', y='word', ax=axes[1,1]).set(title="2-gram for positive")
sns.barplot(data=save[5], x='count', y='word', ax=axes[1,2]).set(title="2-gram for negative")
sns.barplot(data=save[6], x='count', y='word', ax=axes[2,0]).set(title="3-gram for total")
sns.barplot(data=save[7], x='count', y='word', ax=axes[2,1]).set(title="3-gram for positive")
sns.barplot(data=save[8], x='count', y='word', ax=axes[2,2]).set(title="3-gram for negative")

plt.show()
```

## Number of characters in text

The number of characters in the text refers to simply that: the number of written characters. These may be extracted for each dataframe and plotted accordingly.

```
f = plt.figure(figsize=(12,8))
    
text_len=train_data[11].str.len()
f.add_subplot(211)
plt.hist(text_len,color='blue')
plt.title("Total Reviews")
plt.xlabel("number of characters")
plt.ylabel("number of reviews")
    
text_len=train_data_pos[11].str.len()
f.add_subplot(223)
plt.hist(text_len,color='green')
plt.title("Text with Good Reviews")
plt.xlabel("number of characters")
plt.ylabel("number of reviews")
    
text_len=train_data_neg[11].str.len()
f.add_subplot(224)
plt.hist(text_len,color='red')
plt.title("Text with Bad Reviews")
plt.xlabel("number of characters")
plt.ylabel("number of reviews")
    
f.suptitle("Characters in Texts")
plt.show(block=True)  
```

## Number of words in text
## Average word length represented as probability density

# 2. Neural Network Models
## CNN
## RNN
## RCNN
## LSTM
