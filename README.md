# Data Visualization and Sentiment Analysis of Movie Reviews across Four Neural Network NLP Models

<p align="center"> 
<img src="/assets/georges_melies.jpg" alt="Georges Melies">
</p>

The goal of this project is twofold: 1) **practice with data exploration and visualization** 2) **classify sentiment polarity of IMBD reviews as "positive" or "negative"**. The latter is achieved through four neural network models, whose performances are compared. This project serves as a learning tool for practicing with data visualization and building neural networks.

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

It should be noted that this project was conducted using the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). To optimize completion time, each task was invited to perform on the GPU: NVIDIA GeFORCE RTX 3070. Preliminary tests were done on the CPU: AMD Ryzen 7 3700X 8-Core Processor 3.59 GHz. Your completion times may be different according to the processing unit you use.

# 1. Visualization

The tasks of data visualization are contained in a single python file: _data_visualization.py_. The plots of distributions are saved as .png files.

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

Each dataframe, however, is still working with raw data; we need to clean the data so we can have good information free of personal idiosyncrasies and can meaningfully compare the data. We map common contractions to theyr pariphrastic forms such as "isn't" to "is not" and "they're" to "they are". Other noise in the data likewise cleaned (this particular cleaning follows from [this work](https://github.com/072arushi/Movie_review_analysis)). These get appended as dataframe[11].

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

Some visualization tasks ask for data _in toto_, so the total training dataframe, the positive dataframe, and the negative dataframe get joined into respective continuous texts, which are delineated so that tokens can be accessed and analyzed.

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

Generating wordclouds for each of the total training data and the positive and negative data follows from defining the numerical limiter for how many words will be considered after sifting through stopwords.

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
plt.savefig("vis_data_cloud.png", dpi=300)
plt.show(block=True)
```

Inspection of each wordcloud is not immediately useful; overall sentiment cannot be meaningfully extracted by inspection, as no extremes in language can be easily identified.

![vis_data_cloud](https://github.com/deltaquebec/sentiment_imdb_reviews/blob/main/assets/vis_data_cloud.png)

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

plt.savefig("vis_data_ngram.png", dpi=300)
plt.show()
```

n-grams are useful in that they tell us exactly word distributions (once appropriately filtered). Word pairings are exceptionally useful in many contexts, not limited to sentiment. Reconstruction of broken or incomplete texts, for example, and auto-correct are applications of n-grams. 

![vis_data_ngram](https://github.com/deltaquebec/sentiment_imdb_reviews/blob/main/assets/vis_data_ngram.png)

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
plt.savefig("vis_data_char.png", dpi=300)
plt.show(block=True)  
```

![vis_data_char](https://github.com/deltaquebec/sentiment_imdb_reviews/blob/main/assets/vis_data_char.png)

## Number of words in text

While we have already defined an index of our dataframes for the number of words (index 3: train_data[3], train_data_pos[3], train_data_neg[3]), such an index does not refer to the cleaned data; rather, it tracks the number of words in the raw data. Indeed, we could append another set relative to the cleaned data; however, in this visualization, we simply extract the word count using train_data[11], train_data_pos[11], train_data_neg[11].

```
f = plt.figure(figsize=(12,8))

text_len=train_data[11].str.split().map(lambda x: len(x))
f.add_subplot(211)
plt.hist(text_len,color='blue')
plt.title("Total Reviews")
plt.xlabel("number of words")
plt.ylabel("number of reviews")

text_len=train_data_pos[11].str.split().map(lambda x: len(x))
f.add_subplot(223)
plt.hist(text_len,color='green')
plt.title("Text with Good Reviews")
plt.xlabel("number of words")
plt.ylabel("number of reviews")

text_len=train_data_neg[11].str.split().map(lambda x: len(x))
f.add_subplot(224)
plt.hist(text_len,color='red')
plt.title("Text with Bad Reviews")
plt.xlabel("number of words")
plt.ylabel("number of reviews")

f.suptitle("Words in texts")
plt.savefig("vis_data_words.png", dpi=300)
plt.show()
```

![vis_data_words](https://github.com/deltaquebec/sentiment_imdb_reviews/blob/main/assets/vis_data_words.png)

## Average word length represented as probability density

While we have already defined an index of our dataframes for average word length (index 3: train_data[7], train_data_pos[7], train_data_neg[7]), such an index does not refer to the cleaned data; rather, it tracks the number of words in the raw data. Indeed, we could append another set relative to the cleaned data; however, in this visualization, we simply extract the average word length using train_data[11], train_data_pos[11], train_data_neg[11].

```
f = plt.figure(figsize=(20,10))

word=train_data[11].str.split().apply(lambda x : [len(i) for i in x])
f.add_subplot(131)
sns.histplot(word.map(lambda x: np.mean(x)),stat='density',kde=True,color='blue')
plt.title("Total Reviews")
plt.xlabel("average word length")
plt.ylabel("probability density")

word=train_data_pos[11].str.split().apply(lambda x : [len(i) for i in x])
f.add_subplot(132)
sns.histplot(word.map(lambda x: np.mean(x)),stat='density',kde=True,color='green')
plt.title("Text with Good Reviews")
plt.xlabel("average word length")
plt.ylabel("probability density")

word=train_data_neg[11].str.split().apply(lambda x : [len(i) for i in x])
f.add_subplot(133)
sns.histplot(word.map(lambda x: np.mean(x)),stat='density',kde=True,color='red')
plt.title("Text with Bad Reviews")
plt.xlabel("average word length")
plt.ylabel("probability density")

f.suptitle("Average word length in texts")
plt.savefig("vis_data_leng.png", dpi=300)
plt.show()
```

The average word length is represented by a _probability density_, the values of which may be greater than 1; the distribution itself, however, will integrate to 1. The values of the y-axis, then, are useful for relative comparisons between categories. Converting to a probability (in which the bar heights sum to 1) in the code is simply a matter of changing the argument stat='density' to stat='probability', which is essentially equivalent to finding the area under the curve for a specific interval. See [this article](https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0) for more details.

![vis_data_leng](https://github.com/deltaquebec/sentiment_imdb_reviews/blob/main/assets/vis_data_leng.png)


# 2. Neural Network Models

The neural network models are contained in their own respective python files: _model_cnn.py_; _model_rnn.py_; _model_rcnn.py_; _model_lstm.py_. Throughout, results are reported in text files named appropriately for each model as: _results_cnn.txt_; _results_rnn.py_; _results_rcnn.py_; _results_lstm.py_. Visualizations of results follow as: _vis.png_ for each model and each model's metrics. The script _display.py_ graphs the training time for each model.

Neural networks in the project generally follow the same structure:

1. Acquire data
2. Split data into train and test sets
3. Clean the data
4. Tokenize and pad the data
5. Define validation sets
6. Build model
7. Train
8. Test
9. Analyze
10. Report
11. Visualization

Tasks 1 through 3 are identical to the data preparation and data cleaning from Visualization; they differ, however, in that the Visualization datasets followed for train_data, train_data_pos, and train_data_neg, whereas for the Neural Network Models, the datasets used are train_data and test_data. Each model trains on ten epochs (looped 1 through 10), and have equivalent input lengths (500), batch size (64), and embedding vector length (64). While this allows for a fairer cross-comparison, it does sacrifice optimization of each model. Optimizastion of individual models by tuning their hyperparameters is as much an art as a science, and is generally left for a return project. All accuracies reported here are >82%.

```
imdbdata = pd.concat(clean)
epochs = [1,2,3,4,5,6,7,8,9,10]

X_train=train_data[11]
X_test=test_data[11]
y_train=train_data[1]
y_test=test_data[1]
```

The data must be tokenized, in which we split text into individual words and turn them into a sequence of integers such that the model can process them. Padding refers to the amount of data added such that each sequence of data is the same length; to maintain uniformity, we want to pad the data. Finally, we define a validation set that is held back from training to be used to give an estimate of model skill while tuning hyperparameters. Batch size is the number of samples that will be propagated through the network.

```
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(imdbdata[11])
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

batch_size = 64
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
```

Finally, since we are working with words, we need to encode the data. Word embedding has been used to represent features of the words with semantic vectors and map each movie review into a real vector domain. This begins the model, and we add the first layer of the neural network, which is the embedding layer.

```
embedding_size=64

model = Sequential()

model.add(Embedding(vocabulary_size, embedding_size, input_length= max_words))
```

What remains is to build each model, train and test the model on our data, analyze each model's performance (we measure for accuracy, loss, F1, misclassification rate, and training time) across ten epochs, and report the results.

## CNN

Convolutional Neural Networks --- CNNs --- are multi-layered feed-forward neural networks. We load hidden layers one on top of the other in a sequence, which enables the CNN to observe and learn hierarchical features. Data convolution extracts feature variables; 1D convolutional layers help learn patterns at a specific position in a sentence which are used to recognize patters elsewhere throughout. Pooling is applied to each patch of feature map to extract particular (maximum) values, which reduces inputs to the next layer. The final layer has units 1, which is equal to the number of outputs.

```
model.add(Dropout(0.5))
model.add(Conv1D(filters = 128, kernel_size = 3, strides= 1, padding='same', activation= 'relu'))
model.add(GlobalMaxPooling1D())

model.add(Dense(units = 512, activation= 'relu', kernel_initializer= 'TruncatedNormal'))
model.add(Dropout(0.5))
model.add(Dense(units = 512, activation= 'relu', kernel_initializer= 'TruncatedNormal'))
model.add(Dropout(0.5))

model.add(Dense(1, activation= 'sigmoid'))
```

Before trianing, we compile model, which gives specifications about the model. We specify: _error_ (loss) to minimize over epochs as binary crossentropy ([see more](https://keras.io/api/losses/)); _optimization_ method as adam ([see more](https://keras.io/api/optimizers/)); list of _metrics_ in which evaluation needs to be reported as accuracy ([see more](https://keras.io/api/metrics/)).

```
model.compile(loss='binary_crossentropy',optimizer= 'adam',metrics=['accuracy'])
```

We can get the summary of the model, which we report in the resulting text file.

```
def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

model_summary_string = get_model_summary(model)
with open("results_cnn.txt", "a+") as h:
        print(model_summary_string,file=h)
```

To avoid overfitting, we specify our stopping condition over an arbitrary number of training epochs and stop training once the model performance stops improving on a hold out validation dataset. We specift such stopping to monitor loss on the valdiation set.

```
earlystopper = EarlyStopping(monitor='val_loss', patience= 2, verbose=1)
```

Since we are interested in loss, accuracy, F1 scores, misclassification rate, and training time, we save empty arrays such that those values may be accessed across all epochs. We loop through ten values for epoch 1 through 10 and train our model by using the "fit" method. We want to see how our model works on individual predictions on test examples, so we ask it to predict the sentiment, from which we can get confusion matrix values. Results are reported for each epoch.

```
loss = []
acc = []
f1one = []
f1onemic = []
misr = []
time = []

for i in epochs:

    start = datetime.now()
    model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid),
              batch_size=batch_size, epochs=i, callbacks= [earlystopper])
    end = datetime.now()
    delta=(end-start).total_seconds()
    delta_r=round(delta,2)
    
    scores = model.evaluate(X_test, y_test, verbose=0)
    l=scores[0]
    l_r=round(scores[0],2)
    a=scores[1]
    a_r=round(scores[1],2)
    
    loss.append(l_r)
    acc.append(a_r)
    time.append(delta_r)

    y_pred = model.predict_classes(np.array(X_test))

    target_names = ["pos", "neg"]
    cm = confusion_matrix(y_test, y_pred)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels={"negative": 0, "positive": 1})
    #disp.plot()

    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    pres = TP/(TP+FP)
    reca = TP/(TP+FN)
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    ce_r=round(classification_error,2)

    misr.append(ce_r)

    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_macro_r=round(f1_macro,2)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_micro_r=round(f1_micro,2)

    f1one.append(f1_macro_r)
    f1onemic.append(f1_micro_r)

    with open("results_cnn.txt", "a+") as h:
        print("Number of epochs:",i,file=h)
        print("Test loss:", l_r,file=h)
        print("Test accuracy:", a_r,file=h)
        print("F1 (Macro): ",f1_macro_r,file=h)
        print("F1 (Micro): ",f1_micro_r,file=h)
        print("Misclassification rate: ", ce_r,file=h)
        print("Training time: ", delta_r,file=h)
        #print(classification_report(y_test, y_pred, target_names=target_names),file=h)
        #print(cm,file=h)
        print("\n",file=h)
```

The scores for loss, accuracy, F1, misclassification rate, and training time are averaged and reported. This information is saved to a dataframe such that we may visualize the results graphically.

```
aloss = round(mean(loss),2)
average = round(mean(acc),2)
f1avg = round(mean(f1one),2)
f1avgmic = round(mean(f1onemic),2)
misravg = round(mean(misr),2)
timeavg = round(mean(time),2)

loss.append(aloss)
acc.append(average)
f1one.append(f1avg)
f1onemic.append(f1avgmic)
misr.append(misravg)
time.append(timeavg)

with open("results_cnn.txt", "a+") as h:
    print("Average loss of all epochs: ",aloss,file=h)
    print("Average accuracy of all epochs: ",average,file=h)
    print("Average F1 (Macro) of all epochs: ",f1avg,file=h)
    print("Average F1 (Micro) of all epochs: ",f1avgmic,file=h)
    print("Average misclassification rate of all epochs: ",misravg,file=h)
    print("Average training time (s) of all epochs: ",timeavg,file=h)
    print("Array of loss: ",loss,file=h)
    print("Array of accuracy: ",acc,file=h)
    print("Array of f1 macro: ",f1one,file=h)
    print("Array of f1 micro: ",f1onemic,file=h)
    print("Array of missclassification rate: ",misr,file=h)
    print("Array of time: ",time,file=h)
    
array = np.array([loss,acc,misr,time])
index_values = ['loss','acc','miss','t']
column_values = ['ep1','ep2','ep3','ep4','ep5','ep6','ep7','ep8','ep9','ep10','avg']
df = pd.DataFrame(data=array, index = index_values, columns = column_values)
dr_tr=df.transpose()
```

Results of the CNN model across ten epochs and averaged scores are reported below.

|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Avg |
|-------|--------|---------|--------|---------|--------|---------|--------|---------|--------|---------|---------|
| Loss                      | 0.39 | 0.37 | 0.38 | 0.41 | 0.47 | 0.58 | 0.57 | 0.64 | 0.56 | 0.59 | 0.5 |
| Accuracy                  | 0.82 | 0.84 | 0.83 | 0.83 | 0.83 | 0.83 | 0.82 | 0.82 | 0.82 | 0.82 | 0.83 |
| F1<sub>macro</sub>        | 0.82 | 0.84 | 0.83 | 0.83 | 0.83 | 0.83 | 0.82 | 0.82 | 0.82 | 0.82 | 0.83 |
| F1<sub>micro</sub>        | 0.82 | 0.84 | 0.83 | 0.83 | 0.83 | 0.83 | 0.82 | 0.82 | 0.82 | 0.82 | 0.83 |
| Misclassification rate    | 0.18 | 0.16 | 0.17 | 0.17 | 0.17 | 0.17 | 0.18 | 0.18 | 0.18 | 0.18 | 0.17 |
| Training time (s)         | 7.93 | 8.68 | 12.92 | 17.24 | 12.89 | 25.67 | 17.21 | 17.27 | 12.92 | 12.86 | 14.56 |

The results suggest that a model at two epochs has minimized loss. An ideal model will minimize validation loss, misclassification rate, and training time while maximizing accuracy and F1 scores. All other metrics relatively equal across epochs, this particular architecture favors two epochs. 

We see this in the visualizations. Three graphs result from the code that report training time per epoch, metrics (loss, accuracy, and missclassification rate) per epoch without average values, and metrics with average values; for brevity, we report only the metrics graph with averages as a bar graph. 

![vis_cnn_metrics_avec_avg](https://github.com/deltaquebec/sentiment_imdb_reviews/blob/main/assets/vis_cnn_metrics_avec_avg.png)

We see that the CNN model performs best at two epochs.

## RNN

Recurrent Neural Networks --- RNNs --- are multi-layered networks with feedback loops, where some units' current outputs determine some future network inputs. RNNs are useful especuially with temporal inputs (e.g. speech signals) or sequences  (e.g. words of a sentence). We add an LSTM layer as a hidden layer in the RNN such that past information can be retained. Technically, RNNs that use LSTM cells are called LSTM networks, which we investigate in the project. Here, however, the RNN simply uses an LSTM layer while the LSTM model only has multiple LSTM layers (more on this later). The model follows as below.

```
model.add(Embedding(vocabulary_size, embedding_size, input_length= max_words))
model.add(Dropout(0.7))

model.add(LSTM(100))
model.add(Dropout(0.7))

model.add(Dense(1, activation= 'sigmoid'))

model.compile(loss='binary_crossentropy',optimizer= 'adam',metrics=['accuracy'])
```

The remainder of the code is identical to the CNN, but updated to reflect the RNN model. Results of the RNN model across ten epochs and averaged scores are reported below.

|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Avg |
|-------|--------|---------|--------|---------|--------|---------|--------|---------|--------|---------|---------|
| Loss                      | 0.38 | 0.39 | 0.38 | 0.42 | 0.43 | 0.49 | 0.52 | 0.58 | 0.56 | 0.65 | 0.48 |
| Accuracy                  | 0.84 | 0.83 | 0.84 | 0.83 | 0.83 | 0.83 | 0.83 | 0.82 | 0.82 | 0.82 | 0.83 |
| F1<sub>macro</sub>        | 0.84 | 0.83 | 0.84 | 0.83 | 0.83 | 0.83 | 0.83 | 0.82 | 0.82 | 0.82 | 0.83 |
| F1<sub>micro</sub>        | 0.84 | 0.83 | 0.84 | 0.83 | 0.83 | 0.83 | 0.83 | 0.82 | 0.82 | 0.82 | 0.83 |
| Misclassification rate    | 0.16 | 0.17 | 0.16 | 0.17 | 0.17 | 0.17 | 0.17 | 0.18 | 0.18 | 0.18 | 0.17 |
| Training time (s)         | 13.31 | 19.95 | 29.74 | 29.84 | 39.64 | 39.71 | 70.1 | 30.24 | 40.35 | 50.05 | 36.29 |

The results suggest that a model at either one or three epochs has minimized loss. An ideal model will minimize validation loss, misclassification rate, and training time while maximizing accuracy and F1 scores. All other metrics relatively equal across epochs, this particular architecture favors one or three epochs.

We see this in the visualizations. Three graphs result from the code that report training time per epoch, metrics (loss, accuracy, and missclassification rate) per epoch without average values, and metrics with average values; for brevity, we report only the metrics graph with averages as a bar graph. 

![vis_rnn_metrics_avec_avg](https://github.com/deltaquebec/sentiment_imdb_reviews/blob/main/assets/vis_rnn_metrics_avec_avg.png)

We see that the RNN model performs best at one or two epochs.

## RCNN

Recurrent Convolutional Neural Networks --- RCNNs --- follow from [Lai et al. (2015)](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf), in which the authors's goal is to combine the CNN and RNN architectures such that the CNN accounts for semantics of textual tokens and the RNN accounds for contextual and longer distance information. This follows as the inclusion of multiple LSTM layers following the convolutional layer (with dropouts).

```
model.add(Embedding(vocabulary_size, embedding_size, input_length= max_words))
model.add(Dropout(0.7))

model.add(Conv1D(filters = 256, kernel_size = 3, strides= 2, padding='same', activation= 'relu'))
#model.add(GlobalMaxPooling1D())

model.add(LSTM(128, dropout=0.7, recurrent_dropout=0.0,return_sequences=True))
model.add(LSTM(128, dropout=0.7, recurrent_dropout=0.0,return_sequences=True))
model.add(LSTM(128, dropout=0.7, recurrent_dropout=0.0))

model.add(Dense(1, activation= 'sigmoid'))

model.compile(loss='binary_crossentropy',optimizer= 'adam',metrics=['accuracy'])
```

The remainder of the code is identical to the CNN and RNN, but updated to reflect the RCNN model. Results of the RCNN model across ten epochs and averaged scores are reported below.


|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Avg |
|-------|--------|---------|--------|---------|--------|---------|--------|---------|--------|---------|---------|
| Loss                      | 0.39 | 0.36 | 0.38 | 0.42 | 0.47 | 0.54 | 0.55 | 0.64 | 0.6 | 0.62 | 0.5 |
| Accuracy                  | 0.83 | 0.85 | 0.84 | 0.83 | 0.83 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 0.83 |
| F1<sub>macro</sub>        | 0.83 | 0.85 | 0.84 | 0.83 | 0.83 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 0.83 |
| F1<sub>micro</sub>        | 0.83 | 0.85 | 0.84 | 0.83 | 0.83 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 0.83 |
| Misclassification rate    | 0.17 | 0.15 | 0.16 | 0.17 | 0.17 | 0.18 | 0.18 | 0.18 | 0.18 | 0.18 | 0.17 |
| Training time (s)         | 31.2 | 43.62 | 60.73 | 62.0 | 103.06 | 122.37 | 82.54 | 103.41 | 163.71 | 102.22 | 87.49 |

The results suggest that a model at either two epochs has minimized loss. An ideal model will minimize validation loss, misclassification rate, and training time while maximizing accuracy and F1 scores. All other metrics relatively equal across epochs, this particular architecture favors two epochs.

We see this in the visualizations. Three graphs result from the code that report training time per epoch, metrics (loss, accuracy, and missclassification rate) per epoch without average values, and metrics with average values; for brevity, we report only the metrics graph with averages as a bar graph. 

![vis_rcnn_metrics_avec_avg](https://github.com/deltaquebec/sentiment_imdb_reviews/blob/main/assets/vis_rcnn_metrics_avec_avg.png)

We see that the RCNN model performs best at two epochs.

## LSTM

Depending on the context, the term "LSTM" --- Long Short-Term Memory --- alone can refer to (see [here](https://ai.stackexchange.com/questions/18198/what-is-the-difference-between-lstm-and-rnn)):

1. LSTM unit (or neuron)
2. LSTM layer (many LSTM units)
3. LSTM neural network (a neural network with LSTM units or layers)

This model follows from the third item listed above. RNN models enjoy sequences (sentences and words), and so are better suited for those kinds of tasks. RNNs like sequential data because they are feedback models. LSTMs were introduced to solve the [vanishing gradient problem](https://ai.stackexchange.com/questions/18198/what-is-the-difference-between-lstm-and-rnn) (among other things) of RNNS, and compared to RNNs are computationally effective (though temporally cumbersome). The purpose of the LSTM is to retain information over longer periods of time.

To force the LSTM model to run on the GPU via CUDA, we need cuDNN, which has some requirements that must be met. 

- activation == tanh
- recurrent_activation == sigmoid
- recurrent_dropout == 0
- unroll is False
- use_bias is True
- Inputs, if use masking, are strictly right-padded
- Eager execution is enabled in the outermost context

Our model, then, follows as:

```
model.add(Embedding(vocabulary_size, embedding_size, input_length= max_words))
model.add(Dropout(0.7))

model.add(LSTM(128, dropout=0.7, recurrent_dropout=0.0,return_sequences=True))
model.add(LSTM(128, dropout=0.7, recurrent_dropout=0.0,return_sequences=True))
model.add(LSTM(128, dropout=0.7, recurrent_dropout=0.0))

model.add(Dense(1, activation= 'sigmoid'))

model.compile(loss='binary_crossentropy',optimizer= 'adam',metrics=['accuracy'])
```

The remainder of the code is identical to the CNN, RNN, and RCNN but updated to reflect the LSTM model. Results of the LSTM model across ten epochs and averaged scores are reported below.

|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Avg |
|-------|--------|---------|--------|---------|--------|---------|--------|---------|--------|---------|---------|
| Loss                      | 0.44 | 0.37 | 0.35 | 0.35 | 0.37 | 0.36 | 0.39 | 0.36 | 0.38 | 0.38 | 0.38 |
| Accuracy                  | 0.82 | 0.84 | 0.85 | 0.85 | 0.84 | 0.85 | 0.84 | 0.85 | 0.84 | 0.84 | 0.84 |
| F1<sub>macro</sub>        | 0.82 | 0.84 | 0.85 | 0.85 | 0.84 | 0.85 | 0.84 | 0.85 | 0.84 | 0.84 | 0.84 |
| F1<sub>micro</sub>        | 0.82 | 0.84 | 0.85 | 0.85 | 0.84 | 0.85 | 0.84 | 0.85 | 0.84 | 0.84 | 0.84 |
| Misclassification rate    | 0.18 | 0.16 | 0.15 | 0.15 | 0.16 | 0.15 | 0.16 | 0.15 | 0.16 | 0.16 | 0.16 |
| Training time (s)         | 35.31 | 59.82 | 88.8 | 118.68 | 148.61 | 150.03 | 90.38 | 121.04 | 119.72 | 210.68 | 114.31 |

The results suggest that a model at either three or four epochs has minimized loss. An ideal model will minimize validation loss, misclassification rate, and training time while maximizing accuracy and F1 scores. All other metrics relatively equal across epochs, this particular architecture favors either three or four epochs.

We see this in the visualizations. Three graphs result from the code that report training time per epoch, metrics (loss, accuracy, and missclassification rate) per epoch without average values, and metrics with average values; for brevity, we report only the metrics graph with averages as a bar graph. 

![vis_lstm_metrics_avec_avg](https://github.com/deltaquebec/sentiment_imdb_reviews/blob/main/assets/vis_lstm_metrics_avec_avg.png)

We see that the LSTM model performs best at three or four epochs.

Interestingly, even when forcing the model onto the GPU, the LSTM is markedly temporially expensive. This expense follows from the sequential computation of the LSTM layers. We can compare the training times of each model graphically via a quick script _display.py_ that is hardcoded for comparing training times.

![vis_time](https://github.com/deltaquebec/sentiment_imdb_reviews/blob/main/assets/vis_time.png)

Ultimately, however, deep learning is slow, even if we optimize processing units and model architectures. 

Learning rate is related as the network adapts to new information. Consider the following (original citation of this example is unknown; it is adapted here from a class lecture): suppose a scientist from Mars encounters examples of cats on Earth. Each cat n in the example set N of cats has orange fur. It stands to reason, then, that the Martian scientist will think that cats have orange fur. When trying to identify whether an animal is or is not a cat, our scientist will look for orange fur.

Suppose now that our scientist encounters a black cat, and the representatives of Earth tell the bewildered Martian scientist that that animal is indeed also a cat. This is an example of supervised learning: the Martian scientist is provided with data (a black animal) that is appropriately identified as a cat. Learning rate follows (where "large" and "small" are relative to each other, and their values will depend on the problem being solved):

- Large learning rate indicates that the Martian scientist will realize that “orange fur” is not the most important or useful feature of cats
- Small learning rate indicates that the Martian scientist will simply think that this black cat is an outlier and that cats are still definitively orange

As the learning rate increases, the network will "change its mind". If it is too high, however, our Martian scientist will think that all cats are black even though more were orange than were black; in other words, our scientist went too far the other way. We want to find a reasonable balance in which the learning rate is small enough that we converge to something useful, but large enough that we do not have to spend unreasonable time training it.
