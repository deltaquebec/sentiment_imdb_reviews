import csv
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import string
import re
import io
import time
from datetime import datetime
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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, LSTM
from statistics import mean
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, f1_score
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
with open("train.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    for i in range(0, len(X_train)):
        label = y_train[i]
        review = " ".join([indxword[o] for o in X_train[i]])
        writer.writerow([review, label])

# move testing data to csv
with open("test.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    for i in range(0, len(X_test)):
        label = y_test[i]
        review = " ".join([indxword[o] for o in X_test[i]])
        writer.writerow([review, label])

#########################################################################
# move data from csv to pandas dataframes 
#########################################################################
# save training and test data as respective dataframe
train_data = pd.read_csv("train.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)

data = [train_data,test_data]
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

# cleaned train_data and test_data
imdbdata = pd.concat(clean)

#########################################################################
# begin meat and potatoes: buidl model with tokenized text
#########################################################################

# define number of epochs
epochs = [1,2,3,4,5,6,7,8,9,10]

# assign training and test data to appropriately cleaned daraframe
X_train=train_data[11]
X_test=test_data[11]
y_train=train_data[1]
y_test=test_data[1]

# start txt file to store results; print size of training and test data; print max and min reviews
with open("results_rcnn.txt", "w+") as h:
    print("Results of RCNN with cleaned data",file=h)
    print("Loaded dataset with {} training samples, {} test samples".format(len(X_train), len(X_test)),file=h)
    print("\n",file=h)

# tokenize
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(imdbdata[11])
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# specify length of inputs such that all inpuits are the same length
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# specify batch size as 64
batch_size = 64
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]

# specify embedding vector length as 64
embedding_size=64

# begin model
model = Sequential()

# Embedding layer
model.add(Embedding(vocabulary_size, embedding_size, input_length= max_words))
model.add(Dropout(0.7))

# Convolutional Layer(s)
model.add(Conv1D(filters = 256, kernel_size = 3, strides= 2, padding='same', activation= 'relu'))
#model.add(GlobalMaxPooling1D())

# LSTM layer
model.add(LSTM(128, dropout=0.7, recurrent_dropout=0.0,return_sequences=True))
model.add(LSTM(128, dropout=0.7, recurrent_dropout=0.0,return_sequences=True))
model.add(LSTM(128, dropout=0.7, recurrent_dropout=0.0))

# Output layer
model.add(Dense(1, activation= 'sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy',optimizer= 'adam',metrics=['accuracy'])

# document the model summary
def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

model_summary_string = get_model_summary(model)
with open("results_rcnn.txt", "a+") as h:
        print(model_summary_string,file=h)

#########################################################################
# define stopping for epochs
#########################################################################

# specify early stopping criteria as val_loss
earlystopper = EarlyStopping(monitor='val_loss', patience= 2, verbose=1)

# save empty arrays for average loss, accuracy, f1, misclassification rate, and training time across all epochs
loss = []
acc = []
f1one = []
f1onemic = []
misr = []
time = []

#########################################################################
# loop through each epoch
#########################################################################

for i in epochs:

    # create RCNN model with parameters specified above; get training time
    start = datetime.now()
    model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid),
              batch_size=batch_size, epochs=i, callbacks= [earlystopper])
    end = datetime.now()
    delta=(end-start).total_seconds()
    delta_r=round(delta,2)

    # RCNN model evaluated on test sets for loss, accuracy. training time
    scores = model.evaluate(X_test, y_test, verbose=0)
    l=scores[0]
    l_r=round(scores[0],2)
    a=scores[1]
    a_r=round(scores[1],2)

    loss.append(l_r)
    acc.append(a_r)
    time.append(delta_r)

    # test model
    y_pred = model.predict_classes(np.array(X_test))

    # prepare confusion matrices
    target_names = ["pos", "neg"]
    cm = confusion_matrix(y_test, y_pred)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels={"negative": 0, "positive": 1})
    #disp.plot()

    # get values of confusion matrix
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    # get presision, recall, misclassification rate
    pres = TP/(TP+FP)
    reca = TP/(TP+FN)
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    ce_r=round(classification_error,2)

    misr.append(ce_r)

    # f1 scores
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_macro_r=round(f1_macro,2)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_micro_r=round(f1_micro,2)

    f1one.append(f1_macro_r)
    f1onemic.append(f1_micro_r)

    # report metrics
    with open("results_rcnn.txt", "a+") as h:
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

#########################################################################
# report average metrics and prepare visualization
#########################################################################

# metric averages
aloss = round(mean(loss),2)
average = round(mean(acc),2)
f1avg = round(mean(f1one),2)
f1avgmic = round(mean(f1onemic),2)
misravg = round(mean(misr),2)
timeavg = round(mean(time),2)

# append averages to metric arrays
loss.append(aloss)
acc.append(average)
f1one.append(f1avg)
f1onemic.append(f1avgmic)
misr.append(misravg)
time.append(timeavg)

# report results
with open("results_rcnn.txt", "a+") as h:
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

# prepare dataframe for graphical represenation of results
array = np.array([loss,acc,misr,time])
index_values = ['loss','acc','miss','t']
column_values = ['ep1','ep2','ep3','ep4','ep5','ep6','ep7','ep8','ep9','ep10','avg']
df = pd.DataFrame(data=array, index = index_values, columns = column_values)
dr_tr=df.transpose()

#########################################################################
# visualization
#########################################################################

# plot time per epoch (sans average in linegraph)
def time_graph():
    ax=sns.lineplot(data = dr_tr.iloc[0:10]['t'], marker='o').set(xlabel ="epoch", ylabel = "time (s)",title="RCNN time per epoch")
    plt.legend(labels=["RCNN",],loc = 'lower right')
    plt.savefig("vis_rcnn_time.png", dpi=300)
    plt.show()
    return ax
time_graph()

# plot metrics for model per epoch as line graph sans average
def metrics_line():
    ax=sns.lineplot(data = dr_tr.iloc[0:10][['loss','acc','miss']], markers=True).set(xlabel ="epoch", ylabel = "metric value",title="RCNN metrics per epoch")
    plt.legend(labels=["loss","acc","miss"],loc = 'right')
    plt.savefig("vis_rcnn_metrics_sans_avg.png", dpi=300)
    plt.show()
    return ax
metrics_line()

# plot metrics for model per epoch as bar graph avec average
def metrics_bar():
    ax = dr_tr[['loss','acc','miss']].plot.bar(rot=0,title="RCNN metrics per epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric value")
    for i in ax.patches:
        ax.annotate(format(i.get_height(), '.2f'),
                   (i.get_x() + i.get_width() / 2,
                    i.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')
    plt.legend(loc = 'lower right')
    plt.savefig("vis_rcnn_metrics_avec_avg.png", dpi=300)
    plt.show()
    return ax
metrics_bar()
