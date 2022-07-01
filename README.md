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

## Wordcloud representations
## n-gram (mono-,bi-,tri-gram)
## Number of characters in text
## Number of words in text
## Average word length represented as probability density

# 2. Neural Network Models
## CNN
## RNN
## RCNN
## LSTM
