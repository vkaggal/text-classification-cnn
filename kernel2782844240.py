#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Uncomment and run the commands below if imports fail
#!conda install numpy pandas pytorch torchvision cpuonly -c pytorch -y
#!pip install matplotlib --upgrade --quiet
#!pip install torch
#!pip install torchtext


# In[ ]:


# https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e'
# here is an example of sentiment analysis - https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

# text related
from torchtext import data

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


project_name = '05-course-project-text-classification'


# ## Let's inspect the data
# 
# > You can see in the dataset has accompanied code that can help load the data into a dataframe. We use that code snippet to load the initial 1000 rows and inspect data to get an intuition.
# 
# It turns out that there may be latin characters that requires to use `encoding` parameter when reading the csv as you see below.

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
df1 = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv', delimiter=',', nrows = None, encoding='latin-1', names=["target", "id", "date", "flag", "user", "text"])
df1.dataframeName = 'training.1600000.processed.noemoticon.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df1.head(10)


# ### Let's ensure that we have the right classes that have been [documented here](https://www.kaggle.com/kazanova/sentiment140) *0 = negative, 4 = positive* 

# In[ ]:


df1.columns


# Using the code from here, let inspect the distrubution of columns that have unique values between 1 and 50. The plot below shows the target or column-0 from the input data.

# In[ ]:


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# In[ ]:


plotPerColumnDistribution(df1, 10, 5)


# Next, we need to create PyTorch datasets & data loaders for training & validation. We'll start by creating a TensorDataset.

# In[ ]:


# We are not interested in anything other than the text and the target columns, let's
df_input = df1[['target','text']]


# In[ ]:


len(df1[['target','text']])


# In[ ]:


print(df_input.shape, df_input.columns)


# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
def remove_stopwords(text):
    word_tokens = word_tokenize(text) 
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]           
    return ' '.join(filtered_sentence)


# In[ ]:


# use nltk to remove stop words, remove, newlines, extra spaces, and punctuations
from nltk.tokenize import RegexpTokenizer

def preprocess_text(texts):
    return_text = []
    tokenizer = RegexpTokenizer(r'\w+')
    text = remove_stopwords(texts)
    text = tokenizer.tokenize(text)
    return_text = text
    print('return_text:', return_text, type(return_text))
    return ' '.join(return_text)


# In[ ]:


size = 100
for idx, t in df1.iterrows():
    #print(preprocess_text(t['text']),";", type(preprocess_text(t['text'])))
    if len(t['text'].split()) > size:
        size = len(t['text'].split())
print(size)


# ### Preprocess data and prepare tensors
# Let's define instances of `Field` for our `text` and `target` that can contain the Vocabolary and their corresponding numeric representations as [detailed here](https://torchtext.readthedocs.io/en/latest/data.html#field). We will use the our function `preprocess_text` to remove stop words, trim spaces, remove special characters. We perform all these steps on the text field that contain the text of the tweets
# 
# 
# How do we split the data, the options include subclassing the Dataset class or using the methods detailed here https://torchtext.readthedocs.io/en/latest/datasets.html
# 

# In[ ]:


# given the 280 chars limit on tweets and based on the average size of the data, we can set max_document_length to be 100
max_document_length = 100
Text = data.Field(preprocessing=preprocess_text, tokenize="spacy", batch_first=True, include_lengths=True, fix_length=max_document_length)
Target = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)


# In[ ]:


# let's associate the fields in the dataset to the definitions above - `Text` and `Target`
#fields = [('text', Text), ('target', Target)]
fields = [('text', Text)]


# **Split the dataset**
# 
#  sentiment140 dataset has 1600000 samples, we need to split that dataset into tran and validation sets. ~~We will be using `random_split` from the package `torch.utils.data` to accomplish this~~ I am not sure why this results in an *TypeError: expected string or bytes-like object* error.
#  
# 

# In[ ]:


# from torchtext import data
# some_train_dataset = data.TabularDataset.splits(
#     path='../input/sentiment140',
#     train='training.1600000.processed.noemoticon.csv',
#     format='csv',
#     fields=fields,
#     skip_header=False
# )


# In[ ]:


# Code copied from https://averdones.github.io/reading-tabular-data-with-pytorch-and-training-a-multilayer-perceptron/

from torch.utils.data import Dataset

class TwitterSentimentDataset(Dataset):
    """Twitter sentiment dataset."""

    def __init__(self, csv_file):
        """Initializes instance of class TwitterSentimentDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        df = pd.read_csv(csv_file, delimiter=',', nrows = None, encoding='latin-1', names=["target", "text"])

        # Grouping variable names
        self.categorical = ['text'] # dont think we need all these columns - [ 'id', 'date', 'flag', 'user', 'text']
        self.target = "target"

        # One-hot encoding of categorical variables
        self.tweet_frame = pd.get_dummies(df['text'], prefix=self.categorical)

        # Save target and predictors
        self.X = self.tweet_frame.drop(self.target, axis=1)
        self.y = self.tweet_frame[self.target]

    def __len__(self):
        return len(self.tweet_frame)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]


# In[ ]:


# Load dataset
dataset = TwitterSentimentDataset('../input/sentiment140/training.1600000.processed.noemoticon.csv')

# Split into training and test
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
trainset, valset = random_split(dataset, [train_size, val_size])


# In[ ]:


# make splits for data
train, val = datasets.IMDB.splits(Text, Target)

# build the vocabulary
Text.build_vocab(train, vectors=GloVe(name='6B', dim=300))

# make iterator for splits
train_iter, test_iter = data.BucketIterator.splits(
    (train, val), batch_size=3, device=0)


# In[ ]:


print(len(some_train_data))


# In[ ]:


# val_percent = 0.15 # between 01 and 0.2
# val_size = int(val_percent * len(df1[['target','text']]))
# train_size = len(df1[['target','text']]) - val_size
# train_data, val_data = torch.utils.data.random_split(df1[['text']], [train_size, val_size])



seed = 46
train_data, val_data = some_train_data.split(split_ratio=0.8, random_state=random.seed(seed))


# In[ ]:


print(type(train_data))


# In[ ]:


print(len(train_data), len(val_data))


# In[ ]:


# https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets

seed = 42
train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(seed))


# In[ ]:


print(list(torch.utils.data.DataLoader(trainloader_1, num_workers=2)))


# In[ ]:


# Python program to generate WordCloud 
  
# importing all necessery modules 
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
  
def make_word_cloud(df):  
    comment_words = '' 
    stopwords = set(STOPWORDS) 
  
    # iterate through the csv file 
    for val in df.CONTENT: 
      
        # typecaste each val to string 
        val = str(val) 
  
        # split the value 
        tokens = val.split() 
      
        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 
      
        comment_words += " ".join(tokens)+" "
  
        wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
  
    plt.show() 


# ### Build the vocabulary
# 
# We have the train, validation data sets. We also have out "containers" or objects defined to hold the vocabolary for Text. We can use the build_vocab on the Text field to build a vocab object using one or more datasets. We can find [more details here](https://torchtext.readthedocs.io/en/latest/data.html#field)

# In[ ]:


print(type(train_dataset))


# In[ ]:


Text.build_vocab(train_data, val_data, max_size=5000)
Label.build_vocab(train_data)
vocab_size = len(Text.vocab)


# In[ ]:


#dataset = TensorDataset(inputs, targets)

#val_percent = 0.15 # between 0.1 and 0.2
#val_size = int(num_rows * val_percent)
#train_size = num_rows - val_size

#print(train_size, val_size, len(dataset))
#train_ds, val_ds = random_split(dataset, [train_size, val_size]) # Use the random_split function to split dataset into 2 parts of the desired length


# # This implementation can be split up into the following high level steps
# 
# * Preprocess data to remove unwanted characters and tokenize
# * Process input data to build vocabolary and load Glove Embeddings
# * build the model
#     - train 
#     - test
# 

# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


import jovian
jovian.commit(project=project_name, environment=None)

