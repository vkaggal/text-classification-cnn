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


from gensim.parsing.preprocessing import remove_stopwords

text = "this is a test sentence, I like cricket and not other sports."
ret_val = []
ret_val.append(remove_stopwords(text))
print(ret_val)


# In[ ]:


import nltk
# from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
def my_remove_stopwords(text):
    word_tokens = word_tokenize(text) 
#     stop_words = set(stopwords.words('english'))
#     filtered_sentence = [w for w in word_tokens if not w in stop_words]  
    filtered_sentence = remove_stopwords(" ".join(word_tokens))
    return filtered_sentence


# In[ ]:


# use nltk to remove stop words, remove, newlines, extra spaces, and punctuations
from nltk.tokenize import RegexpTokenizer
from gensim.parsing.preprocessing import remove_stopwords

def preprocess_text(texts):
    return_text = []
#     tokenizer = RegexpTokenizer(r'\w+')
    #preprocess_pipeline = data.Pipeline(lambda x: x.decode('latin1').encode('utf8'))
    for i, text in enumerate(texts):
        #remove non ascii chars
        #text = remove_stopwords(text)
        #text = tokenizer.tokenize(text)
        
        return_text.append(remove_stopwords(text))
    return_text_str = " ".join(str(x) for x in return_text)
    # print('return_text:', return_text, type(return_text), return_text_str, type(return_text_str))

    return return_text_str


# In[ ]:


# Note: now that we know 100 is a good size, we dont need to execute this anymore
# size = 100
# for idx, t in df1.iterrows():
#     #print(preprocess_text(t['text']),";", type(preprocess_text(t['text'])))
#     if len(t['text'].split()) > size:
#         size = len(t['text'].split())
# print(size)


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
fields = [('target', Target),('id', None), ('date', None), ('flag', None), ('user', None) ,('text', Text)]


# In[ ]:


# import torch
# print(torch.__version__)


# **Split the dataset**
# 
#  sentiment140 dataset has 1600000 samples, we need to split that dataset into tran and validation sets. ~~We will be using `random_split` from the package `torch.utils.data` to accomplish this~~ I am not sure why this results in an *TypeError: expected string or bytes-like object* error.
#  
# 

# In[ ]:


# Note: dont want to spend more time on how to write to output, maybe later!
# infile = '../input/sentiment140/training.1600000.processed.noemoticon.csv'
# outfile = '../output/training.1600000.processed.noemoticon.vinod.csv'
# BLOCKSIZE = 1024*1024
# with open(infile, 'rb') as inf:
#     with open(outfile, 'w') as ouf:
#         while True:
#             data = inf.read(BLOCKSIZE)
#             if not data: break
#             converted = data.decode('latin-1').encode('utf-8')
#             ouf.write(converted)


# In[ ]:


# Extend TabularDataset so we can set the encoding to latin-1; based on [source here](https://pytorch.org/text/_modules/torchtext/data/dataset.html#TabularDataset)

from  torchtext.data.dataset import Dataset
from torchtext.data import Example
from torchtext.utils import download_from_url, unicode_csv_reader

import io
import os

class MyTabularDataset(Dataset):
    def __init__(self, path, format, fields, skip_header=False,
                 csv_reader_params={}, **kwargs):
        format = format.lower()
        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromCSV, 'csv': Example.fromCSV}[format]

        with io.open(os.path.expanduser(path), encoding="latin-1") as f:
            if format == 'csv':
                reader = unicode_csv_reader(f, **csv_reader_params)
            elif format == 'tsv':
                reader = unicode_csv_reader(f, delimiter='\t', **csv_reader_params)
            else:
                reader = f

            if format in ['csv', 'tsv'] and isinstance(fields, dict):
                if skip_header:
                    raise ValueError('When using a dict to specify fields with a {} file,'
                                     'skip_header must be False and'
                                     'the file must have a header.'.format(format))
                header = next(reader)
                field_to_index = {f: header.index(f) for f in fields.keys()}
                make_example = partial(make_example, field_to_index=field_to_index)

            if skip_header:
                next(reader)

            examples = [make_example(line, fields) for line in reader]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(MyTabularDataset, self).__init__(examples, fields, **kwargs)
        #super(MyTabularDataset, self).__init__(self, path, format, fields, **kwargs)
        
    def check_split_ratio(split_ratio):
        valid_ratio = 0.
        if isinstance(split_ratio, float):
            # Only the train set relative ratio is provided
            # Assert in bounds, validation size is zero
            assert 0. < split_ratio < 1., (
                "Split ratio {} not between 0 and 1".format(split_ratio))

            test_ratio = 1. - split_ratio
            return (split_ratio, test_ratio, valid_ratio)
        elif isinstance(split_ratio, list):
            # A list of relative ratios is provided
            length = len(split_ratio)
            assert length == 2 or length == 3, (
                "Length of split ratio list should be 2 or 3, got {}".format(split_ratio))

            # Normalize if necessary
            ratio_sum = sum(split_ratio)
            if not ratio_sum == 1.:
                split_ratio = [float(ratio) / ratio_sum for ratio in split_ratio]

            if length == 2:
                return tuple(split_ratio + [valid_ratio])
            return tuple(split_ratio)
        else:
            raise ValueError('Split ratio must be float or a list, got {}'
                             .format(type(split_ratio)))


    def stratify(examples, strata_field):
        # The field has to be hashable otherwise this doesn't work
        # There's two iterations over the whole dataset here, which can be
        # reduced to just one if a dedicated method for stratified splitting is used
        unique_strata = set(getattr(example, strata_field) for example in examples)
        strata_maps = {s: [] for s in unique_strata}
        for example in examples:
            strata_maps[getattr(example, strata_field)].append(example)
        return list(strata_maps.values())


    def rationed_split(examples, train_ratio, test_ratio, val_ratio, rnd):
        
        N = len(examples)
        randperm = rnd(range(N))
        train_len = int(round(train_ratio * N))

        # Due to possible rounding problems
        if not val_ratio:
            test_len = N - train_len
        else:
            test_len = int(round(test_ratio * N))

        indices = (randperm[:train_len],  # Train
                   randperm[train_len:train_len + test_len],  # Test
                   randperm[train_len + test_len:])  # Validation

        # There's a possibly empty list for the validation set
        data = tuple([examples[i] for i in index] for index in indices)

        return data


# In[ ]:


# from torchtext import data
# some_train_dataset = data.TabularDataset.splits(
#     path='../input/sentiment140',
#     train='training.1600000.processed.noemoticon.csv',
#     format='csv',
#     fields=fields,
#     skip_header=False
# )
#'id', 'date', 'flag', 'user', 'text'
train = MyTabularDataset(
    path= '../input/sentiment140/training.1600000.processed.noemoticon.csv', format='csv', skip_header=False,fields=fields)


# In[ ]:


#print(vars(trainset[2000]))


# In[ ]:


print(len(train))


# In[ ]:


print("type(train):", type(train), vars(train[0]))


# In[ ]:


print( train[0].target, train[0].text)


# In[ ]:


# # Python program to generate WordCloud 
  
# # importing all necessery modules 
# from wordcloud import WordCloud, STOPWORDS 
# import matplotlib.pyplot as plt 
# import pandas as pd 
  
# def make_word_cloud(df):  
#     comment_words = '' 
#     stopwords = set(STOPWORDS) 
  
#     # iterate through the csv file 
#     for val in df.CONTENT: 
      
#         # typecaste each val to string 
#         val = str(val) 
  
#         # split the value 
#         tokens = val.split() 
      
#         # Converts each token into lowercase 
#         for i in range(len(tokens)): 
#             tokens[i] = tokens[i].lower() 
      
#         comment_words += " ".join(tokens)+" "
  
#         wordcloud = WordCloud(width = 800, height = 800, 
#                 background_color ='white', 
#                 stopwords = stopwords, 
#                 min_font_size = 10).generate(comment_words) 
  
#     # plot the WordCloud image                        
#     plt.figure(figsize = (8, 8), facecolor = None) 
#     plt.imshow(wordcloud) 
#     plt.axis("off") 
#     plt.tight_layout(pad = 0) 
  
#     plt.show() 


# ### Build the vocabulary
# 
# We have the train, validation data sets. We also have out "containers" or objects defined to hold the vocabolary for Text. We can use the build_vocab on the Text field to build a vocab object using one or more datasets. We can find [more details here](https://torchtext.readthedocs.io/en/latest/data.html#field)

# In[ ]:


import random
# Split into training and test
# train_size = int(0.8 * len(train))
# val_size = len(train) - train_size
# trainset, valset = random_split(train, [train_size, val_size])
seed = 42
max_vocab_size = 5000 

train_data, valid_data = train.split(split_ratio=0.8, random_state=random.seed(seed))

# Text.build_vocab(train_data, valid_data, max_size=max_vocab_size)
# Target.build_vocab(train_data)
# vocab_size = len(Text.vocab)


# # This implementation can be split up into the following high level steps
# 
# * Preprocess data to remove unwanted characters and tokenize
# * Process input data to build vocabolary and load Glove Embeddings
# * build the model
#     - train 
#     - test
# 

# In[ ]:


from torchtext import vocab
from torchtext.vocab import GloVe
# the following results in an error - 
# embeddings = vocab.Vectors('glove.840B.300d.txt', '../input/pickled-glove840b300d-for-10sec-loading/')
# Text.build_vocab(train_data, valid_data, max_size=max_vocab_size, vectors=embeddings) 

# based on documentation here, use the steps detailed https://pytorch.org/text/datasets.html : glove.6B.200d
Text.build_vocab(train_data, vectors=GloVe(name='6B', dim=200))
Target.build_vocab(train_data)


# In[ ]:


print(vars(Text.vocab))


# In[ ]:


print(Text.numericalize(([['good','stuff']],2),device=None))


# In[ ]:


# this is not how we get the embeddings
# import torch.nn as nn
# embed_size = 300
# vocab_size = 5000
# offsets = [len(train[0].text)]
# print(offsets)
# embedding = nn.Embedding(vocab_size, embed_size)
# embedding(train[0].text)


# In[ ]:


# examples from pytorch docs I think
# word_to_ix = {"hello": 0, "world": 1}
# embeds =nn.Embedding(vocab_size, embed_size)  # 2 words in vocab, 5 dimensional embeddings
# lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
# hello_embed = embeds(lookup_tensor)
# print(hello_embed)


# In[ ]:


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# In[ ]:


device = get_default_device()
print("device: ",device)


# **Train the model 4-5 times with different learning rates & for different number of epochs.**

# In[ ]:


lr = 1e-4
num_classes = 5 # 0=negative, 4=positive

hidden_l1_size = 128
hidden_l2_size = 64
batch_size=100
epochs = 10


dropout_keep_prob = 0.5
embedding_size = 300
max_document_length = 100  
dev_size = 0.8 

hidden_size = 128
pool_size = 2
n_filters = 128
filter_sizes = [3, 8]


# In[ ]:


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[ ]:


device = get_default_device()
train_iterator, valid_iterator = data.BucketIterator.splits((train_data, valid_data),
        batch_size = batch_size,
        sort_key = lambda x: len(x.text), sort_within_batch = True,
        device = device)


# ### Data loaders, in other words iterators

# ### CNN class Implementation
# Let's implement our CNN class that inherits from Module with two fully connected or dense layers

# Note: credit goes to [galhev](https://github.com/galhev/Neural-Sentiment-Analyzer-for-Modern-Hebrew/blob/master/models/cnn_model.py) as I have heavily borrowed code here.

# Let's recall max-pooling ![max-pool](https://computersciencewiki.org/images/8/8a/MaxpoolSample2.png)

# Read more about Conv1D, Conv2D and Conv3D [here](https://medium.com/@xzz201920/conv1d-conv2d-and-conv3d-8a59182c4d6)
# 
# 
# Conv2D used in Image processing where kernel traverses in two dimensions see diagram below
# 
# ![kernel traversal good diagram for conv2d](https://miro.medium.com/max/1400/0*k0YJHHjTUY1WSsT-.png)

# 

# In[ ]:


import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, n_filters, 
                 filter_sizes, pool_size, hidden_size, num_classes,
                 dropout):
        super().__init__()        
        # initalize embedding with our vocabulary size and embedding size (the number of dimensions for each token)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        #
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=(fs,embed_size))  for fs in filter_sizes])
        #max pool layer
        self.max_pool1 = nn.MaxPool1d(pool_size)
        # ReLU
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(95*n_filters, hidden_size, bias=True)  
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True)  

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)  
#     def forward(embedded):
        convolution = [conv(embedded) for conv in self.convs]   
        max1 = self.max_pool1(convolution[0].squeeze()) 
        max2 = self.max_pool1(convolution[1].squeeze())
        cat = torch.cat((max1, max2), dim=2)      
        x = cat.view(cat.shape[0], -1) 
        x = self.fc1(self.relu(x))
        x = self.dropout(x)
        x = self.fc2(x)  
        return x


# ### fit and accuracy functions

# In[ ]:


# def accuracy(outputs, labels):
#     _, preds = torch.max(outputs, dim=1)
#     return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# class ImageClassificationBase(nn.Module):
#     def training_step(self, batch):
#         images, labels = batch 
#         out = self(images)                  # Generate predictions
#         loss = F.cross_entropy(out, labels) # Calculate loss
#         return loss
    
#     def validation_step(self, batch):
#         images, labels = batch 
#         out = self(images)                    # Generate predictions
#         loss = F.cross_entropy(out, labels)   # Calculate loss
#         acc = accuracy(out, labels)           # Calculate accuracy
#         return {'val_loss': loss.detach(), 'val_acc': acc}
        
#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
#         batch_accs = [x['val_acc'] for x in outputs]
#         epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
#         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
#     def epoch_end(self, epoch, result):
#         print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
#             epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def accuracy(probs, target):
    predictions = probs.argmax(dim=1)
    corrects = (predictions == target)
    accuracy = corrects.sum().float() / float(target.size(0))
    return accuracy


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
#         # let's 
        predictions = model(text, text_lengths)
#         predictions = model(batch.embedded)
        loss = criterion(predictions, batch.target.squeeze()) #batch.target.squeeze()
        acc = accuracy(predictions, batch.target)        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def run_train_and_val(epochs, model, train_iterator, valid_iterator, optimizer, criterion, model_type):
    best_valid_loss = float('inf')

    for epoch in range(epochs):

        # train the model
        train_loss, train_acc = train(model, train_iterator,    optimizer, criterion)

        # evaluate the model
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights'+'_'+model_type+'.pt')

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.target)
            acc = accuracy(predictions, batch.target)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[ ]:


device = get_default_device()
# loss_func = nn.CrossEntropyLoss()
# model = CNN(max_vocab_size, embedding_size, n_filters, filter_sizes, pool_size, hidden_size, num_classes,
#                     dropout_keep_prob)
# cnn_model = to_device(model, device)
# model_type = "CNN"
# #cnn_model
# optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)
# run_train_and_val(epochs, cnn_model, train_iterator, valid_iterator, optimizer, loss_func, model_type)
path = "./"
cnn_model.load_state_dict(torch.load(os.path.join(path, "saved_weights_CNN.pt")))
test_loss, test_acc = evaluate(cnn_model, valid_iterator, loss_func)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


import jovian
jovian.commit(project=project_name, environment=None)

