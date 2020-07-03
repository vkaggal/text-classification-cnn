## Text classification using CNN - a basic course project

### Background
Traditional text classification tasks involves laborious work including data extraction, gold-standard creation, manual annotation where all of the manual annotation is handled by domain experts - which then leads to applying traditional, resourse intensive methods for many Named Entity recognition tasks. These activities can broadly categorized as classificaiton tasks. However, the process of manual annotation has been the most resource intensive step.

The recent developments have changed the landscape of NLP, specifically with [the release of BERT](https://github.com/google-research/bert). Although I have graduate level training in ML, this is my effort to see how I can educate myself about deeplearning. This training from JovianML has definetly helped me take a good step toward acheiving that goal.

### The Problem 

Given that I had to start somewhere and that I did not want to chew off too much, I picked sentiment analysis, a classification problem. Whew! I now have narrowed down on a topic, this is great I though until I searched for "sentiment" on [Kaggle datasets](https://www.kaggle.com/datasets?search=sentiment). After looking through a bit, I decided to go with [this dataset](https://www.kaggle.com/kazanova/sentiment140) to analyze given that it had 1.6 million examples and 2 classes (or three).

In essense, the problem statement is to classify tweets as **negative** or **positive**

### The implementation

The idea is to start with `Linear` model two dense `nn.Linear` layer with `__init__`, `forward` functions and the `fit` function that can iterate through the batches of data utilizing the loss function `nn.CrossEntropyLoss()` and the optimizer `torch.optim.Adam`. This lead me to look into the dataloader described in the next section.

#### The Dataloader

This should have been as simple as the following:

```
from torchtext import data
some_train_dataset = data.TabularDataset.splits(
     path='../input/sentiment140',
     train='training.1600000.processed.noemoticon.csv',
     format='csv',
     fields=fields,
     skip_header=False
)
```

As you may have noted that I used *should have been*, I encountered an endoding error with the `TabularDataset` implementation which seems to be wanting to load this as `utf8`. After spending a good couple of days trying to chase down the error, I decided to implement my custom Dataset:

```
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
```

#### The preprocessor

We define instances of `Field` for our `text` and `target` that can contain the Vocabolary and their corresponding numeric representations as [detailed here](https://torchtext.readthedocs.io/en/latest/data.html#field). We will use our function `preprocess_text` to remove stop words, trim spaces, remove special characters. We perform all these steps on the text field that contain the text of the tweets


How do we split the data, the options include subclassing the Dataset class or using the methods detailed here https://torchtext.readthedocs.io/en/latest/datasets.html

```
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
def remove_stopwords(text):
    word_tokens = word_tokenize(text) 
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]           
    return ' '.join(filtered_sentence)
    
# use nltk to remove stop words, remove, newlines, extra spaces, and punctuations
from nltk.tokenize import RegexpTokenizer

def preprocess_text(texts):
    return_text = []
    tokenizer = RegexpTokenizer(r'\w+')
    #preprocess_pipeline = data.Pipeline(lambda x: x.decode('latin1').encode('utf8'))
    for i, text in enumerate(texts):
        #remove non ascii chars
        text = ''.join(i for i in text if ord(i)<128)
        text = remove_stopwords(text)
        text = tokenizer.tokenize(text)
        return_text += text
    return_text_str = " ".join(str(x) for x in return_text)
    # print('return_text:', return_text, type(return_text), return_text_str, type(return_text_str))

    return return_text_str
```

#### The Ebeddings

Its now time to represent our tokens from the tweets as numbers. This is accomplished by the embedding layer (the second code snippet below) based on the vocabolary that is built utilizing **GloVe** pretrained word vectors.

##### Build the vocabulary
Th following is an attempt to build vocabulary using *GloVe* based on the documentation [found here](https://pytorch.org/text/datasets.html). This results in an error `OSError: [Errno 28] No space left on device` on Kaggle as the instance runs out of space. I guess, that is why we just need to "attach" data? I need to spend time to understand this a bit better.

```
from torchtext import vocab
from torchtext.vocab import GloVe
Text.build_vocab(train_data, vectors=GloVe(name='840B', dim=300))
Label.build_vocab(train_data)
```

Another attempt! The following tries to load GloVe vectors that has been attached/added to the instance on Kaggle. 

```
Text.build_vocab(train_data, vectors=GloVe(name='6B', dim=200))
Label.build_vocab(train_data)
```


##### Embedding layer

```
self.embedding = nn.Embedding(vocab_size, embed_size)
```

#### The CNN implementation

The CNN architecture that is yet being fixed is as follows:

- Input Vectors
- Embedding Layer (embedding size = 300)
- 1D Convolutional layer
- Max pooling layer (size = 2)
- Activation of ReLU
- Dropout rate of 0.5
- Two fully connected layers (128, 5)

```
CNN(
  (embedding): Embedding(5000, 300)
  (convs): ModuleList(
    (0): Conv1d(1, 128, kernel_size=(3, 300), stride=(1,))
    (1): Conv1d(1, 128, kernel_size=(8, 300), stride=(1,))
  )
  (max_pool1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu): ReLU()
  (dropout): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=12160, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=5, bias=True)
)
```

### Work in progress

This so far has been an exciting learning experience that started by chance, facilitated by **Jovian ML** and **FreeCodeCamp.org**. There is still more work here to be done, specifically, text embeddings, convolutions and text data among a lot more.
