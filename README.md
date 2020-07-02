## Text classification using CNN

### Background
I have been working on various NLP related projects for a long time as part of my daily work. The work mostly revolves around data analytics, gold-standard creation, manual annotation which then leads to applying traditional methods for many Named Entity recognition tasks. These work-related activities can broadly categorized as classificaiton tasks. However, the process of manual annotation has been the most resource intensive step.

The recent developments have changed the landscape of NLP, specifically with [the release of BERT](https://github.com/google-research/bert). Although I have graduate level training in ML, this is my effort to see how I can educate myself about deeplearning. This training from JovianML has definetly helped me take a good step toward acheiving that goal.

### The Problem 

Given that I had to start somewhere and that I did not want to chew off too much, I picked sentiment analysis, a classification problem. Whew! I know have narrowed down to a topic, this is great I though until I searched for "sentiment" on [Kaggle datasets](https://www.kaggle.com/datasets?search=sentiment). I decided to go with [this dataset](https://www.kaggle.com/kazanova/sentiment140) to analyze given that it had 1.6 million examples and 2 classes (or three).

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


#### The Ebeddings

#### The CNN implementation


