# Story Cloze Test - NLU Project 2

The goal of this project is to predict the right ending of a 4-sentence story 
among two alternatives.

## Data
- training set: containing 88161 five-sentence short stories that include only the correct ending
- validation set: containing 1871 stories with positive and negative endings
- test set: containing stories with two endings
- cloze test set: additional test set with right ending labels


## Our Models

To predict the right endings, we experimented with the following models:
- CNN ngrams
- CNN LSTM
- Siamese LSTM
- Feed-forward neural network

For details, please refer to the report in `nlu_project_2/report`.


## Getting Started

### Virtual environment

Create a new conda virtual environment with required packages

```
conda env create -n nlu_project -f=/path/to/requirements.txt
```

Activate the virtual environment

```
source activate nlu_project
```

### Preprocess data
You can generate preprocessed data with pos tags (needed for some models) by running:

```
python preprocessing.py
```

To get augmented training data with wrong endings randomly sampled from the context, run:
```
python negative_endings.py
```
and it will create the file `nlu_project_2/data/train_set_sampled.csv`

You can also download the preprocessed data from this [link](https://polybox.ethz.ch/index.php/s/PQ6bl6fPqKDn9vz) 
(or alternatively, [here](https://drive.google.com/open?id=1wjolQtvZZHWZSd3MOfufIaPNsYZkxXxY)). 
Then you need to copy the files in `nlu_project_2/data/` to run the models directly.


### Pre-trained skip-thought embeddings

For the feed-forward neural network, we used pre-trained skip-thoughts embeddings 
from [ryankiros](https://github.com/ryankiros/skip-thoughts). You need to download the 
embedding files as specified in the project's readme:
```
wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
```
and copy the files in `nlu_project_2/src/models/skip_thoughts/data`.

Please *note* that we modified his code to make it work for our project.


## Running
To **train** our models, run:
```
python run.py -m model-name -t
```
where `model-name` refers to one of our models, namely `cnn_ngrams`, `siameseLSTM`, `cnn_lstm`, 
`ffnn` `ffnn_val`.

The models are saved after every epoch in `nlu_project_2/trained_models/model-name/date[hour]/model.h5`.

You can find the pretrained models and the prediction files for test set
 in this [folder](https://polybox.ethz.ch/index.php/s/bRnpIz66EB7g1xD).
 
Please *note* that for evaluation and prediction, it will retrieve the last trained model. If you
would like to test on our pretrained models, we suggest you to do prediction first, and
then do training / testing again to verify our model.

To **evaluate** our trained models on the cloze test set, run:
```
python run.py -m model-name -e
```

To **predict** the endings for the given test set, run:
```
python run.py -m model-name -p
```
It will generate a csv file with the right ending labels in 
`nlu_project_2/trained_models/model-name/date[hour]` for the last trained model.

