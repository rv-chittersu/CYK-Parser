# word2vec skipgram model
This work is done as part of [assignment](https://sites.google.com/site/2019e1246/schedule/assignment-3) for [E1 246: Natural Language Understanding (2019)](https://sites.google.com/site/2019e1246/basics). The report for the same can be found [here](https://github.com/rv-chittersu/CYK-Parser/blob/master/report.pdf)

## Data
NLTK's treebank dataset is used to train the generate PCFG.

## File Structure
Project layout
```
data/
EVALB/
results/
ckpt/
requirements.txt
config.py
data_handler.py
parser.py
utils.py
driver.py
init.sh
report.pdf
```

### Config

-- config.py --

```python
# train and test split of file ids
train_set = 'data/train.txt'
test_set = 'data/test.txt'

# to save during training or load during testing or parsing
model_path = 'ckpt/model.pt'

# folders to store gold and generated probabilities
target_folder = 'target'

# number of processes to spawn during test time
processes = 4
```

make sure *config.py* has right values set before running the program

### data

The data folder contains *train.txt* and *text.txt* holding comma separated fileids split for training and testing

*Note:* One can use the their own splits by either updating *train.txt* and *test.txt* or by adding new files and updating *config.py* 

### EVALB

[EVALB](https://nlp.cs.nyu.edu/evalb/) is program used to generate scores.

### results

*gold.txt* and *result.txt* files are stored in results folder after testing phase.
These files are used for evaluation

*Note:* One can change the results path by updating in *config.py*


### ckpt

Saves checkpoint which contains production rules and counts

### Code

**data_handler.py** responsible for reading training or test file mentioned in config and generating sentences<br>
**parser.py** has core implementation of CYKParser<br>
**utils.py** has methods used by other files<br>
**driver.py** contains main<br>

## How to Run

install requirements and initialize

```
pip3 install -r requirements.txt
./init.sh

```

*Note* IN all cases below model path can be chosen from either *config.py* or **-- model <path>** path argument.

(1) parse sentence from existing model

```
python driver.py --mode parse --sent "This sentence will be parsed ."

```

(2) train the model

```
python driver.py --mode train
```

(3) test the model

```
python driver.py --mode test
./eval -p param results/gold.txt results/result.txt

```
**Note** result directory which contain gold and result parses can be found at **target_folder** in **config.py**.

