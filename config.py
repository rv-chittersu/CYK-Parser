# train and test split of file ids
train_set = 'data/train.txt'
test_set = 'data/test.txt'

# to save during training or load during testing or parsing
model_path = 'ckpt/model.pt'

# folders to store gold and generated probabilities
target_folder = 'results'

# number of processes to spawn during test time
processes = 4

# smoothing type prob/add_one
smoothing = 'prob'