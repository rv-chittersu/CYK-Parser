import argparse
import config
from parse_utils import CYKParser
from data_handler import DataHandler
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
from utils import base_category, stitch_files
from tqdm import tqdm
import time
import os


def nap():
    time.sleep(1)


def train():
    parser = CYKParser()
    data_handler = DataHandler(config.train_set)
    parser.train(data_handler)
    parser.save(config.model_path)


def parse_tree(parser, sent):
    result_tree = parser.parse(sent['raw'])
    tree = sent['parsed']
    [base_category(subtrees) for subtrees in tree.subtrees()]
    tree.chomsky_normal_form()
    if result_tree is None:
        # write not parsed to separate file
        print(str(os.getpid()) + ": Unable to parse sent-" + str(sent[id]) + " from file " + sent['file'])
    else:
        with open(config.target_folder + "/result-" + str(os.getpid()) + ".txt", 'a+') as f:
            f.write(str(result_tree[0]).replace('\n', ''))
            f.flush()
        with open(config.target_folder + "/gold-" + str(os.getpid()) + ".txt", 'a+') as f:
            f.write(str(tree).replace('\n', ''))
            f.flush()


def test(path):
    parser = CYKParser.load(path)
    data_handler = DataHandler(config.test_set)
    executor = ProcessPoolExecutor(4)
    futures = [executor.submit(parse_tree, parser, sent) for sent in data_handler.generator()]
    kwargs = {
        'total': len(futures),
        'unit': 'nap',
        'unit_scale': True,
        'leave': True
    }
    for _ in tqdm(as_completed(futures), **kwargs):
        pass
    for future in futures:
        if future.exception() is not None:
            print(future.exception())
    print("Done parsing")
    stitch_files()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', default='train', help='train or test', required=True)
    parser.add_argument('--model', dest='model_path', default=config.model_path, help='model path')
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test(args.model_path)
