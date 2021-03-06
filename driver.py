import argparse
from parser import CYKParser
from data_handler import DataHandler
from concurrent.futures.process import ProcessPoolExecutor
from nltk.tokenize import TreebankWordTokenizer
from concurrent.futures import as_completed
from utils import *
from tqdm import tqdm
import os
import traceback


def train():
    parser = CYKParser()
    data_handler = DataHandler(config.train_set)
    parser.train(data_handler)
    parser.save(config.model_path)


def parse_tree(parser, sent, run_id):
    # get result tree and gold tree
    try:
        result_tree = parser.parse(sent['raw'])

        gold_tree = sent['parsed']
        [base_category(subtrees) for subtrees in gold_tree.subtrees()]
        gold_tree.chomsky_normal_form()

        if result_tree is None:
            # write not parsed to separate file
            print(str(os.getpid()) + ": Unable to parse sent-" + str(sent[id]) + " from file " + sent['file'])
        else:
            # write to files
            with open(config.target_folder + "/result-" + str(run_id) + "-" + str(os.getpid()) + ".txt", 'a+') as f:
                f.write(get_string(result_tree[0]))
                f.flush()
            with open(config.target_folder + "/gold-" + str(run_id) + "-" + str(os.getpid()) + ".txt", 'a+') as f:
                f.write(get_string(gold_tree))
                f.flush()
    except Exception as e:
        traceback.print_exc()
        print(e)


def test(path, run_id, runs):
    # load parser and data handler
    parser = CYKParser.load(path)
    data_handler = DataHandler(config.test_set, run_id, runs)

    # parse sentences in parallel
    executor = ProcessPoolExecutor(config.processes)
    futures = [executor.submit(parse_tree, parser, sent, run_id) for sent in data_handler.generator()]

    # following code is to track progress
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

    # stitch files if number of runs is 1
    if runs == 1:
        stitch_files()
    print("Done parsing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', default='train', help='train or test', required=True)
    parser.add_argument('--model', dest='model_path', default=config.model_path, help='model path')
    parser.add_argument('--sent', dest='sent', help='input sentence to parse')

    # pass only if running test on multiple nodes manually
    parser.add_argument('--runs', dest='runs', default=1, type=int, help='number of runs for parallel testing')
    parser.add_argument('--run_id', dest='run_id', default=1, type=int, help='run id')

    args = parser.parse_args()
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test(args.model_path, args.run_id, args.runs)
    elif args.mode == 'stitch':
        stitch_files()
    elif args.mode == 'parse':
        parser = CYKParser.load(args.model_path)
        sentence = TreebankWordTokenizer().tokenize(args.sent)
        result = parser.parse(sentence)
        if result is None:
            print("Cannot get a valid parse")
        else:
            print("Found a parse with probability - " + str(result[1]) + '\n')
            print("Constituency parsing..")
            print(result[0], '\n')
            result[0].pretty_print()

