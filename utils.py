from nltk.corpus import treebank
from nltk.tree import Production
import re
import nltk
import random
import config
import glob


def is_num(word):
    return re.search('[a-zA-Z*]', word) is None and re.search('[0-9]', word) is not None and word != '0'


def is_valid(terminal, word):
    if terminal == str.lower(word):
        return True
    if terminal == '<NUM>' and is_num(word):
        return True
    return False


def get_training_and_test_split(config):
    files = treebank.fileids()
    random.shuffle(files)
    training = files[:int(0.9 * len(files))]
    test = files[int(0.9 * len(files)):]

    with open(config.train_set, 'w') as file:
        file.write(",".join(training))

    with open(config.test_set, 'w') as file:
        file.write(",".join(test))


def process_parse_tree(tree):
    tree.chomsky_normal_form()


def parse_production(rule: Production):
    head = rule.lhs().symbol()
    body = rule.rhs()
    if rule.is_lexical():
        body = str.lower(body[0])
        if is_num(body):
            body = "<NUM>"
    else:
        body = " ".join([nt.symbol() for nt in body])
    return head, body


def base_category(t):
    if isinstance(t, nltk.tree.Tree):
        m = re.match("^(-[^-]+-|[^-=]+)", t.label())
        if m is not None:
            t.set_label(m.group(1))


def stitch_files():
    print("Stitching files")
    gold_list = sorted(glob.glob(config.target_folder + "/gold*"))
    result_list = sorted(glob.glob(config.target_folder + "/result*"))

    if len(gold_list) != len(result_list):
        raise Exception("Results and Gold didn't match")

    gold = []
    result = []
    for i in range(len(gold_list)):
        with open(gold_list[i]) as f:
            gold.append(f.read().strip('\n'))
        with open(result_list[i]) as f:
            result.append(f.read().strip('\n'))

    with open(config.target_folder + '/gold.txt', 'w') as f:
        f.write("\n".join(gold))
    with open(config.target_folder + '/result.txt', 'w') as f:
        f.write("\n".join(result))
