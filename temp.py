def demo():
    """
    A demonstration showing how each tree transform can be used.
    """

    from nltk.draw.tree import draw_trees
    from nltk import tree, treetransforms
    from copy import deepcopy
    from nltk.corpus import treebank
    # original tree from WSJ bracketed text
    sentence = """(TOP
  (S
    (S
      (VP
        (VBN Turned)
        (ADVP (RB loose))
        (PP
          (IN in)
          (NP
            (NP (NNP Shane) (NNP Longman) (POS 's))
            (NN trading)
            (NN room)))))
    (, ,)
    (NP (DT the) (NN yuppie) (NNS dealers))
    (VP (AUX do) (NP (NP (RB little)) (ADJP (RB right))))
    (. .)))"""
    # t = tree.Tree.fromstring(sentence, remove_empty_top_bracketing=True)
    t = treebank.parsed_sents('wsj_0111.mrg')[0]
    # collapse subtrees with only one child
    collapsedTree = deepcopy(t)
    treetransforms.collapse_unary(collapsedTree)

    # convert the tree to CNF
    cnfTree = deepcopy(collapsedTree)
    treetransforms.chomsky_normal_form(cnfTree)

    # convert the tree to CNF with parent annotation (one level) and horizontal smoothing of order two
    parentTree = deepcopy(collapsedTree)
    treetransforms.chomsky_normal_form(parentTree, horzMarkov=2, vertMarkov=1)

    # convert the tree back to its original form (used to make CYK results comparable)
    original = deepcopy(parentTree)
    treetransforms.un_chomsky_normal_form(original)

    # convert tree back to bracketed text
    sentence2 = original.pprint()
    print(sentence)
    print(sentence2)
    print("Sentences the same? ", sentence == sentence2)

    draw_trees(cnfTree)


if __name__ == '__main__':
    demo()

from nltk.corpus import treebank
from parse_utils import *
parser = CYKParser.load('ckpt/model.pt')
sentence = treebank.sents('wsj_0111.mrg')[0]
parse_table = ParseTable(sentence)
sentence_len = len(sentence)
for substring_len in range(1, sentence_len + 1):
    executor = ProcessPoolExecutor(10)
    start_indices = range(sentence_len + 1 - substring_len)
    futures = [executor.submit(parser.parse_sub_str, substring_len, i, parse_table) for i in start_indices]
    wait(futures)
