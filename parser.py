from data_handler import DataHandler
from nltk.tree import Tree
from utils import *
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import wait


# has mapping to lhs -> rhs, count
class Rule:
    def __init__(self, head):
        self.lhs = head
        self.rhs_dict = {}
        self.count = 0

    def add_rhs(self, body):
        if body in self.rhs_dict:
            self.rhs_dict[body] += 1
        else:
            self.rhs_dict[body] = 1
        self.count += 1

    def set_rhs(self, body, score):
        self.rhs_dict[body] = score

    def normalize(self):
        if self.count == 0:
            for key in self.rhs_dict:
                self.count += self.rhs_dict[key]
        for key in self.rhs_dict:
            self.rhs_dict[key] = self.rhs_dict[key] / self.count
        self.count = 1

    def __str__(self):
        str_list = []
        for body, count in self.rhs_dict.items():
            str_list.append(self.lhs + " -> " + body + "," + str(count))
        return "\n".join(str_list)


# data structure for CYK Parse Table
class ParseTable:

    def __init__(self, sentence):
        # sentence to parse as list of strings
        self.sentence = sentence

        # keys start_pos-end_pos
        # values dictionary (lhs : rhs, prob)
        self.table = dict()

    @staticmethod
    def get_key(start, end):
        return str(start) + "-" + str(end)

    # should add rule if it's valid
    def populate(self, start, end, productions: Rule):
        non_term = productions.lhs
        length = end - start
        for rhs, probability in productions.rhs_dict.items():
            if length == 1:
                if rhs == self.sentence[start]:
                    self.update_prob(start, end, non_term, rhs, probability)
            else:
                if len(rhs.split(" ")) == 1:
                    continue
                k1, k2 = rhs.split(" ")
                for mid in range(start + 1, end):
                    prob_k1 = self.get_prob(start, mid, k1)
                    prob_k2 = self.get_prob(mid, end, k2)
                    if prob_k1 > 0 and prob_k2 > 0:
                        prob = prob_k1*prob_k2*probability
                        k1 += ":" + self.get_key(start, mid)
                        k2 += ":" + self.get_key(mid, end)
                        self.update_prob(start, end, non_term, k1 + " " + k2, prob)

    # gives probability of non terminal entry in the table cell
    def get_prob(self, start, end, non_term):
        partition_key = self.get_key(start, end)
        if partition_key not in self.table:
            return -1
        non_term_dict = self.table[partition_key]
        if non_term not in non_term_dict:
            return -1
        return non_term_dict[non_term][1]

    # updates entry in the table cell
    def update_prob(self, start, end, non_term, rhs, prob):
        partition_key = self.get_key(start, end)
        if prob < self.get_prob(start, end, non_term):
            return
        if partition_key not in self.table:
            self.table[partition_key] = {}
        non_term_dict = self.table[partition_key]
        non_term_dict[non_term] = [rhs, prob]

    # get the entry by non terminal in the table
    def get_entry(self, partition_key, non_term):
        if partition_key not in self.table:
            raise Exception("Un Expected Partition Key " + partition_key)
        if non_term not in self.table[partition_key]:
            return None
        return self.table[partition_key][non_term]

    # build tree from generated table
    def build_tree(self, node, partition_key):
        start, end = [int(x) for x in partition_key.split("-")]
        sentence_len = end - start
        result = self.get_entry(partition_key, node)
        if result is None:
            return node, -1
        production, probability = result
        tree = Tree(node, list())
        parts = production.split(" ")
        if sentence_len == 1:
            if len(parts) != 1:
                raise Exception("Un Expected Rule!!")
            if parts[0] == node:
                tree.append(parts[0])
            else:
                tree.append(self.build_tree(parts[0], partition_key)[0])
        else:
            if len(parts) != 2:
                raise Exception("Un Expected Rule!!")
            node1, key1 = parts[0].rsplit(":", 1)
            tree.append(self.build_tree(node1, key1)[0])
            node2, key2 = parts[1].rsplit(":", 1)
            tree.append(self.build_tree(node2, key2)[0])
        return tree, probability


class CYKParser:

    def __init__(self):
        # key: lhs
        # value: Rule
        self.rules = {}

        # priors are generated when parsing
        # useful when token is OOV
        self.priors = None

    # load saved model
    @staticmethod
    def load(path):
        model = CYKParser()
        with open(path) as file:
            for line in file.read().split("\n"):
                if len(line) == 0:
                    continue
                head, body_part = line.split(" -> ")
                body = ",".join(body_part.split(",")[:-1])
                count = int(body_part.split(",")[-1:][0])
                if head not in model.rules:
                    model.rules[head] = Rule(head)
                model.rules[head].set_rhs(body, count)
        model.normalize()
        model.initialize_priors()
        return model

    # calculate production occurrences
    def train(self, data_handler: DataHandler):
        for sentence in data_handler.generator():
            tree = sentence['parsed']
            [base_category(subtrees) for subtrees in tree.subtrees()]
            tree.chomsky_normal_form()
            productions = tree.productions()
            for production in productions:
                head, body = parse_production(production)
                self.update_counts(head, body)
        print("Finished Processing Data")

    def update_counts(self, head, body):
        if head not in self.rules:
            self.rules[head] = Rule(head)
        self.rules[head].add_rhs(body)

    # normalize counts to get probability
    def normalize(self):
        for key in self.rules:
            self.rules[key].normalize()

    # save model
    def save(self, path):
        with open(path, 'w') as file:
            for head in self.rules:
                file.write(str(self.rules[head]))
                file.write("\n")

    # initialize prob of non terminal generating token
    def initialize_priors(self):
        nt_dict = dict()
        count = 0
        for rule in self.rules:
            for rhs in self.rules[rule].rhs_dict:
                if len(rhs.split(" ")) != 1:
                    continue
                if rhs in self.rules:
                    continue
                if rule in nt_dict:
                    nt_dict[rule] += self.rules[rule].rhs_dict[rhs]
                else:
                    nt_dict[rule] = self.rules[rule].rhs_dict[rhs]
                count += self.rules[rule].rhs_dict[rhs]
        for key in nt_dict:
            nt_dict[key] /= count
        self.priors = nt_dict

    # special case of parse sub_str where str_len is 1.
    def parse_terminals(self, start, parse_table: ParseTable):
        word = parse_table.sentence[start]
        nt_dict = dict()
        i = 0
        while True:
            i += 1
            updated = False
            for rule in self.rules:
                for rhs in self.rules[rule].rhs_dict:

                    # We have added this rule
                    if rule in nt_dict and (nt_dict[rule][0] == rhs or nt_dict[rule][0] == word):
                        continue

                    # is_valid takes care of case and numerical tokens
                    if is_valid(rhs, word):
                        nt_dict[rule] = [word, self.rules[rule].rhs_dict[rhs]]
                        updated = True
                    if rhs in nt_dict.keys():
                        new_prob = nt_dict[rhs][1] * self.rules[rule].rhs_dict[rhs]
                        if rule in nt_dict and nt_dict[rule][1] > new_prob:
                            continue
                        nt_dict[rule] = [rhs, new_prob]
                        updated = True
                        break
            if not updated:
                if i == 1:
                    # first iteration!!. The token is OOV. initialize with priors
                    for key, value in self.priors.items():
                        nt_dict[key] = [parse_table.sentence[start], value]
                else:
                    break
        parse_table.table[str(start) + "-" + str(start + 1)] = nt_dict

    def parse_sub_str(self, substring_len, start_index, parse_table):
        if substring_len == 1:
            self.parse_terminals(start_index, parse_table)
            return
        for rule in self.rules:
            parse_table.populate(start_index, start_index + substring_len, self.rules[rule])

    def parse(self, sentence):
        parse_table = ParseTable(sentence)
        sentence_len = len(sentence)
        for substring_len in range(1, sentence_len + 1):
            executor = ThreadPoolExecutor(10)
            start_indices = range(sentence_len + 1 - substring_len)
            futures = [executor.submit(self.parse_sub_str, substring_len, i, parse_table) for i in start_indices]
            wait(futures)
        init_key = parse_table.get_key(0, str(sentence_len))
        if init_key not in parse_table.table:
            print("Cannot parse")
            return None
        if "S" not in parse_table.table[init_key]:
            print("Cannot build tree with S as node. Giving other optimal result")
            nt_with_prob = {nt: parse_table.get_entry(init_key, nt)[1] for nt in parse_table.table[init_key]}
            best_nt = max(nt_with_prob, key=nt_with_prob.get)
            return parse_table.build_tree(best_nt, init_key)
        return parse_table.build_tree("S", init_key)
