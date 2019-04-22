from data_handler import DataHandler
from nltk.tree import Tree
from utils import *
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import wait


class Rule:
    def __init__(self, head):
        self.head = head
        self.expansions = {}
        self.count = 0

    def add_expansion(self, body):
        if body in self.expansions:
            self.expansions[body] += 1
        else:
            self.expansions[body] = 1
        self.count += 1

    def set_expansion(self, body, score):
        self.expansions[body] = score

    def normalize(self):
        if self.count == 0:
            for key in self.expansions:
                self.count += self.expansions[key]
        for key in self.expansions:
            self.expansions[key] = self.expansions[key]/self.count
        self.count = 1

    def __str__(self):
        str_list = []
        for body, count in self.expansions.items():
            str_list.append(self.head + " -> " + body + "," + str(count))
        return "\n".join(str_list)


class ParseTable:

    def __init__(self, sentence):
        self.sentence = sentence
        self.table = dict()

    # should add rule if it's valid
    def populate(self, start, end, productions: Rule):
        non_term = productions.head
        length = end - start
        for expansion, probability in productions.expansions.items():
            if length == 1:
                if expansion == self.sentence[start]:
                    self.update_prob(start, end, non_term, expansion, probability)
            else:
                if len(expansion.split(" ")) == 1:
                    continue
                k1, k2 = expansion.split(" ")
                for mid in range(start + 1, end):
                    prob_k1 = self.get_prob(start, mid, k1)
                    prob_k2 = self.get_prob(mid, end, k2)
                    if prob_k1 > 0 and prob_k2 > 0:
                        prob = prob_k1*prob_k2*probability
                        k1 += ":" + str(start) + "-" + str(mid)
                        k2 += ":" + str(mid) + "-" + str(end)
                        self.update_prob(start, end, non_term, k1 + " " + k2, prob)

    def get_prob(self, start, end, non_term):
        partition_key = str(start) + "-" + str(end)
        if partition_key not in self.table:
            return -1
        non_term_dict = self.table[partition_key]
        if non_term not in non_term_dict:
            return -1
        return non_term_dict[non_term][1]

    def update_prob(self, start, end, non_term, expansion, prob):
        partition_key = str(start) + "-" + str(end)
        if prob < self.get_prob(start, end, non_term):
            return
        if partition_key not in self.table:
            self.table[partition_key] = {}
        non_term_dict = self.table[partition_key]
        non_term_dict[non_term] = [expansion, prob]

    def get_entry(self, partition_key, non_term):
        if partition_key not in self.table:
            raise Exception("Un Expected Partition Key " + partition_key)
        if non_term not in self.table[partition_key]:
            return None
        return self.table[partition_key][non_term]

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
        self.rules = {}
        self.priors = None

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
                model.rules[head].set_expansion(body, count)
        model.normalize()
        model.initialize_priors()
        return model

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
        # self.normalize()

    def update_counts(self, head, body):
        if head not in self.rules:
            self.rules[head] = Rule(head)
        self.rules[head].add_expansion(body)

    def normalize(self):
        for key in self.rules:
            self.rules[key].normalize()

    def save(self, path):
        with open(path, 'w') as file:
            for head in self.rules:
                file.write(str(self.rules[head]))
                file.write("\n")

    def initialize_priors(self):
        nt_dict = dict()
        count = 0
        for rule in self.rules:
            for expansion in self.rules[rule].expansions:
                if len(expansion.split(" ")) != 1:
                    continue
                if expansion in self.rules:
                    continue
                if rule in nt_dict:
                    nt_dict[rule] += self.rules[rule].expansions[expansion]
                else:
                    nt_dict[rule] = self.rules[rule].expansions[expansion]
                count += self.rules[rule].expansions[expansion]
        for key in nt_dict:
            nt_dict[key] /= count
        self.priors = nt_dict

    def handle_init(self, start, parse_table: ParseTable):
        word = parse_table.sentence[start]
        nt_dict = dict()
        # nt_dict[parse_table.sentence[start]] = [parse_table.sentence[start], 1]
        i = 0
        while True:
            i += 1
            updated = False
            for rule in self.rules:
                for expansion in self.rules[rule].expansions:

                    # We have added this rule
                    if rule in nt_dict and (nt_dict[rule][0] == expansion or nt_dict[rule][0] == word):
                        continue

                    if is_valid(expansion, word):
                        nt_dict[rule] = [word, self.rules[rule].expansions[expansion]]
                        updated = True
                        # print("iter: " + str(i) + " " + rule + " -> " + expansion)
                    if expansion in nt_dict.keys():
                        new_prob = nt_dict[expansion][1] * self.rules[rule].expansions[expansion]
                        if rule in nt_dict and nt_dict[rule][1] > new_prob:
                            continue
                        nt_dict[rule] = [expansion, new_prob]
                        updated = True
                        # print("iter: " + str(i) + " " + rule + " -> " + expansion)
                        break
            if not updated:
                if i == 1:
                    # first iteration!!
                    for key, value in self.priors.items():
                        nt_dict[key] = [parse_table.sentence[start], value]
                else:
                    break
        parse_table.table[str(start) + "-" + str(start + 1)] = nt_dict

    def parse_sub_str(self, substring_len, start_index, parse_table):
        if substring_len == 1:
            self.handle_init(start_index, parse_table)
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
            # print(substring_len)
        init_key = "0-"+str(sentence_len)
        if init_key not in parse_table.table:
            print("Cannot parse")
            return None
        if "S" not in parse_table.table[init_key]:
            print("Cannot build tree with S as node. Giving other optimal result")
            nt_with_prob = {nt: parse_table.get_entry(init_key, nt)[1] for nt in parse_table.table[init_key]}
            best_nt = max(nt_with_prob, key=nt_with_prob.get)
            return parse_table.build_tree(best_nt, init_key)
        return parse_table.build_tree("S", init_key)
