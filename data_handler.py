from nltk.corpus import treebank
import math


class DataHandler:

    def __init__(self, path, run_id=1, runs=1):
        with open(path) as file:
            file_ids = file.read().split(",")
            if runs == 1:
                self.file_ids = file_ids
                return
            count = math.ceil(len(file_ids)/runs)
            self.file_ids = file_ids[(run_id - 1)*count: run_id*count]

    def generator(self):
        for index, file in enumerate(self.file_ids):
            if index % 10 == 0:
                print("Processed " + str(index) + " of " + str(len(self.file_ids)) + " files")
            parsed_sentences = treebank.parsed_sents(file)
            sentences = treebank.sents(file)
            for i in range(len(parsed_sentences)):
                yield {
                    'file': file,
                    'id': i,
                    'raw': sentences[i],
                    'parsed': parsed_sentences[i]
                }
