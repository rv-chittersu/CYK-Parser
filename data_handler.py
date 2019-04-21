from nltk.corpus import treebank


class DataHandler:

    def __init__(self, path):
        with open(path) as file:
            self.file_ids = file.read().split(",")

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
