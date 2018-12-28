import os

temp = os.listdir('data/')


class Preprocess:
    files = os.listdir('data/')

    def __init__(self, threshold, files):
        self.threshold = threshold
        self.files = files
        self.thresholded_sentences = []

    def prepare_sentences(self):
        for file in self.files:
            file_content = open('data/' + file).read()
            line_seprated_content = [line.strip() for line in file_content.splitlines()]
            for sentences in set(line_seprated_content):
                if len(sentences.split(' ')) == self.threshold:
                    self.thresholded_sentences.append(line_seprated_content)
        return self.thresholded_sentences


preprocess = Preprocess(8, temp)
print(len(preprocess.prepare_sentences()))
