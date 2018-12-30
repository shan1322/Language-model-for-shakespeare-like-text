import os
import nltk
import numpy as np
from keras.utils import np_utils
from nltk.corpus import stopwords
from tqdm import tqdm


class Preprocess:

    def __init__(self, threshold, files):
        """

        :param threshold:Number of words delay before predicting the next worsd
        :param files:location of directory having the txt files
        """
        self.threshold = threshold
        self.files = files
        self.thresholded_sentences = []
        self.agg_files = " "
        self.agg_word_features = []
        self.agg_no_features = []
        self.agg_word_label = []
        self.agg_no_label = []
        self.cleanup = " "
        self.n_to_char = {}
        self.char_to_n = {}
        self.charcters = []

    def aggregate_file(self):
        """

        :return: all the files aggregrated into one variable
        """
        (self.files) =self.files[0:10]

        for file in self.files:
            file_content = open('data/' + file).read()
            self.agg_files = self.agg_files + "\n" + file_content
        return self.agg_files

    def cleanup_doc(self):
        """

        :return: cleaned up doc without stop words and tokens
        """
        stopset = set(stopwords.words('english'))
        self.cleanup = " ".join(filter(lambda word: word not in stopset, self.agg_files.split()))
        return self.cleanup

    def word_token(self):
        """

        :return: dict with no as key and word as value,dict with word as key and no as value,total unique charcters
        """
        tokens = nltk.word_tokenize(self.cleanup)
        self.characters = sorted(list(set(tokens)))
        self.n_to_char = {n: char for n, char in enumerate(self.characters)}
        self.char_to_n = {char: n for n, char in enumerate(self.characters)}
        return self.n_to_char, self.char_to_n, (self.characters)

    def feature_label_extraction(self):
        """

        :return: array containing 4 words ,array containing no of those 4 words,array containig word to be predicted(label),array containg value of the word label
        """
        key = self.char_to_n.keys()
        count = 0
        single_feature = []
        single_feature_no = []
        for word in tqdm(self.cleanup.split(' ')):
            for word1 in key:
                if word == word1 and count < self.threshold:
                    single_feature_no.append(self.char_to_n[word] / (len(self.characters)))
                    single_feature.append(word1)
                    count = count + 1
                    break
                elif word == word1 and count == self.threshold:
                    self.agg_no_label.append(self.char_to_n[word])
                    self.agg_no_features.append(single_feature_no)
                    self.agg_word_features.append(single_feature)
                    self.agg_word_label.append(word)
                    single_feature = []
                    single_feature_no = []
                    count = 0
        self.agg_no_label = np_utils.to_categorical(self.agg_no_label, len(self.characters))
        return self.agg_word_features, self.agg_word_label, self.agg_no_features, self.agg_no_label

    def save_npy(self):
        """
        
        :return:saved numpy matrices 
        """

        return np.save("../processed data/word_features.npy", self.agg_word_features), np.save(
            "../processed data/word_label.npy", self.agg_word_label), np.save("../processed data/no_features.npy",
                                                                            self.agg_no_features), np.save(
            "../processed data/no_label.npy", self.agg_no_label), np.save("../processed data/charcters.npy", self.characters)


location = os.listdir('../data/')
pre = Preprocess(4, location)
pre.aggregate_file()
pre.cleanup_doc()
pre.word_token()
pre.word_token()
pre.feature_label_extraction()
pre.save_npy()
