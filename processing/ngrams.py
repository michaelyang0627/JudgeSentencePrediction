import os
from glob import glob
from nltk import bigrams as bigramize, ngrams, word_tokenize, PorterStemmer, trigrams
import pandas as pd
import re
from collections import Counter
from zipfile import ZipFile
from random import shuffle
from nltk.corpus import stopwords
import numpy as np

os.chdir("/Users/michaelyang/Desktop/MLSummerProject/10sentence/JudgeSentences/")


def get_ngrams(n=2, top_k=200000, year_range=(2002, 2016)):
    dict_grams = Counter()
    for judge in judges:
        opinions = judge2year_case[judge]
        for year in opinions.keys():
            if year < year_range[0] or year > year_range[1]:
                continue
            opilist = opinions[year]
            for opi in opilist:
                fold = opinion2floder[str(opi)]
                filepath = 'rawdata/text/' + fold + '/' + str(opi) + '.txt'
                # print(filepath)
                text = open(filepath).read()
                normtext = re.sub('[^a-z0-9]', ' ', text.lower())
                tokens = normtext.split()
                tokens = [porter.stem(w) for w in tokens if len(w) >= 3 and w not in stoplist]
                if n == 1:
                    dict_grams.update(tokens)
                    continue
                if n == 2:
                    gramset = set(['_'.join(b) for b in bigramize(tokens)])
                if n == 3:
                    gramset = set(['_'.join(b) for b in trigrams(tokens)])
                if n > 3:
                    gramset = set(['_'.join(b) for b in ngrams(tokens, n)])
                dict_grams.update(gramset)
        print(judge + " : " + str(len(dict_grams)))
    return dict_grams.most_common(top_k)


if __name__ == '__main__':
    porter = PorterStemmer()
    stoplist = set(porter.stem(w) for w in stopwords.words('english'))

    case2year = pd.read_pickle('datasets/case2year.pkl')
    judge2year_case = pd.read_pickle('datasets/judge2year_case.pkl')
    opinion2floder = pd.read_pickle('datasets/opinion2floder.pkl')
    judges = judge2year_case.keys()

    grams_len = 0
    grams = []
    for i in range(1, 6):
        dict_grams = get_ngrams(n=i, year_range=(2002, 2016))
        # make it ordinary dictionary instead of counter
        grams_n = {k: v for k, v in dict_grams}
        grams_len += len(grams_n)
        grams += list(grams_n.keys())
        print(grams_len)
    index = np.random.permutation(range(grams_len))
    grams_dict = dict(zip(grams, index))
    grams_dict_inverse = dict(zip(index,grams))

    pd.to_pickle(grams_dict,"datasets/grams_dict2002-2016/grams_dict.pkl")
    pd.to_pickle(grams_dict_inverse,"datasets/grams_dict2002-2016/grams_dict_inverse.pkl")

