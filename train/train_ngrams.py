import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from gensim import corpora, models, similarities
from gensim import corpora
import gensim

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn import cross_decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import hstack, vstack

import os, sys


os.chdir("/Users/michaelyang/Desktop/MLSummerProject/JudgeSentencePrediction/train/")

class textfeature():
    """docstring for ClassName"""

    def __init__(self, top_k=200, sparse = True):
        self.vectorizer = DictVectorizer(sparse=sparse)
        self.tfidf = TfidfTransformer()
        self.top_k = top_k
        self.f_reg = SelectKBest(f_regression, k=self.top_k)
        self.acc = 0
        self.index2judge_year = Counter()

    def load_data(self, path, format='csv', index_col=0):
        if format == 'csv':
            print('load sucess from ' + path)
            return pd.read_csv(path, index_col=index_col)
        if format == 'pkl':
            print('load sucess from ' + path)
            return pd.read_pickle(path)

    def process_data(self, data, judge_year_index, bow_feature,
                     y_label='demean_harshness',
                     drops=['judgeid', 'demean_harshness', 'sentyr'],
                     istrain=True):
        y_dict = {}
        ori = {}
        for judge in bow_feature.keys():
            d = {}
            for year in bow_feature[judge].keys():
                try:
                    index = judge_year_index[judge][year]
                except TypeError:
                    # print(judge, year)
                    pass
                logit = (data.sentyr == year) & (data.judgeid == judge)
                harshness = data[y_label][logit]
                if harshness.shape[0] == 0:
                    continue
                y_dict[index] = harshness.values[0]
                ori[index] = list(data.drop(drops, axis=1).loc[logit.index[logit == True][0]])
                d['judge'] = judge
                d['year'] = year
                self.index2judge_year[index] = d

        if istrain:
            self.train_y_dict = y_dict
            self.train_ori = ori
        else:
            self.test_y_dict = y_dict
            self.test_ori = ori

    def stack_processed_features(self, train_data, test_data, judge_year_index,
                  features_list,drops=['judgeid', 'demean_harshness', 'sentyr'],
                  y_label='demean_harshness',to_array = False):

        processed_features_list_train = []
        processed_features_list_test = []

        for i in range(0, len(features_list)):
            self.process_data(train_data, judge_year_index, features_list[i],
                              istrain=True, drops=drops,
                              y_label=y_label)
            self.process_data(test_data, judge_year_index, features_list[i],
                              istrain=False, drops=drops,
                              y_label=y_label)

            self.get_train_test(features_list[i])
            vec_train, vec_test = self.get_vector()

            # tfidf makes the type sparse matrix even if vectorize as numpy array
            X_train, X_test = self.get_tfidf(vec_train, vec_test)

            print(X_train.shape, X_test.shape)

            processed_features_list_train.append(X_train)
            processed_features_list_test.append(X_test)

        X_train_stacked = self.stack_features(processed_features_list_train)
        X_test_stacked = self.stack_features(processed_features_list_test)

        train_total, test_total = self.combine_data(X_train_stacked, X_test_stacked)

        # Make it numpy array? if we want to use PLS
        if to_array:
            self.train_matrix = train_total.toarray()
            self.test_matrix = test_total.toarray()
        else:
            self.train_matrix = train_total
            self.test_matrix = test_total

        print("total feature shape: ", train_total.shape, test_total.shape)


    def get_train_test(self, bow_feature):
        if self.train_ori is None:
            print('Process the data first')
            return 0
        train_X = {}
        test_X = {}
        for key in self.train_y_dict.keys():
            judge = self.index2judge_year[key]['judge']
            year = self.index2judge_year[key]['year']
            value = self.train_y_dict[key]
            train_X[key] = bow_feature[judge][year]

        for key in self.test_y_dict.keys():
            judge = self.index2judge_year[key]['judge']
            year = self.index2judge_year[key]['year']
            value = self.test_y_dict[key]
            test_X[key] = bow_feature[judge][year]

        self.train_X = train_X
        self.test_X = test_X

    def id2ngram(self, ngram_dict):
        self.id2ngram = {item[1]: item[0] for item in ngram_dict.items()}

    def get_vector(self):
        bow_train = self.vectorizer.fit_transform(list(self.train_X.values()))
        bow_test = self.vectorizer.transform(list(self.test_X.values()))

        return bow_train, bow_test

    def get_tfidf(self, bow_train, bow_test):
        X_train = self.tfidf.fit_transform(bow_train)
        X_test = self.tfidf.transform(bow_test)
        return X_train, X_test

    def selectKbest_subset(self, bow_train, bow_test):
        label = list(self.train_y_dict.values())
        best_train = self.f_reg.fit_transform(bow_train, label)
        best_test = self.f_reg.transform(bow_test)
        return best_train, best_test

    def stack_features(self,processed_feature_list):
        return hstack(processed_feature_list)

    def combine_data(self, X_train, X_test):
        train_total = hstack([np.array(list(self.train_ori.values())), X_train])
        test_total = hstack([np.array(list(self.test_ori.values())), X_test])
        return train_total, test_total

    def get_k_features(self, k=200):
        feature_names = self.vectorizer.get_feature_names()
        return [self.id2ngram[feature_names[i]] for i in self.f_reg.get_support(indices=True)]

    def model_pre(self, reg, scale=1.0, dataset='Holger', model_name='regression'):
        # self.model = reg()
        y_train = np.array(list(self.train_y_dict.values())) * scale
        y_test = np.array(list(self.test_y_dict.values())) * scale

        reg.fit(self.train_matrix, y_train)
        y_pre = reg.predict(self.test_matrix)
        y_pre_train = reg.predict(self.train_matrix)
        # MSE
        self.acc = np.mean((y_pre - y_test) ** 2)
        self.std = np.std((y_pre - y_test) ** 2)
        acc_train = np.mean((y_pre_train - y_train) ** 2)
        std_train = np.std((y_pre_train - y_train) ** 2)

        # %MAD
        # self.acc = np.mean(((y_pre - y_test)/y_test))
        # self.std = np.std(((y_pre - y_test)/y_test))
        # acc_train = np.mean(((reg.predict(X_train) - y_train) / y_train))
        # std_train = np.std(((reg.predict(X_train) - y_train) / y_train))

        print("Base line all zero, test: ",np.mean(y_test**2),"train: ",np.mean(y_train**2))
        print(("Mean squared error for test data %s on %s : %.4f" % (dataset, model_name, self.acc)))
        print(("Std on squared error for test data %s on %s : %.4f" % (dataset, model_name, self.std)))
        print(("Mean squared error for train data %s on %s : %.4f" % (dataset, model_name, acc_train)))
        print(("Std on squared error for test data %s on %s : %.4f" % (dataset, model_name, std_train)))

        # plt.hist((y_pre - y_test) ** 1,bins=300,range=(-0.04,0.04))
        # plt.show()

        return reg

    def plot_scatter(self, reg, save_path='../result/', scale=1.0, name='bow_feature'):
        y_test = np.array(list(self.test_y_dict.values())) * scale
        y_train = np.array(list(self.train_y_dict.values())) * scale

        y_pre_train = reg.predict(self.train_matrix)
        y_pre_test = reg.predict(self.test_matrix)

        fig = plt.figure()
        plt.scatter(y_train, y_pre_train,c='r')
        plt.scatter(y_test, y_pre_test)
        plt.xlabel('true value')
        plt.ylabel('prediction value')
        # draw reference line
        plt.plot([-1, 1],[-1,1])
        savename = save_path + name + '.png'
        fig.savefig(savename)


    def run_model(self, model_zoo, plot_name='uni_features',
                  dataset='Holger', model_name='OLS'):

        ## model fit
        print("Start fitting.")
        reg = self.model_pre(model_zoo[model_name], dataset=dataset, model_name=model_name)

        ## plot
        # self.plot_scatter(reg, name=plot_name + '_' + model_name + '_' + dataset)

        return reg


if __name__ == '__main__':
    ngrams = textfeature()

    ## load data
    train_data = ngrams.load_data('../datasets/holger_train_judgeyear_final.csv', index_col=0)
    test_data = ngrams.load_data('../datasets/holger_test_judgeyear_final.csv', index_col=0)
    judge_year_index = ngrams.load_data('../datasets/judge_year2index.pkl', format='pkl')
    ngram_dict = ngrams.load_data('../datasets/grams_dict2002-2016/grams_dict.pkl', format='pkl')

    uni_feature = ngrams.load_data('../datasets/grams_dict2002-2016/1grams_feature_relative.pkl', format='pkl')
    bi_features = ngrams.load_data('../datasets/grams_dict2002-2016/2grams_feature_relative.pkl', format='pkl')
    tri_features = ngrams.load_data('../datasets/grams_dict2002-2016/3grams_feature_relative.pkl', format='pkl')
    forth_features = ngrams.load_data('../datasets/grams_dict2002-2016/4grams_feature_relative.pkl', format='pkl')
    fifth_features = ngrams.load_data('../datasets/grams_dict2002-2016/5grams_feature_relative.pkl', format='pkl')

    # List of models to run
    model_zoo = Counter()
    # model_zoo['OLS'] = linear_model.LinearRegression()
    model_zoo['OLS'] = linear_model.RidgeCV(np.array([10, 1, 0.1, 0.01, 0.001, 0.0001,0.00001]))
    model_zoo['PLS'] = cross_decomposition.PLSRegression(n_components=200)
    model_zoo['RF-30'] = RandomForestRegressor(max_features="sqrt", n_estimators=30)
    model_zoo['RF-log50'] = RandomForestRegressor(max_features="log2", n_estimators=50)
    model_zoo['Elastic Net'] = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.7)

    # choosing number of grams
    features_list = [uni_feature, bi_features,tri_features,forth_features,fifth_features]

    # process and stack ngram features, make it numpy array if PLS will be used
    ngrams.stack_processed_features(train_data, test_data, judge_year_index,features_list,to_array=True)

    # Run various models

    ngrams.run_model(model_zoo, model_name="PLS",
                      plot_name="1-5grams ")
    # RF_reg = ngrams.run_model(model_zoo, model_name="RF-30", # 28
    #                  plot_name="1-5grams ")
    # ngrams.run_model(model_zoo, model_name="RF-log70", # 28
    #                  plot_name="1-5grams ")
    OLS_reg = ngrams.run_model(model_zoo, model_name="OLS",
                     plot_name="1-5grams ")
    # ngrams.run_model(model_zoo, model_name="Elastic Net",
    #                  plot_name="1-5grams ")

    # OLS & RF prediction comparision
    # plt.scatter(RF_reg.predict(ngrams.test_matrix),OLS_reg.predict(ngrams.test_matrix))
    # plt.plot([-0.1,0.1],[-0.1,0.1])
    # plt.show()


    # cvres = []
    # alphas = np.linspace(0.01, 1, 50)
    # for a in alphas:
    #     ela = ngrams.model_pre(model_zoo['Elastic Net'], train_total, test_total, dataset='Holger',
    #                            model_name='Elastic Net')
    #     acc_test = np.mean((ela.predict(X_test) - y_test) ** 2)
    #     acc_train = np.mean((ela.predict(X_train) - y_test) ** 2)
    #     cvres.append([acc_train, acc_test])
