import pandas as pd
import numpy as np
from collections import Counter
from glob import glob
import os

os.chdir("/Users/michaelyang/Desktop/MLSummerProject/10sentence/JudgeSentences/datasets/")

for num_grams in range(1,2):
    pkl_path = ("grams_dict2002-2016/"+str(num_grams)+"grams_feature.pkl")
    grams_features = pd.read_pickle(pkl_path)

    print(str(num_grams)+"grams_features.pkl loaded")

    judges = grams_features.keys()
    for judge in judges:
        years = grams_features[judge].keys()
        for year in years:
            grams_features_judge_year_normal = grams_features[judge][year]
            # get total words said in a year
            total_word_count = sum(grams_features_judge_year_normal.values(),0.0)
            for word in grams_features_judge_year_normal.keys():
                #loop through all the word counts
                grams_features_judge_year_normal[word] /= total_word_count
            # update the dict
            grams_features[judge][year] = grams_features_judge_year_normal
        print(judge + " is completed for " + str(num_grams) + " grams")

    pd.to_pickle(grams_features,"grams_dict2002-2016/%sgrams_feature_relative.pkl" %str(num_grams))
    print("%s grams relative frequency completed" %str(num_grams))

#Test for results
# pkl_path = ("grams_dict2002-2016/1grams_feature_relative.pkl")
# pkl_path2 = ("grams_dict2002-2016/1grams_feature_relative.pkl")
#
# # test = pd.read_pickle(pkl_path)
# test2 = pd.read_pickle(pkl_path2)
# # print(test.keys())
# test_judge_year2 = test2["ACKER, WILLIAM MARSH, JR."][2003]
# test_judge_year3 = test2[list(test2.keys())[10]][2007]
# # print(sum(test_judge_year.values()))
# print(sum(test_judge_year2.values()))
# print(sum(test_judge_year3.values()))
# print(list(test2.keys())[10])

# from scipy.sparse import hstack, vstack,coo_matrix
# a = coo_matrix([[1,2],[3,4]])
# print(hstack((a,a)),a)
