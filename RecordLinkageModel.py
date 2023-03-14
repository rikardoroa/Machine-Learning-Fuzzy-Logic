import pandas as pd
import numpy as np
import re
import recordlinkage
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
nltk.download("popular")


# step 1 - cleaning
# previous cleaning rules with regex method applied in aws environment
def cleaning_datasets(df, col1):
    try:
        # pattern for cleaning
        pattern = re.compile(r'([^A-Za-z0-9\s])')
        class_le = LabelEncoder()
        categorie_val = class_le.fit_transform([x for x in df.columns])
        custom_stop_word_list = ['THE', 'Y', 'AS', 'IS', 'NOT', 'TO']
        df_temp = df.copy()
        df_temp["ORIGINAL_NAMES_COMPANIES"] = df_temp[[col1]]
        for col, label in zip(df_temp.columns, categorie_val):
            df_temp[col] = df_temp[col].apply(lambda val: re.sub(pattern, ' ',val))
            if label == 1 or label == 0:
                df_temp[col] = df_temp[col].apply(
                    lambda x: ' '.join([word for word in word_tokenize(x) if word.upper() not in custom_stop_word_list]))
                df_temp = df_temp.drop_duplicates().reset_index(drop=True)
        return df_temp
    except Exception as e:
        print("an error occurred, could not finish the dataset cleaning, please review the process", e)


# step 2 - reindexing
def temp_index(df, id_col, col, col1):
    try:
        lower_col = col
        df_temp = df[[col, col1]]

        # dropping duplicates and cluster id creation
        def copy_(df):
            df_f = df.copy()
            df_f[id_col] = "NaN"
            df_f = df_f.drop_duplicates().reset_index(drop=True)
            for i in range(0, df_f.shape[0]):
                index_val = "rec_" + str(i) + "_" + lower_col.lower()
                df_f.loc[i:i, id_col] = index_val
            df_f.set_index([id_col], inplace=True, drop=True)
            return df_f
        df_temp = copy_(df_temp)
        df = copy_(df)
        return df_temp, df
    except Exception as e:
        print("an error occurred, the index can not be created, please review the process", e)


# step 3 - comparision
def df_comparision(df1, df2, blockindexer_df1, label1, label2, label3):
    try:
        # indexing
        indexer = recordlinkage.Index()
        indexer.block(blockindexer_df1)
        candidate_links = indexer.index(df1, df2)

        # comparision
        method = 'jarowinkler'
        compare = recordlinkage.Compare()
        compare.string(label1, label2, method=method, label=label3)
        features = compare.compute(candidate_links, df1, df2)
        return features
    except Exception as e:
        print("an error occurred, the cluster can not be created, please review the process", e)


# step 4 - creating truelinks
def true_links(df, col):
    try:
        # threshold
        threshold = 0.85
        # passing df data with 85%  of threshold (85% of matches)
        true_links = df.loc[(df[col] > threshold)].index
        return true_links
    except Exception as e:
        print("an error occurred, can not load the dataset, please review", e)


# step 4 - model training and evaluation
def svm_classifier(df, true_links):
    try:
        # svm model
        X = df
        X_train, X_test = train_test_split(X, test_size=0.60, random_state=42)
        golden_matches_test_index = X_test.index.intersection(true_links)
        svm = recordlinkage.SVMClassifier()
        svm.fit(X_test, golden_matches_test_index)
        result_svm = svm.predict(df)
        return result_svm, golden_matches_test_index
    except Exception as e:
        print("an error occurred, the dataset is not balanced or you have few data for training please review", e)


def eval_fscore(true_links, result_svm):
    # model recall,fscore and precision
    try:
        print(f"the model fscore is:{np.round((recordlinkage.fscore(true_links, result_svm) * 100), 1)}%")
        print(f"the model recall is:{np.round((recordlinkage.recall(true_links, result_svm) * 100), 1)}%")
        print(f"the model precision is:{np.round((recordlinkage.precision(true_links, result_svm) * 1), 1)}")
    except Exception as e:
        print("can not print the model evaluation, please review the input data", e)


def confusion_matrix(df, golden_matches_test_index, result_svm):
    # confusion matrix
    try:
        matrix = recordlinkage.confusion_matrix(golden_matches_test_index, result_svm, len(df))
        return matrix
    except Exception as e:
        print("could not print the model evaluation, please review the process", e)


# step 5 - model application
def matching_index_results(df_match_data, result_svm):
    try:
        df = df_match_data.loc[result_svm.values.tolist()].reset_index(drop=False)  # df1_c1, #df2_c1
        return df
    except Exception as e:
        print("can not merge the model results into the dataset, please review the input data", e)


# step 6 - merging all data
def matching_data(df_match_data, col_1, df, col_2):  # match_data = df1_c1
    try:
        values = []
        for item in df_match_data[col_1].items():  # col_1 = id_1(df1_a1), col_1=id_2(df1_b1)
            values.append(df.loc[item[1]:item[1]:, col_2][0])  # df1_a1,col_2="CPAIQ_MATCH", df1_b1,col_2="CENCIA"
        df_f = pd.DataFrame(values).rename(columns={0: col_2})
        df_f = pd.concat([df_match_data, df_f], axis=1)
        return df_f
    except Exception as e:
        print("could not merge the dataset results, please review the input data", e)


def matching_original_values(df_match_data, df):
    try:
        df_match_data = df_match_data.rename(columns={"id_1": "id"})
        df = pd.merge(df, df_match_data, on=["id"], how="inner").drop(columns=["CPAIQ_MATCH_y"]).reset_index(drop=True)
        df = df[[df.columns[0], df.columns[3], df.columns[1], df.columns[2], df.columns[4], df.columns[5]]]
        for col in df.columns:
            if "CPAIQ_MATCH_x" in col:
                col_ = "CPAIQ_MATCH_x"
                df = df.rename(columns={col_: "CPAIQ_MATCH"})
        return df
    except Exception as e:
        print("can create the final dataset with the model results, please review the process or the input data", e)
