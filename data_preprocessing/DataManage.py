from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from numpy import tensordot
import pandas as pd
from tabulate import tabulate
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from typing import Union
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def variables_type_check(df):

    print(
        f"\n\n- Are these the correct types for the feature? Press 'n' for no or just enter.\n\n{df.dtypes}")

    answer = input()
    if answer == 'n':
        table = []
        counter = 1

        print("\n- Are these features all categorial? Press 'n' for no or just enter.")
        for column in df.columns:
            if df[column].dtype != "float64":
                table.append([counter, column, df[column].dtype])
                counter += 1
        print(tabulate(table))
        answer = input()
        if answer == 'n':
            print("\nEnter the index or indexes comma separated")
            answer = input()
            incorrect_types = re.findall(r'\d+', answer)
            if len(incorrect_types) != 1:
                for i in [incorrect_types]:
                    df[table[int(i)-1][1]] = df[table[int(i)-1]
                                                [1]].astype('float')
            else:
                df[table[int(incorrect_types[0])-1][1]
                   ] = df[table[int(incorrect_types[0])-1][1]].astype('float')


def random_split(df: pd.DataFrame,
                 label_col_name: str,
                 val_percentage: float,
                 test_percentage: float,
                 features_name: list,
                 random_state: int = 1):

    val_relative_percentage = val_percentage/(1 - 0.33)

    X_train, X_test, y_train, y_test = train_test_split(df[features_name], df[label_col_name],
                                                        test_size=test_percentage,
                                                        random_state=random_state)

    X_train, X_val, y_train,  y_val = train_test_split(X_train, y_train,
                                                       test_size=val_relative_percentage,
                                                       random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


class DataManage:
    def __init__(self,
                 data_frame: pd.DataFrame,
                 text_col_name: str = "CleanText",
                 label_col_name: str = "AssegneeLogin",
                 random_split_method: bool = False,
                 test_percentage: float = 20,
                 val_percentage: float = 20,
                 ):

        self.DataFrame = data_frame
        self.text_col_name = text_col_name
        self.text = self.DataFrame[text_col_name]
        self.label_col_name = label_col_name
        self.test_percentage = test_percentage
        self.val_percentage = val_percentage
        self.random_split_method = random_split_method
        self.features_name = [
            feature for feature in self.DataFrame.columns if feature != self.label_col_name]

        self.numerical_encoder = None
        self.categorical_encoder = None
        self.label_encoder = None

        self.info = {}
        self.X_train_pp = None
        self.X_val_pp = None
        self.X_test_pp = None

    def split(self, type_check: bool = False, show_info: bool = False):

        if type_check:
            variables_type_check(self.DataFrame)

        if self.random_split_method:
            X_train, X_val, X_test, y_train, y_val, y_test = random_split(
                df=self.DataFrame,
                label_col_name=self.label_col_name,
                val_percentage=self.val_percentage,
                test_percentage=self.test_percentage,
                features_name=self.features_name)

        else:
            df_shape = self.DataFrame.shape[0]

            test_border = (df_shape * self.test_percentage//100)
            val_border = (df_shape * self.val_percentage//100) + test_border

            X_test = self.DataFrame[self.features_name].iloc[-test_border:]
            y_test = self.DataFrame[self.label_col_name].iloc[-test_border:]

            X_val = self.DataFrame[self.features_name].iloc[-val_border:-test_border]
            y_val = self.DataFrame[self.label_col_name].iloc[-val_border:-test_border]

            X_train = self.DataFrame[self.features_name].iloc[:-val_border]
            y_train = self.DataFrame[self.label_col_name].iloc[:-val_border]

        for i in [X_train, X_val, X_test]:
            i.columns = self.features_name
        for i in [y_train, y_val, y_test]:
            i.columns = self.label_col_name

        if show_info:
            print(
                f"\n\n----- TRAIN -----\nX)\n{tabulate(X_train.head())}\ny)\n{tabulate([y_train])}\n\n\n----- VAL -----\nX)\n{tabulate(X_val.head())}\ny)\n{tabulate([y_val.head()])}\n\n----- TEST -----\nX)\n{tabulate(X_test.head())}\ny)\n{tabulate([y_test.head()])}")

            train_labels_set = set(y_train)
            val_labels_set = set(y_val)
            test_labels_set = set(y_test)

            missing_labels_vs_val = list(
                val_labels_set.difference(train_labels_set))
            missing_labels_vs_test = list(
                test_labels_set.difference(train_labels_set))

            count_occurency_missing_label_vs_val = 0
            for missing_label_vs_val in missing_labels_vs_val:
                occurencies = y_val.value_counts()[missing_label_vs_val]
                count_occurency_missing_label_vs_val += occurencies

            count_occurency_missing_label_vs_test = 0
            for missing_label_vs_test in missing_labels_vs_test:
                occurencies = y_test.value_counts()[missing_label_vs_test]
                count_occurency_missing_label_vs_test += occurencies

            max_val_accuracy_achievable = 1 - \
                (count_occurency_missing_label_vs_val / len(y_val))
            max_val_accuracy_achievable = "{:.2%}".format(
                max_val_accuracy_achievable)

            max_test_accuracy_achievable = 1 - (count_occurency_missing_label_vs_test /
                                                len(y_test))
            max_test_accuracy_achievable = "{:.2%}".format(
                max_test_accuracy_achievable)

            print("\n\n-----------------------------------------------")
            print(
                f"- Maximum now achievable accuracy on val set: {max_val_accuracy_achievable}")
            print(
                f"- Maximum now achievable accuracy on test set: {max_test_accuracy_achievable}")
            print("-----------------------------------------------\n\n")

        self.X_train_text = X_train[self.text_col_name]
        self.X_val_text = X_val[self.text_col_name]
        self.X_test_text = X_test[self.text_col_name]

        self.X_train_text_pp = None
        self.X_val_text_pp = None
        self.X_test_text_pp = None

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.info["target_label_count"] = len(
            list(set(y_train)))  # Store for the model

    def get_split(self):
        if not self.train:
            self.split()
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def transform(self,
                  X_train: Union[pd.DataFrame, None] = None,
                  X_val: Union[pd.DataFrame, None] = None,
                  X_test: Union[pd.DataFrame, None] = None,
                  y_train: Union[pd.DataFrame, None] = None,
                  y_val: Union[pd.DataFrame, None] = None,
                  y_test: Union[pd.DataFrame, None] = None):

        if X_train == None:
            X_train = self.X_train
            X_val = self.X_val
            X_test = self.X_test
            y_train = pd.DataFrame(self.y_train)
            y_val = pd.DataFrame(self.y_val)
            y_test = pd.DataFrame(self.y_test)

        # LabelEncoder for labels
        lencoder = OneHotEncoder(handle_unknown='ignore')
        # y_train.columns = ['Labels']
        print("y", y_train)
        print("type y_train", type(y_train))

        lencoder.fit(y_train)
        self.label_encoder = lencoder

        self.y_train_pp = pd.DataFrame(lencoder.transform(y_train).toarray())
        self.y_val_pp = pd.DataFrame(lencoder.transform(y_val).toarray())
        self.y_test_pp = pd.DataFrame(lencoder.transform(y_test).toarray())

        label_feature_names_pp = lencoder.get_feature_names_out()
        for X_cat in [self.y_train_pp, self.y_val_pp, self.y_test_pp]:
            X_cat.columns = label_feature_names_pp

        # self.y_train_pp.columns = self.y_val_pp.columns = self.y_test_pp.columns = [
        #     self.label_col_name]

        # Columns types
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric_feature_names = list(
            X_train.select_dtypes(include=numerics).columns)

        excluded_from_categorical = numeric_feature_names + \
            list([self.label_col_name]) + list([self.text_col_name])
        categorical_feature_names = [
            feature for feature in list(X_train.columns) if feature not in excluded_from_categorical]

        X_train_num = pd.DataFrame()
        X_train_cat = pd.DataFrame()

        if len(numeric_feature_names) > 0:
            # MinMaxScaler for numerical features
            # Fit on train

            scaler = MinMaxScaler()
            scaler.fit(X_train[numeric_feature_names])
            self.numerical_encoder = scaler

            # Transform
            X_train_num = pd.DataFrame(
                scaler.transform(X_train[numeric_feature_names]))
            X_val_num = pd.DataFrame(
                scaler.transform(X_val[numeric_feature_names]))
            X_test_num = pd.DataFrame(
                scaler.transform(X_test[numeric_feature_names]))

            for X_num in [X_train_num, X_val_num, X_test_num]:
                X_num.columns = numeric_feature_names

        if len(categorical_feature_names) > 0:
            # OneHotEncoding for categorial features
            # OneHotEncoding on train
            ohe = OneHotEncoder(
                handle_unknown='ignore', sparse=False)
            ohe.fit(X_train[categorical_feature_names])
            self.categorical_encoder = ohe

            # Transform
            X_train_cat = pd.DataFrame(ohe.transform(
                X_train[categorical_feature_names]))
            X_val_cat = pd.DataFrame(ohe.transform(
                X_val[categorical_feature_names]))
            X_test_cat = pd.DataFrame(ohe.transform(
                X_test[categorical_feature_names]))

            categorical_feature_names_pp = ohe.get_feature_names_out()
            for X_cat in [X_train_cat, X_val_cat, X_test_cat]:
                X_cat.columns = categorical_feature_names_pp

        # join
        if (len(numeric_feature_names) > 0) and (len(categorical_feature_names) > 0):
            self.X_train_pp = X_train_num.join(
                X_train_cat, lsuffix="_num", rsuffix="_cat")
            self.X_val_pp = X_val_num.join(
                X_val_cat, lsuffix="_num", rsuffix="_cat")
            self.X_test_pp = X_test_num.join(
                X_test_cat, lsuffix="_num", rsuffix="_cat")
            self.info["numerical_entries_count"] = X_train_num.shape[1]
            self.info["categorical_entries_count"] = X_train_cat.shape[1]
            self.info["additional_input_count"] = self.info["numerical_entries_count"] + \
                self.info["categorical_entries_count"]

        elif (len(numeric_feature_names) > 0):
            self.X_train_pp = X_train_num
            self.X_val_pp = X_val_num
            self.X_test_pp = X_test_num
            self.info["additional_input_count"] = self.info["numerical_entries_count"] = X_train_num.shape[
                1]
            self.info["categorical_entries_count"] = None

        elif (len(categorical_feature_names) > 0):
            self.X_train_pp = X_train_cat
            self.X_val_pp = X_val_cat
            self.X_test_pp = X_test_cat
            self.info["additional_input_count"] = self.info["categorical_entries_count"] = X_train_cat.shape[
                1]
            self.info["numerical_entries_count"] = None

        else:
            self.info["additional_input_count"] = None

    def get_trasform(self):
        if not self.label_encoder:
            self.transform()
        return self.X_train_pp, self.X_val_pp, self.X_test_pp, self.y_train_pp, self.y_val_pp, self.y_test_pp

    def remove_html_tags(self):
        """Remove html tags from a string"""
        clean = re.compile('<.*?>')
        if type(self.X_train_text_pp) != pd.Series:
            self.X_train_text_pp = self.X_train_text.apply(lambda x: re.sub(clean, '', x))
            self.X_val_text_pp = self.X_val_text.apply(lambda x: re.sub(clean, '', x))
            self.X_test_text_pp = self.X_test_text.apply(lambda x: re.sub(clean, '', x))
        else:
            self.X_train_text_pp = self.X_train_text_pp.apply(lambda x: re.sub(clean, '', x))
            self.X_val_text_pp = self.X_val_text_pp.apply(lambda x: re.sub(clean, '', x))
            self.X_test_text_pp = self.X_test_text_pp.apply(lambda x: re.sub(clean, '', x))


    def remove_special_characters(self):

        if type(self.X_train_text_pp) != pd.Series:
            self.X_train_text_pp = self.X_train_text.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
            self.X_val_text_pp = self.X_val_text.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
            self.X_test_text_pp = self.X_test_text.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
        else:
            self.X_train_text_pp = self.X_train_text_pp.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
            self.X_val_text_pp = self.X_val_text_pp.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
            self.X_test_text_pp = self.X_test_text_pp.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

    def remove_stopwords(self):
        stop_words = set(stopwords.words('english'))

        if type(self.X_train_text_pp) != pd.Series:

            self.X_train_text_pp = self.X_train_text.apply(lambda x: [item for item in str(
                x).split() if item not in stop_words])

            self.X_val_text_pp = self.X_val_text.apply(lambda x: [item for item in str(
                x).split() if item not in stop_words])

            self.X_test_text_pp = self.X_test_text.apply(lambda x: [item for item in str(
                x).split() if item not in stop_words])

        else:
            self.X_train_text_pp.update(self.X_train_text_pp.apply(lambda x: [item for item in str(
                x).split() if item not in stop_words]))

            self.X_val_text_pp.update(self.X_val_text_pp.apply(lambda x: [item for item in str(
                x).split() if item not in stop_words]))

            self.X_test_text_pp.update(self.X_test_text_pp.apply(lambda x: [item for item in str(
                x).split() if item not in stop_words]))

    # Low case
    def to_lowercase(self):

        if type(self.X_train_text_pp) != pd.Series:

            self.X_train_text_pp = self.X_train_text.apply(lambda x: [item.lower() for item in str(
                x).split()])

            self.X_val_text_pp = self.X_val_text.apply(lambda x: [item.lower() for item in str(
                x).split()])

            self.X_test_text_pp = self.X_test_text.apply(lambda x: [item.lower() for item in str(
                x).split()])

        else:
            self.X_train_text_pp.update(self.X_train_text_pp.apply(lambda x: [item.lower() for item in x]))

            self.X_val_text_pp.update(self.X_val_text_pp.apply(lambda x: [item.lower() for item in x]))

            self.X_test_text_pp.update(self.X_test_text_pp.apply(lambda x: [item.lower() for item in x]))

    # Stemming

    def stemming(self):
        ps = PorterStemmer()
        if type(self.X_train_text_pp) != pd.Series:

            self.X_train_text_pp = self.X_train_text.apply(
                lambda x: [ps.stem(item) for item in x])

            self.X_val_text_pp = self.X_val_text.apply(
                lambda x: [ps.stem(item) for item in x])

            self.X_test_text_pp = self.X_test_text.apply(
                lambda x: [ps.stem(item) for item in x])

        else:
            self.X_train_text_pp.update(self.X_train_text_pp.apply(
                lambda x: [ps.stem(item) for item in x]))

            self.X_val_text_pp.update(self.X_val_text_pp.apply(
                lambda x: [ps.stem(item) for item in x]))

            self.X_test_text_pp.update(self.X_test_text_pp.apply(
                lambda x: [ps.stem(item) for item in x]))

    def lemmatization(self, pos="n"):
        lemmatizer = WordNetLemmatizer()
        if type(self.X_train_text_pp) != pd.Series:

            self.X_train_text_pp = self.X_train_text.apply(
                lambda x: [lemmatizer.lemmatize(item) for item in x.split()])

            self.X_val_text_pp = self.X_val_text.apply(
                lambda x: [lemmatizer.lemmatize(item) for item in x.split()])

            self.X_test_text_pp = self.X_test_text.apply(
                lambda x: [lemmatizer.lemmatize(item) for item in x.split()])

        else:
            self.X_train_text_pp.update(self.X_train_text_pp.apply(
                lambda x: [lemmatizer.lemmatize(item) for item in x]))

            self.X_val_text_pp.update(self.X_val_text_pp.apply(
                lambda x: [lemmatizer.lemmatize(item) for item in x]))

            self.X_test_text_pp.update(self.X_test_text_pp.apply(
                lambda x: [lemmatizer.lemmatize(item) for item in x]))
