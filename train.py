import re

from numpy import mean
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd

# nltk.download()
from db.db import get_issues_from_to_id, create_db_connection

set(stopwords.words('english'))


def split_text(text):
    return re.split('[\n| ]', text)


def remove_stopwords1(words):
    all_stopwords = set(stopwords.words('english'))
    clean_words = [
        word for word in words if word not in all_stopwords and word]

    return clean_words


def lem_words(words):
    lem = WordNetLemmatizer()
    lemmed_words = [lem.lemmatize(word, pos="v") for word in words]
    return lemmed_words


def syn_words(words):
    # TODO
    pass


def _count_assigned_issues_at_creation(issues):
    def h(row):
        assignee = row["AssigneeLogin"]
        date_closed = row["ClosedAt"]
        ai = issues[issues["AssigneeLogin"] == assignee]
        work_issues = ai[(ai["CreatedAt"] < date_closed) &
                         (ai["ClosedAt"] > date_closed)].shape[0]
        return work_issues

    return h


def get_df():
    issues = get_issues_from_to_id(130000, 150000)
    # issues = issues[issues["Status"] == "closed"]
    df = pd.DataFrame.from_dict(issues)
    df["Text"] = df["Title"] + " " + df["Body"]
    df["CleanText"] = df["Text"].map(lambda text: " ".join(
        lem_words(remove_stopwords1(split_text(str(text))))))
    df["AssignedIssues"] = df.apply(
        _count_assigned_issues_at_creation(df), axis=1)
    df["MeanAssignedIssues"] = df.groupby(
        "AssigneeLogin")["AssignedIssues"].transform(mean)
    return df


def main(data):
    # tokenizer to remove unwanted elements from out data like symbols and numbers
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(lowercase=True, stop_words='english',
                         ngram_range=(1, 1), tokenizer=token.tokenize)
    text_counts = cv.fit_transform(data["text"])
    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data['AssigneeLogin'], test_size=0.3, random_state=1)

    clf = MultinomialNB().fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))


if __name__ == '__main__':
    issues = get_issues_from_to_id(140000, 150000)
    df = pd.DataFrame.from_dict(issues)
    df["Text"] = df["Title"] + " " + df["Body"]
    df["CleanText"] = df["Text"].map(lambda text: " ".join(
        lem_words(remove_stopwords1(split_text(str(text))))))
    print(df)
