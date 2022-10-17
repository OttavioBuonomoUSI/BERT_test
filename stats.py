import os
from os import listdir
from os.path import isfile, join

from train import get_df

if __name__ == '__main__':
    df = get_df("microsoft", "vscode")
    print(df.head()["Title"])
    count = df.groupby(["AssigneeLogin"]).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

    print(count.head())
    acc = count.iloc[0]["counts"] / df.size

    print(count.iloc[0:5].sum(axis=0)['counts'])

    a = [(i, (count.iloc[0:i].sum(axis=0)['counts']/df.size)) for i in range(1, 100)]
    print(a)
