
import argparse

# --help

# ingest <github owner>/<github repo > : ingest given git repo if not already ingested
# ingest -l : list all the ingested repos
# ingest -d <github owner>/<github repo > : delete data about the ingested repo


# train -m=<model>  <github owner>/<github repo > : trains a given model on the repo
# train --all_models <github owner>/<github repo > : trains all models on a given repo
# train -d -m=<model>  <github owner>/<github repo > : delete training data
# train -l  <github owner>/<github repo >  : list all trained models

# evaluate -m=<model>  <github owner>/<github repo > : returns statistics about the given model


# predict -m=<model> <github owner>/<github repo > : returns a prediction of the most likely assignees


if __name__ == '__main__':
    pass