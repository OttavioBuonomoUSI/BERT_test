from __future__ import annotations

import json
from os import listdir
from os.path import isfile, join
from db.db import create_db_connection, create_issue_table, insert_issues


# IssueId
# Owner
# Repo
# AssigneeLogin
# Title
# Body
# CreatorLogin
# Status
# ClosedAt
# CreatedAt


def read_issues(root_dir: str, owner: str, repo: str):
    directory = f"{root_dir}/{owner}/{repo}"
    filenames = [f for f in listdir(directory) if isfile(join(directory, f))]
    items = []
    for filename in filenames:
        with open(f"{directory}/{filename}", 'r', encoding="utf-8") as f:
            text = f.read()
            j_text = json.loads(text)
            for issue in j_text:
                assignees = issue.get("assignees")
                if len(assignees) != 1:
                    continue

                IssueId = issue.get("number")
                AssigneeLogin = assignees[0]["login"]
                Title: None | str = issue.get("title")
                Body:  str = issue.get("body")
                Body = "" if Body is None else Body
                CreatorLogin: None | str = issue["user"].get("login")
                Status: None | str = issue.get("state")
                ClosedAt: None | str = issue["closed_at"]
                CreatedAt: None | str = issue["created_at"]
                items.append(
                    {"Repo": repo, "Owner": owner, "IssueID": IssueId,
                     "AssigneeLogin": AssigneeLogin,
                     "Title": Title, "Body": Body,
                     "CreatorLogin": CreatorLogin, "Status": Status,
                     "ClosedAt": ClosedAt, "CreatedAt": CreatedAt})
    return items


def ingest(owner, repo):
    issues = read_issues(f"./repositories", owner, repo)
    create_issue_table()
    insert_issues(issues)


# -> remove stop -> synonyms ->  stem=
if __name__ == '__main__':
    issues = read_issues("./repositories", "microsoft", "vscode")
    print("Read issues")
    create_issue_table()
    print("Created Issue table")
    insert_issues(issues)
    print("Ingested issues")
