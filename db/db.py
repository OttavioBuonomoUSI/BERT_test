import os
import sqlite3


def create_db_connection() -> sqlite3.Connection:
    con = sqlite3.connect(
        "db/project.db")

    con.row_factory = sqlite3.Row
    return con


connection = create_db_connection()
print("connected to DB")


def create_issue_table():
    query = '''
    CREATE TABLE IF NOT EXISTS Issues (
    IssueId integer primary key,
    Owner  text NOT NULL,
    Repo  text NOT NULL,
    AssigneeLogin   text NOT NULL,
    Title  text NOT NULL,
    Body  text NOT NULL,
    CreatorLogin text NOT NULL,
    Status  text NOT NULL,
    ClosedAt  text,
    CreatedAt text NOT NULL
);'''
    res = connection.execute(query, [])
    return res


#  items.append((issue_id, assignee_login, title, body, creator))
def insert_issues(issues):
    # query = f'INSERT INTO Issues VALUES ({("?," * len(issues[0]))[:-1]});'
    query = """INSERT INTO 
    Issues (IssueID, Owner, Repo, AssigneeLogin, Title, Body, CreatorLogin, Status, ClosedAt, CreatedAt) 
    VALUES (:IssueID, :Owner, :Repo, :AssigneeLogin, :Title, :Body, :CreatorLogin, :Status, :ClosedAt, :CreatedAt) """
    cursor = connection.cursor()
    cursor.executemany(query, issues)
    connection.commit()


def count_issues_by_assignee():
    query = "SELECT Issues.AssigneeLogin, count(AssigneeLogin) FROM Issues GROUP BY AssigneeLogin"
    return connection.cursor().execute(query, []).fetchall()


def get_issues_from_to_id(owner, repo, start_id, end_id):
    query = "SELECT * FROM Issues WHERE  IssueId >= ? AND IssueId < ?"
    res = connection.cursor().execute(query, [start_id, end_id]).fetchall()
    res = [dict(row) for row in res]
    return res


def drop_repo_issues(owner, repo):
    query = "DELETE FROM ISSUES WHERE  owner=? AND repo=? "
    connection.cursor().execute(query, [owner, repo]).fetchall()
    connection.commit()


if __name__ == '__main__':

    print(get_issues_from_to_id(0, 10))
