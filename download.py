import os
from os import listdir
from os.path import isfile, join
import requests
import pathlib


def download_github_repo_issues(startpage):
    pass


def download_github_repo_issue_page(owner, project, page):
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("missing auth token")

    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://api.github.com/repos/{owner}/{project}/issues?state=all&sort=created&direction=asc&per_page=100&page={page}"
    return requests.get(url, headers=headers)


def main(owner: str, repository: str) -> None:
    directory: str = f"./repositories/{owner}/{repository}"
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    page_count = len([f for f in listdir(directory) if isfile(join(directory, f))]) or 1
    max_page = 2000
    # TODO limit requests to 5000/h
    # max_request_ratio = 5000
    for page in range(page_count, max_page):
        r = download_github_repo_issue_page("microsoft", "vscode", page)
        text = r.text
        if r.status_code != 200:
            raise Exception("failure to make request")
        # don't parse JSON for efficiency
        if text == "[]":
            print(f"finished downloading")
            break
        print(f"downloaded page {page}")

        with open(f"{directory}/page_{page}.json", 'w') as f:
            f.write(text)



if __name__ == '__main__':
    main(owner="microsoft", repository="vscode")
