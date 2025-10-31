import json
import os
import requests
from ghapi.all import GhApi
from bs4 import BeautifulSoup
from pathlib import Path

from util import get_active_bugs, defects4j_project_name_url_map

# GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GITHUB_TOKEN = ''
BASE_DIR = Path(__file__).resolve().parents[2]
base_path = BASE_DIR / 'vendor' / "defects4j" / "framework" / "projects"


def main():
    active_bugs = get_active_bugs(base_path)
    for project_name, bug_ids in active_bugs.items():
        result = {}
        dir = BASE_DIR / 'dataset' / 'defects4j' / 'bug_description' / project_name
        if not os.path.exists(dir):
            os.makedirs(dir)
        bugs_description_file = dir / "bug_description.json"
        if os.path.exists(bugs_description_file):
            os.remove(bugs_description_file)

        # bug description link from active-bugs.csv
        active_bugs_csv_file = os.path.join(base_path, project_name, 'active-bugs.csv')
        bug_to_report_link = {}
        with open(active_bugs_csv_file, 'r') as f:
            # Skip header line
            next(f)
            for line in f:
                row = line.strip().split(',')
                # CSV format: bug.id,revision.id.buggy,revision.id.fixed,report.id,report.url
                if len(row) >= 5 and str(row[4]).startswith("https://"):
                    bug_id = row[0]
                    report_url = row[4]
                    bug_to_report_link[int(bug_id)] = report_url
        for bug_id in bug_ids:
            bug_description_link = bug_to_report_link.get(int(bug_id), "")
            description = ""
            desc_source = ""
            if 'https://github.com/' in bug_description_link:
                # github issue page: 8 projects
                description = mining_bug_desc_from_github_issue_page(
                    bug_description_link,
                    defects4j_project_name_url_map.get(project_name)
                )
                desc_source = "github_issue"

            elif 'https://issues.apache.org/jira/' in bug_description_link:
                # apache jira page: 7 projects
                description = mining_bug_desc_from_apache_jira(bug_description_link)
                desc_source = "jira"

            elif 'https://storage.googleapis.com/' in bug_description_link:
                # google json file
                description = mining_bug_desc_from_google_code_archive(bug_description_link)
                desc_source = "google"

            elif 'https://code.google.com/archive/' in bug_description_link:
                # google archive file
                description = mining_bug_desc_from_google_issue(bug_description_link)
                desc_source = "google issue"

            elif 'https://sourceforge.net' in bug_description_link:
                description = get_sourceforge_bug_details(bug_description_link)
                desc_source = "sourceforge"

            if bug_description_link == "" or description == "":
                description = ""
                desc_source = "commit_msg"

            result[f"{project_name}_{bug_id}"] = {'description': description, 'desc_source': desc_source}
            json.dump(result, open(bugs_description_file, 'w'), indent=2)


def mining_bug_desc_from_github_issue_page(github_issue_link, project_url):
    issue_number = github_issue_link.split('/')[-1]
    project_owner = project_url.split('/')[-2]
    project_name = project_url.split('/')[-1].split('.git')[0]
    api = GhApi(owner=project_owner, repo=project_name, token=GITHUB_TOKEN)
    try:
        project_issue = api.issues.get(issue_number)
        title = project_issue.title if project_issue.title else ""
        body = project_issue.body if project_issue.body else ""
        description = f"{title}\n{body}\n"
    except:
        print(f"Failed to retrieve github_issue_link. {github_issue_link}")
        description = ""
    return description


def mining_bug_desc_from_apache_jira(jira_link):
    # Extract the issue key from the URL
    issue_key = jira_link.rstrip('/').split('/')[-1]
    api_url = f'https://issues.apache.org/jira/rest/api/2/issue/{issue_key}?fields=summary,description'
    response = requests.get(api_url, headers={'Accept': 'application/json'})

    if response.status_code == 200:
        data = response.json()
        description = f"{data['fields']['summary']}\n{data['fields']['description']}"
    else:
        description = ""
        print(f"Failed to retrieve jira_link. {jira_link}")
    return description


def mining_bug_desc_from_google_code_archive(google_link):
    response = requests.get(google_link)
    if response.status_code == 200:
        data = response.json()
        title = data.get('summary', "").strip()
        comments = data.get('comments', [])
        comment_first = comments[0].get('content', "").strip() if comments else ""
        if not title and not comment_first:
            return ""
        description = f"{title}\n{comment_first}"

    else:
        print(f"Failed to retrieve google_link. {google_link}")
        description = ""
    return description


def mining_bug_desc_from_google_issue(google_issue_link):
    """
    Extract bug description from Google Code Archive issue pages.
    Example URL: https://code.google.com/archive/p/mockito/issues/484
    
    This function converts the archive URL to the corresponding JSON API URL
    and extracts the issue data directly.
    """
    try:
        parts = google_issue_link.split('/')
        if len(parts) < 8:
            print(f"Invalid Google Code Archive URL format: {google_issue_link}")
            return ""
        
        project_name = parts[5]  # mockito
        issue_number = parts[7]  # 188
        
        # Convert to JSON API URL (same pattern as google_code_archive function)
        json_url = f"https://storage.googleapis.com/google-code-archive/v2/code.google.com/{project_name}/issues/issue-{issue_number}.json"
        
        response = requests.get(json_url)
        if response.status_code == 200:
            data = response.json()
            title = data.get('summary', "").strip()
            comments = data.get('comments', [])
            comment_first = comments[0].get('content', "").strip() if comments else ""
            
            if not title and not comment_first:
                return ""
            description = f"{title}\n{comment_first}"
            return description
        else:
            print(f"Failed to retrieve JSON data for {google_issue_link}. Status: {response.status_code}")
            return ""
            
    except Exception as e:
        print(f"Error processing google_issue_link {google_issue_link}: {str(e)}")
        return ""



def get_sourceforge_bug_details(sourceforge_link):
    # Example usage
    # sourceforge_link = 'https://sourceforge.net/p/jfreechart/bugs/983'
    response = requests.get(sourceforge_link)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract the title
        issue_number_tag = soup.find('h2', class_='dark title')
        title = issue_number_tag.get_text().strip() if issue_number_tag else ''

        # Get the description from the <div id="ticket_content"> inside <div class="markdown_content">
        content_div = soup.find('div', {'id': 'ticket_content'})
        markdown_div = content_div.find('div', class_='markdown_content') if content_div else None
        content = markdown_div.get_text().strip() if markdown_div else ''

        if not title and not content:
            return ""
        description = f"{title}\n{content}"
    else:
        print(f"Failed to retrieve sourceforge_link. {sourceforge_link}")
        description = ""
    return description


if __name__ == '__main__':
    main()
