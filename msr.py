"""
Minimalist Review Metadata Mining Script
Mines PR review metadata from GitHub Archive via BigQuery and saves to HuggingFace dataset.
"""

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv
from google.cloud import bigquery

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/swe_agents"
REVIEW_METADATA_REPO = "SWE-Arena/review_metadata"
LEADERBOARD_TIME_FRAME_DAYS = 180  # 6 months

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_jsonl(filename):
    """Load JSONL file and return list of dictionaries."""
    if not os.path.exists(filename):
        return []

    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data


def save_jsonl(filename, data):
    """Save list of dictionaries to JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


def get_bigquery_client():
    """
    Initialize BigQuery client using credentials from environment variable.

    Expects GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable containing
    the service account JSON credentials as a string.
    """
    # Get the JSON content from environment variable
    creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')

    if creds_json:
        # Create a temporary file to store credentials
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            temp_file.write(creds_json)
            temp_path = temp_file.name

        # Set environment variable to point to temp file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_path

        # Initialize BigQuery client
        client = bigquery.Client()

        # Clean up temp file
        os.unlink(temp_path)

        return client
    else:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON not found in environment")


# =============================================================================
# BIGQUERY FUNCTIONS
# =============================================================================

def fetch_reviews_from_bigquery(client, identifier, start_date, end_date):
    """
    Fetch PR review events from GitHub Archive for a specific agent.

    Queries githubarchive.day.YYYYMMDD tables for PullRequestReviewEvent where
    actor.login matches the agent identifier.

    Args:
        client: BigQuery client instance
        identifier: GitHub username or bot identifier (e.g., 'amazon-inspector-beta[bot]')
        start_date: Start datetime (timezone-aware)
        end_date: End datetime (timezone-aware)

    Returns:
        List of review event rows with PR information
    """
    print(f"\n🔍 Querying BigQuery for reviews by {identifier}")
    print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Generate list of table names for each day in the range
    table_refs = []
    current_date = start_date
    while current_date < end_date:
        table_name = f"githubarchive.day.{current_date.strftime('%Y%m%d')}"
        table_refs.append(table_name)
        current_date += timedelta(days=1)

    # Build UNION ALL query for all daily tables
    union_parts = []
    for table_name in table_refs:
        union_parts.append(f"""
        SELECT
            repo.name as repo_name,
            actor.login as actor_login,
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') as pr_url,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') AS INT64) as pr_number,
            JSON_EXTRACT_SCALAR(payload, '$.review.submitted_at') as reviewed_at,
            created_at
        FROM `{table_name}`
        WHERE type = 'PullRequestReviewEvent'
        AND actor.login = @identifier
        """)

    query = " UNION ALL ".join(union_parts)

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("identifier", "STRING", identifier)
        ]
    )

    print(f"   Querying {len(table_refs)} daily tables...")

    try:
        query_job = client.query(query, job_config=job_config)
        results = list(query_job.result())

        print(f"   ✓ Found {len(results)} review events")
        return results

    except Exception as e:
        print(f"   ✗ BigQuery error: {str(e)}")
        return []


def fetch_pr_status_from_bigquery(client, pr_urls, start_date, end_date):
    """
    Fetch PR status (merged/closed) from GitHub Archive PullRequestEvent.

    For each PR URL, looks for PullRequestEvent with action='closed' to determine
    if the PR was merged or just closed.

    Args:
        client: BigQuery client instance
        pr_urls: List of PR URLs to check status for
        start_date: Start datetime (should cover review period and after)
        end_date: End datetime (should be recent/current)

    Returns:
        Dictionary mapping PR URL to status dict:
        {
            'pr_url': {
                'status': 'merged'|'closed'|'open',
                'merged': bool,
                'closed_at': timestamp or None
            }
        }
    """
    if not pr_urls:
        return {}

    print(f"\n🔍 Querying BigQuery for PR status ({len(pr_urls)} PRs)...")

    # Extract repo and PR number from URLs
    # URL format: https://github.com/owner/repo/pull/123
    pr_info = []
    for url in pr_urls:
        try:
            parts = url.replace('https://github.com/', '').split('/')
            if len(parts) >= 4:
                owner = parts[0]
                repo = parts[1]
                pr_number = int(parts[3])
                repo_name = f"{owner}/{repo}"
                pr_info.append({
                    'url': url,
                    'repo': repo_name,
                    'number': pr_number
                })
        except Exception as e:
            print(f"   Warning: Could not parse PR URL {url}: {e}")
            continue

    if not pr_info:
        return {}

    # Build repo filter condition for WHERE clause
    # Group PRs by repo to create efficient filters
    repos_to_prs = defaultdict(list)
    for pr in pr_info:
        repos_to_prs[pr['repo']].append(pr['number'])

    # Generate list of table names for date range
    # Look back 1 full year from end_date to catch PR close events that may have occurred before reviews
    pr_status_start = end_date - timedelta(days=365)
    table_refs = []
    current_date = pr_status_start
    while current_date < end_date:
        table_name = f"githubarchive.day.{current_date.strftime('%Y%m%d')}"
        table_refs.append(table_name)
        current_date += timedelta(days=1)

    # Build WHERE clause to filter by specific repos and PR numbers
    # Format: (repo='owner/repo1' AND pr_number IN (1,2,3)) OR (repo='owner/repo2' AND pr_number IN (4,5))
    filter_conditions = []
    for repo, pr_numbers in repos_to_prs.items():
        pr_list = ','.join(map(str, pr_numbers))
        filter_conditions.append(f"(repo.name = '{repo}' AND CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') AS INT64) IN ({pr_list}))")

    pr_filter = " OR ".join(filter_conditions)

    # Build query to find close/merge events for specific PRs
    union_parts = []
    for table_name in table_refs:
        union_parts.append(f"""
        SELECT
            repo.name as repo_name,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') AS INT64) as pr_number,
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') as pr_url,
            JSON_EXTRACT_SCALAR(payload, '$.action') as action,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') AS BOOL) as merged,
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.closed_at') as closed_at,
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at') as merged_at,
            created_at
        FROM `{table_name}`
        WHERE type = 'PullRequestEvent'
        AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'closed'
        AND ({pr_filter})
        """)

    query = " UNION ALL ".join(union_parts)

    print(f"   Querying {len(table_refs)} daily tables for PR status (1-year lookback: {pr_status_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")
    print(f"   Filtering for {len(pr_info)} specific PRs across {len(repos_to_prs)} repos")

    try:
        query_job = client.query(query)
        results = list(query_job.result())

        print(f"   ✓ Found {len(results)} PR close events")

        # Build status map by PR URL
        status_map = {}
        for row in results:
            pr_url = row.pr_url

            merged = row.merged if row.merged is not None else False
            closed_at = row.closed_at or row.merged_at

            # Convert to ISO format if datetime
            if hasattr(closed_at, 'isoformat'):
                closed_at = closed_at.isoformat()

            status = 'merged' if merged else 'closed'

            status_map[pr_url] = {
                'status': status,
                'merged': merged,
                'closed_at': closed_at
            }

        # Mark remaining PRs as open
        for url in pr_urls:
            if url not in status_map:
                status_map[url] = {
                    'status': 'open',
                    'merged': False,
                    'closed_at': None
                }

        merged_count = sum(1 for s in status_map.values() if s['merged'])
        closed_count = sum(1 for s in status_map.values() if s['status'] == 'closed')
        open_count = sum(1 for s in status_map.values() if s['status'] == 'open')

        print(f"   Status breakdown: {merged_count} merged, {closed_count} closed, {open_count} open")

        return status_map

    except Exception as e:
        print(f"   ✗ BigQuery error: {str(e)}")
        # Return all as open on error
        return {url: {'status': 'open', 'merged': False, 'closed_at': None} for url in pr_urls}


def extract_review_metadata(review_row, status_info):
    """
    Extract minimal PR review metadata from BigQuery row and status info.

    Args:
        review_row: BigQuery row from PullRequestReviewEvent query
        status_info: Status dictionary from fetch_pr_status_from_bigquery

    Returns:
        Dictionary with review metadata
    """
    pr_url = review_row.pr_url
    pr_number = review_row.pr_number
    reviewed_at = review_row.reviewed_at or review_row.created_at

    # Convert to ISO format if datetime
    if hasattr(reviewed_at, 'isoformat'):
        reviewed_at = reviewed_at.isoformat()

    return {
        'html_url': pr_url,
        'reviewed_at': reviewed_at,
        'pr_status': status_info['status'],
        'pr_merged': status_info['merged'],
        'pr_closed_at': status_info['closed_at'],
        'pr_url': pr_url,
        'review_id': f"pr_{pr_number}"
    }


def fetch_all_reviews_metadata(identifier, agent_name):
    """
    Fetch PR reviews associated with a GitHub user or bot for the past LEADERBOARD_TIME_FRAME_DAYS.
    Uses BigQuery to query GitHub Archive instead of GitHub API.

    Args:
        identifier: GitHub username or bot identifier (for BigQuery queries)
        agent_name: Human-readable name of the agent (for display only)

    Returns:
        List of dictionaries containing minimal PR review metadata with PR status
    """
    # Initialize BigQuery client
    try:
        client = get_bigquery_client()
    except Exception as e:
        print(f"✗ Failed to initialize BigQuery client: {str(e)}")
        return []

    # Define time range: past LEADERBOARD_TIME_FRAME_DAYS (excluding today)
    current_time = datetime.now(timezone.utc)
    end_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)  # 12:00 AM UTC today
    start_date = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    print(f"\n{'='*80}")
    print(f"Fetching reviews for: {agent_name} ({identifier})")
    print(f"{'='*80}")

    # Fetch review events from BigQuery
    review_rows = fetch_reviews_from_bigquery(client, identifier, start_date, end_date)

    if not review_rows:
        print(f"   No reviews found for {identifier}")
        return []

    # Extract unique PR URLs
    pr_urls = list(set([row.pr_url for row in review_rows if row.pr_url]))
    print(f"\n📊 Found {len(review_rows)} review events across {len(pr_urls)} unique PRs")

    # Fetch PR status from BigQuery
    # Use extended end date to catch recent merges/closes
    extended_end_date = current_time
    status_map = fetch_pr_status_from_bigquery(client, pr_urls, start_date, extended_end_date)

    # Extract metadata for each review
    print(f"\n📦 Extracting metadata...")
    metadata_list = []

    # Deduplicate by PR URL (multiple reviews on same PR)
    seen_prs = set()
    for row in review_rows:
        pr_url = row.pr_url
        if pr_url in seen_prs:
            continue
        seen_prs.add(pr_url)

        status_info = status_map.get(pr_url, {
            'status': 'open',
            'merged': False,
            'closed_at': None
        })

        metadata = extract_review_metadata(row, status_info)
        metadata_list.append(metadata)

    print(f"   ✓ Extracted {len(metadata_list)} unique PR review records")

    return metadata_list


def fetch_all_reviews_metadata_batch(agents):
    """
    Fetch PR reviews for ALL agents in a single batch operation.
    Uses only 2 BigQuery queries total (instead of 2*N queries for N agents).

    Args:
        agents: List of agent dictionaries with 'github_identifier' and 'name' fields

    Returns:
        Dictionary mapping agent identifier to list of review metadata:
        {
            'agent-identifier': [metadata_list],
            ...
        }
    """
    if not agents:
        return {}

    # Initialize BigQuery client
    try:
        client = get_bigquery_client()
    except Exception as e:
        print(f"✗ Failed to initialize BigQuery client: {str(e)}")
        return {}

    # Define time range: past LEADERBOARD_TIME_FRAME_DAYS (excluding today)
    current_time = datetime.now(timezone.utc)
    end_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    # Extract all identifiers
    identifiers = [agent['github_identifier'] for agent in agents if agent.get('github_identifier')]
    if not identifiers:
        return {}

    print(f"\n🚀 BATCH MODE: Fetching reviews for {len(identifiers)} agents in 2 queries")
    print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # =========================================================================
    # QUERY 1: Fetch ALL review events for ALL agents in one query
    # =========================================================================
    print(f"\n🔍 Query 1/2: Fetching ALL review events...")

    # Generate list of table names
    table_refs = []
    current_date = start_date
    while current_date < end_date:
        table_name = f"githubarchive.day.{current_date.strftime('%Y%m%d')}"
        table_refs.append(table_name)
        current_date += timedelta(days=1)

    # Build IN clause for all identifiers
    identifier_list = ', '.join([f"'{id}'" for id in identifiers])

    # Build UNION ALL query for all daily tables
    union_parts = []
    for table_name in table_refs:
        union_parts.append(f"""
        SELECT
            repo.name as repo_name,
            actor.login as actor_login,
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') as pr_url,
            CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') AS INT64) as pr_number,
            JSON_EXTRACT_SCALAR(payload, '$.review.submitted_at') as reviewed_at,
            created_at
        FROM `{table_name}`
        WHERE type = 'PullRequestReviewEvent'
        AND actor.login IN ({identifier_list})
        """)

    query = " UNION ALL ".join(union_parts)

    print(f"   Querying {len(table_refs)} daily tables...")

    try:
        query_job = client.query(query)
        all_review_rows = list(query_job.result())
        print(f"   ✓ Found {len(all_review_rows)} total review events")
    except Exception as e:
        print(f"   ✗ BigQuery error: {str(e)}")
        return {}

    # Group reviews by agent
    reviews_by_agent = defaultdict(list)
    all_pr_urls = set()
    for row in all_review_rows:
        reviews_by_agent[row.actor_login].append(row)
        if row.pr_url:
            all_pr_urls.add(row.pr_url)

    print(f"   📊 Reviews found for {len(reviews_by_agent)} agents")
    print(f"   📊 {len(all_pr_urls)} unique PRs to check status for")

    # =========================================================================
    # QUERY 2: Fetch ALL PR statuses in one query
    # =========================================================================
    if all_pr_urls:
        print(f"\n🔍 Query 2/2: Fetching ALL PR statuses...")
        extended_end_date = current_time
        status_map = fetch_pr_status_from_bigquery(client, list(all_pr_urls), start_date, extended_end_date)
    else:
        status_map = {}

    # =========================================================================
    # Post-process: Build metadata for each agent
    # =========================================================================
    print(f"\n📦 Processing metadata for each agent...")
    results = {}

    for agent in agents:
        identifier = agent.get('github_identifier')
        if not identifier or identifier not in reviews_by_agent:
            results[identifier] = []
            continue

        review_rows = reviews_by_agent[identifier]

        # Deduplicate by PR URL
        metadata_list = []
        seen_prs = set()
        for row in review_rows:
            pr_url = row.pr_url
            if pr_url in seen_prs:
                continue
            seen_prs.add(pr_url)

            status_info = status_map.get(pr_url, {
                'status': 'open',
                'merged': False,
                'closed_at': None
            })

            metadata = extract_review_metadata(row, status_info)
            metadata_list.append(metadata)

        results[identifier] = metadata_list
        print(f"   ✓ {agent.get('name', identifier)}: {len(metadata_list)} unique PRs")

    return results


# =============================================================================
# HUGGINGFACE STORAGE FUNCTIONS
# =============================================================================

def group_metadata_by_date(metadata_list):
    """
    Group review metadata by exact date (year.month.day) for efficient daily storage.
    Returns dict: {(year, month, day): [metadata_list]}
    """
    grouped = defaultdict(list)

    for review_meta in metadata_list:
        reviewed_at = review_meta.get('reviewed_at')
        if not reviewed_at:
            continue

        try:
            dt = datetime.fromisoformat(reviewed_at.replace('Z', '+00:00'))
            key = (dt.year, dt.month, dt.day)
            grouped[key].append(review_meta)
        except Exception as e:
            print(f"Warning: Could not parse date '{reviewed_at}': {e}")

    return dict(grouped)


def save_review_metadata_to_hf(metadata_list, agent_identifier):
    """
    Save review metadata to HuggingFace dataset, organized by [agent_identifier]/YYYY.MM.DD.jsonl.
    Each file is stored in the agent's folder and named YYYY.MM.DD.jsonl for that day's reviews.

    This function APPENDS new metadata and DEDUPLICATES by review_id.
    Uses batch upload to avoid rate limit (uploads entire folder in single commit).

    Args:
        metadata_list: List of review metadata dictionaries
        agent_identifier: GitHub identifier of the agent (used as folder name)
    """
    import shutil

    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi()

        # Group by exact date (year, month, day)
        grouped = group_metadata_by_date(metadata_list)

        # Create a temporary directory for batch upload
        temp_dir = tempfile.mkdtemp()
        agent_folder = os.path.join(temp_dir, agent_identifier)
        os.makedirs(agent_folder, exist_ok=True)

        try:
            print(f"📦 Preparing batch upload for {len(grouped)} daily files...")

            # Process each daily file
            for (review_year, month, day), day_metadata in grouped.items():
                filename = f"{agent_identifier}/{review_year}.{month:02d}.{day:02d}.jsonl"
                local_filename = os.path.join(agent_folder, f"{review_year}.{month:02d}.{day:02d}.jsonl")

                # Download existing file if it exists
                existing_metadata = []
                try:
                    file_path = hf_hub_download(
                        repo_id=REVIEW_METADATA_REPO,
                        filename=filename,
                        repo_type="dataset",
                        token=token
                    )
                    existing_metadata = load_jsonl(file_path)
                    print(f"   Found {len(existing_metadata)} existing reviews in {filename}")
                except Exception:
                    print(f"   Creating new file: {filename}")

                # Merge and deduplicate by review_id
                existing_by_id = {meta['review_id']: meta for meta in existing_metadata if meta.get('review_id')}
                new_by_id = {meta['review_id']: meta for meta in day_metadata if meta.get('review_id')}

                # Update with new data (new data overwrites old)
                existing_by_id.update(new_by_id)
                merged_metadata = list(existing_by_id.values())

                # Save to temp directory
                save_jsonl(local_filename, merged_metadata)
                print(f"   Prepared {len(merged_metadata)} reviews for {filename}")

            # Upload entire folder in a single commit
            print(f"📤 Uploading {len(grouped)} files in single batch commit...")
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=REVIEW_METADATA_REPO,
                repo_type="dataset",
                token=token,
                commit_message=f"Batch update: {agent_identifier} ({len(grouped)} daily files)"
            )
            print(f"   ✓ Batch upload complete for {agent_identifier}")

            return True

        finally:
            # Always clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"✗ Error saving review metadata: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def load_agents_from_hf():
    """
    Load all agent metadata JSON files from HuggingFace dataset.

    The github_identifier is extracted from the filename (e.g., 'agent-name[bot].json' -> 'agent-name[bot]')
    """
    try:
        api = HfApi()
        agents = []

        # List all files in the repository
        files = api.list_repo_files(repo_id=AGENTS_REPO, repo_type="dataset")

        # Filter for JSON files only
        json_files = [f for f in files if f.endswith('.json')]

        print(f"Found {len(json_files)} agent files in {AGENTS_REPO}")

        # Download and parse each JSON file
        for json_file in json_files:
            try:
                file_path = hf_hub_download(
                    repo_id=AGENTS_REPO,
                    filename=json_file,
                    repo_type="dataset"
                )

                with open(file_path, 'r') as f:
                    agent_data = json.load(f)

                    # Extract github_identifier from filename (remove .json extension)
                    github_identifier = json_file.replace('.json', '')
                    agent_data['github_identifier'] = github_identifier

                    agents.append(agent_data)

            except Exception as e:
                print(f"Warning: Could not load {json_file}: {str(e)}")
                continue

        print(f"✓ Loaded {len(agents)} agents from HuggingFace")
        return agents

    except Exception as e:
        print(f"Could not load agents from HuggingFace: {str(e)}")
        return []


# =============================================================================
# MAIN MINING FUNCTION
# =============================================================================

def mine_all_agents():
    """
    Mine review metadata for all agents within LEADERBOARD_TIME_FRAME_DAYS and save to HuggingFace.
    Uses BigQuery to query GitHub Archive with batch processing (only 2 queries for all agents).
    """
    # Load agent metadata from HuggingFace
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found in HuggingFace dataset")
        return

    print(f"\n{'='*80}")
    print(f"Starting review metadata mining for {len(agents)} agents")
    print(f"Time frame: Last {LEADERBOARD_TIME_FRAME_DAYS} days")
    print(f"Data source: BigQuery + GitHub Archive (BATCH MODE)")
    print(f"{'='*80}\n")

    # Fetch ALL reviews for ALL agents in batch (only 2 BigQuery queries total!)
    try:
        all_metadata = fetch_all_reviews_metadata_batch(agents)
    except Exception as e:
        print(f"✗ Error during batch fetch: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # Save results for each agent
    print(f"\n{'='*80}")
    print(f"💾 Saving results to HuggingFace...")
    print(f"{'='*80}\n")

    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('name', agent.get('agent_name', 'Unknown'))

        if not identifier:
            print(f"Warning: Skipping agent without identifier: {agent}")
            continue

        metadata = all_metadata.get(identifier, [])

        try:
            if metadata:
                print(f"💾 {agent_name}: Saving {len(metadata)} review records...")
                save_review_metadata_to_hf(metadata, identifier)
                print(f"   ✓ Successfully saved")
            else:
                print(f"   No reviews found for {agent_name}")

        except Exception as e:
            print(f"✗ Error saving {identifier}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✅ Mining complete for all agents")
    print(f"{'='*80}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mine_all_agents()
