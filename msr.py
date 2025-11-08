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
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv
from google.cloud import bigquery
import backoff

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/agent_metadata"
REVIEW_METADATA_REPO = "SWE-Arena/review_metadata"
LEADERBOARD_REPO = "SWE-Arena/leaderboard_metadata"  # HuggingFace dataset for leaderboard data
LEADERBOARD_TIME_FRAME_DAYS = 180  # Time frame for leaderboard

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


def normalize_date_format(date_string):
    """
    Convert date strings to standardized ISO 8601 format with Z suffix.
    Handles both 'T' and space-separated datetime formats (including newlines).
    Examples:
    - 2025-10-15T23:23:47.983068 -> 2025-10-15T23:23:47Z
    - 2025-06-17 21:21:07+00 -> 2025-06-17T21:21:07Z
    """
    if not date_string or date_string == 'N/A':
        return 'N/A'

    try:
        import re
        # Remove all whitespace (spaces, newlines, tabs) and replace with single space
        date_string = re.sub(r'\s+', ' ', date_string.strip())

        # Replace space with 'T' for ISO format compatibility
        date_string = date_string.replace(' ', 'T')

        # Fix incomplete timezone offset (+00 or -00 -> +00:00 or -00:00)
        # Check if timezone offset exists and is incomplete
        if len(date_string) >= 3:
            if date_string[-3:-2] in ('+', '-') and ':' not in date_string[-3:]:
                date_string = date_string + ':00'

        # Parse the date string (handles both with and without microseconds)
        dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))

        # Convert to standardized format
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        print(f"Warning: Could not parse date '{date_string}': {e}")
        return date_string


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


# =============================================================================
# HUGGINGFACE API WRAPPERS WITH BACKOFF
# =============================================================================

def is_rate_limit_error(e):
    """Check if exception is a HuggingFace rate limit error (429)."""
    if isinstance(e, HfHubHTTPError):
        return e.response.status_code == 429
    return False


@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    max_tries=8,
    giveup=lambda e: not is_rate_limit_error(e),
    on_backoff=lambda details: print(
        f"⏳ Rate limited. Retrying in {details['wait']:.1f}s (attempt {details['tries']}/8)..."
    )
)
def upload_large_folder_with_backoff(api, **kwargs):
    """Wrapper for api.upload_large_folder() with exponential backoff for rate limits."""
    return api.upload_large_folder(**kwargs)


@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    max_tries=8,
    giveup=lambda e: not is_rate_limit_error(e),
    on_backoff=lambda details: print(
        f"⏳ Rate limited. Retrying in {details['wait']:.1f}s (attempt {details['tries']}/8)..."
    )
)
def list_repo_files_with_backoff(api, **kwargs):
    """Wrapper for api.list_repo_files() with exponential backoff for rate limits."""
    return api.list_repo_files(**kwargs)


@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    max_tries=8,
    giveup=lambda e: not is_rate_limit_error(e),
    on_backoff=lambda details: print(
        f"⏳ Rate limited. Retrying in {details['wait']:.1f}s (attempt {details['tries']}/8)..."
    )
)
def hf_hub_download_with_backoff(**kwargs):
    """Wrapper for hf_hub_download() with exponential backoff for rate limits."""
    return hf_hub_download(**kwargs)


@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    max_tries=8,
    giveup=lambda e: not is_rate_limit_error(e),
    on_backoff=lambda details: print(
        f"⏳ Rate limited. Retrying in {details['wait']:.1f}s (attempt {details['tries']}/8)..."
    )
)
def upload_file_with_backoff(api, **kwargs):
    """Wrapper for api.upload_file() with exponential backoff for rate limits."""
    return api.upload_file(**kwargs)


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


def generate_table_union_statements(start_date, end_date):
    """
    Generate UNION ALL statements for githubarchive.day tables in date range.
    
    Args:
        start_date: Start datetime
        end_date: End datetime
        
    Returns:
        String with UNION ALL SELECT statements for all tables in range
    """
    table_names = []
    current_date = start_date
    
    while current_date < end_date:
        table_name = f"`githubarchive.day.{current_date.strftime('%Y%m%d')}`"
        table_names.append(table_name)
        current_date += timedelta(days=1)
    
    # Create UNION ALL chain
    union_parts = [f"SELECT * FROM {table}" for table in table_names]
    return " UNION ALL ".join(union_parts)


# =============================================================================
# BIGQUERY FUNCTIONS
# =============================================================================

def fetch_all_pr_metadata_batched(client, identifiers, start_date, end_date, batch_size=100):
    """
    Fetch PR review metadata for ALL agents using BATCHED BigQuery queries.
    Splits agents into smaller batches to avoid performance issues with large queries.

    Args:
        client: BigQuery client instance
        identifiers: List of GitHub usernames/bot identifiers
        start_date: Start datetime (timezone-aware)
        end_date: End datetime (timezone-aware)
        batch_size: Number of agents to process per batch (default: 100)

    Returns:
        Dictionary mapping agent identifier to list of PR metadata (same format as single query)
    """
    print(f"\n🔍 Using BATCHED approach: {len(identifiers)} agents in batches of {batch_size}")

    # Split identifiers into batches
    batches = [identifiers[i:i + batch_size] for i in range(0, len(identifiers), batch_size)]
    total_batches = len(batches)

    print(f"   Total batches: {total_batches}")

    # Collect results from all batches
    all_metadata = {}
    successful_batches = 0
    failed_batches = 0

    for batch_num, batch_identifiers in enumerate(batches, 1):
        print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch_identifiers)} agents)...")

        try:
            # Query this batch
            batch_results = fetch_all_pr_metadata_single_query(
                client, batch_identifiers, start_date, end_date
            )

            # Merge results
            for identifier, metadata_list in batch_results.items():
                if identifier in all_metadata:
                    all_metadata[identifier].extend(metadata_list)
                else:
                    all_metadata[identifier] = metadata_list

            successful_batches += 1
            print(f"   ✓ Batch {batch_num}/{total_batches} complete: {len(batch_results)} agents processed")

        except Exception as e:
            failed_batches += 1
            print(f"   ✗ Batch {batch_num}/{total_batches} failed: {str(e)}")
            print(f"   Continuing with remaining batches...")
            continue

    print(f"\n📊 Batching Summary:")
    print(f"   Total batches: {total_batches}")
    print(f"   Successful: {successful_batches}")
    print(f"   Failed: {failed_batches}")
    print(f"   Total agents with data: {len(all_metadata)}")

    return all_metadata


def fetch_all_pr_metadata_single_query(client, identifiers, start_date, end_date):
    """
    Fetch PR review metadata for a BATCH of agents using ONE comprehensive BigQuery query.

    NOTE: This function is designed for smaller batches (~100 agents).
    For large numbers of agents, use fetch_all_pr_metadata_batched() instead.
    
    This query combines:
    1. Review events (PullRequestReviewEvent) for all agents
    2. PR status (PullRequestEvent with action='closed')
    
    Args:
        client: BigQuery client instance
        identifiers: List of GitHub usernames/bot identifiers
        start_date: Start datetime (timezone-aware)
        end_date: End datetime (timezone-aware)
        
    Returns:
        Dictionary mapping agent identifier to list of PR metadata:
        {
            'agent-identifier': [
                {
                    'url': PR URL,
                    'reviewed_at': Review timestamp,
                    'merged_at': Merge timestamp (if merged, else None),
                    'closed_at': Close timestamp (if closed, else None)
                },
                ...
            ],
            ...
        }
    """
    print(f"\n🔍 Querying BigQuery for ALL {len(identifiers)} agents in ONE QUERY")
    print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Generate table UNION statements for review period
    review_tables = generate_table_union_statements(start_date, end_date)
    
    # Generate table UNION statements for PR status (use same lookback as reviews)
    status_start_date = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)
    status_tables = generate_table_union_statements(status_start_date, end_date)
    
    # Build identifier list for IN clause
    identifier_list = ', '.join([f"'{id}'" for id in identifiers])
    
    # Build comprehensive query with CTEs
    query = f"""
    WITH review_events AS (
      -- Get all review events for ALL agents
      SELECT
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') as url,
        COALESCE(
          JSON_EXTRACT_SCALAR(payload, '$.review.submitted_at'),
          CAST(created_at AS STRING)
        ) as reviewed_at,
        actor.login as reviewer,
        repo.name as repo_name,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') AS INT64) as pr_number
      FROM (
        {review_tables}
      )
      WHERE
        type = 'PullRequestReviewEvent'
        AND actor.login IN ({identifier_list})
        AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') IS NOT NULL

      UNION ALL

      -- Get PR comments (IssueCommentEvent on PRs)
      SELECT
        JSON_EXTRACT_SCALAR(payload, '$.issue.html_url') as url,
        CAST(created_at AS STRING) as reviewed_at,
        actor.login as reviewer,
        repo.name as repo_name,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.issue.number') AS INT64) as pr_number
      FROM (
        {review_tables}
      )
      WHERE
        type = 'IssueCommentEvent'
        AND actor.login IN ({identifier_list})
        AND JSON_EXTRACT_SCALAR(payload, '$.issue.pull_request.url') IS NOT NULL
        AND JSON_EXTRACT_SCALAR(payload, '$.issue.html_url') IS NOT NULL

      UNION ALL

      -- Get review comments (PullRequestReviewCommentEvent)
      SELECT
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') as url,
        CAST(created_at AS STRING) as reviewed_at,
        actor.login as reviewer,
        repo.name as repo_name,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') AS INT64) as pr_number
      FROM (
        {review_tables}
      )
      WHERE
        type = 'PullRequestReviewCommentEvent'
        AND actor.login IN ({identifier_list})
        AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') IS NOT NULL
    ),
    
    pr_status AS (
      -- Get merge/close status for those PRs
      SELECT
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') as url,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') AS BOOL) as is_merged,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at') as merged_at,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.closed_at') as closed_at,
        created_at
      FROM (
        {status_tables}
      )
      WHERE
        type = 'PullRequestEvent'
        AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'closed'
        AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') IS NOT NULL
        AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') IN (
          SELECT DISTINCT url FROM review_events
        )
      QUALIFY ROW_NUMBER() OVER (PARTITION BY url ORDER BY created_at DESC) = 1
    )
    
    -- Join review events with PR status
    SELECT DISTINCT
      re.reviewer,
      re.url,
      re.reviewed_at,
      ps.merged_at,
      ps.closed_at
    FROM review_events re
    LEFT JOIN pr_status ps ON re.url = ps.url
    ORDER BY re.reviewer, re.reviewed_at DESC
    """
    
    # Calculate number of days for reporting
    review_days = (end_date - start_date).days
    status_days = (end_date - status_start_date).days
    
    print(f"   Querying {review_days} days for reviews, {status_days} days for PR status...")
    print(f"   Agents: {', '.join(identifiers[:5])}{'...' if len(identifiers) > 5 else ''}")
    
    try:
        query_job = client.query(query)
        results = list(query_job.result())
        
        print(f"   ✓ Found {len(results)} total PR review records across all agents")
        
        # Group results by agent
        metadata_by_agent = defaultdict(list)
        
        for row in results:
            reviewer = row.reviewer

            # Convert datetime objects to ISO strings and normalize
            reviewed_at = row.reviewed_at
            if hasattr(reviewed_at, 'isoformat'):
                reviewed_at = reviewed_at.isoformat()
            reviewed_at = normalize_date_format(reviewed_at) if reviewed_at else None

            merged_at = row.merged_at
            if hasattr(merged_at, 'isoformat'):
                merged_at = merged_at.isoformat()
            merged_at = normalize_date_format(merged_at) if merged_at else None

            closed_at = row.closed_at
            if hasattr(closed_at, 'isoformat'):
                closed_at = closed_at.isoformat()
            closed_at = normalize_date_format(closed_at) if closed_at else None

            metadata_by_agent[reviewer].append({
                'url': row.url,
                'reviewed_at': reviewed_at,
                'merged_at': merged_at,
                'closed_at': closed_at,
            })
        
        # Print breakdown by agent
        print(f"\n   📊 Results breakdown by agent:")
        for identifier in identifiers:
            count = len(metadata_by_agent.get(identifier, []))
            if count > 0:
                metadata = metadata_by_agent[identifier]
                merged_count = sum(1 for m in metadata if m['merged_at'] is not None)
                closed_count = sum(1 for m in metadata if m['closed_at'] is not None and m['merged_at'] is None)
                open_count = count - merged_count - closed_count
                print(f"      {identifier}: {count} PRs ({merged_count} merged, {closed_count} closed, {open_count} open)")
        
        # Convert defaultdict to regular dict
        return dict(metadata_by_agent)
        
    except Exception as e:
        print(f"   ✗ BigQuery error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


# =============================================================================
# HUGGINGFACE STORAGE FUNCTIONS
# =============================================================================

def group_metadata_by_date(metadata_list):
    """
    Group review metadata by date (year.month.day) for daily storage.
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

    This function OVERWRITES existing files completely with fresh data from BigQuery.
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

        api = HfApi(token=token)

        # Group by date (year, month, day)
        grouped = group_metadata_by_date(metadata_list)

        if not grouped:
            print(f"   No valid metadata to save for {agent_identifier}")
            return False

        # Create a temporary directory for batch upload
        temp_dir = tempfile.mkdtemp()
        agent_folder = os.path.join(temp_dir, agent_identifier)
        os.makedirs(agent_folder, exist_ok=True)

        try:
            print(f"   📦 Preparing batch upload for {len(grouped)} daily files...")

            # Process each daily file
            for (review_year, month, day), day_metadata in grouped.items():
                filename = f"{agent_identifier}/{review_year}.{month:02d}.{day:02d}.jsonl"
                local_filename = os.path.join(agent_folder, f"{review_year}.{month:02d}.{day:02d}.jsonl")

                # Sort by reviewed_at for better organization
                day_metadata.sort(key=lambda x: x.get('reviewed_at', ''), reverse=True)

                # Save to temp directory (complete overwrite, no merging)
                save_jsonl(local_filename, day_metadata)
                print(f"      Prepared {len(day_metadata)} reviews for {filename}")

            # Upload entire folder using upload_large_folder (optimized for large files)
            # Note: upload_large_folder creates multiple commits automatically and doesn't support custom commit_message
            print(f"   📤 Uploading {len(grouped)} files ({len(metadata_list)} total reviews)...")
            upload_large_folder_with_backoff(
                api=api,
                folder_path=temp_dir,
                repo_id=REVIEW_METADATA_REPO,
                repo_type="dataset"
            )
            print(f"   ✓ Batch upload complete for {agent_identifier}")

            return True

        finally:
            # Always clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"   ✗ Error saving review metadata: {str(e)}")
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
        files = list_repo_files_with_backoff(api=api, repo_id=AGENTS_REPO, repo_type="dataset")

        # Filter for JSON files only
        json_files = [f for f in files if f.endswith('.json')]

        print(f"Found {len(json_files)} agent files in {AGENTS_REPO}")

        # Download and parse each JSON file
        for json_file in json_files:
            try:
                file_path = hf_hub_download_with_backoff(
                    repo_id=AGENTS_REPO,
                    filename=json_file,
                    repo_type="dataset"
                )

                with open(file_path, 'r') as f:
                    agent_data = json.load(f)

                    # Only process agents with status == "public"
                    if agent_data.get('status') != 'public':
                        print(f"Skipping {json_file}: status is not 'public'")
                        continue

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


def load_review_metadata():
    """
    Load all review metadata from HuggingFace dataset within LEADERBOARD_TIME_FRAME_DAYS.

    Returns:
        List of dictionaries with 'agent_identifier' added to each review metadata.
    """
    # Calculate cutoff date
    current_time = datetime.now(timezone.utc)
    cutoff_date = current_time - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    try:
        api = HfApi()
        token = get_hf_token()

        # List all files in the repository
        files = list_repo_files_with_backoff(api=api, repo_id=REVIEW_METADATA_REPO, repo_type="dataset")

        # Filter for JSONL files matching pattern: [agent_identifier]/YYYY.MM.DD.jsonl
        time_frame_files = []
        for f in files:
            if f.endswith('.jsonl'):
                parts = f.split('/')
                if len(parts) == 2:
                    filename = parts[1]
                    # Parse date from filename: YYYY.MM.DD.jsonl
                    try:
                        date_part = filename.replace('.jsonl', '')
                        date_components = date_part.split('.')
                        if len(date_components) == 3:
                            file_year, file_month, file_day = map(int, date_components)
                            file_date = datetime(file_year, file_month, file_day, tzinfo=timezone.utc)

                            # Only include files within time frame
                            if file_date >= cutoff_date:
                                time_frame_files.append(f)
                    except Exception:
                        continue

        print(f"📥 Loading review metadata from last {LEADERBOARD_TIME_FRAME_DAYS} days ({len(time_frame_files)} daily files)...")

        all_metadata = []

        for filename in time_frame_files:
            try:
                # Extract agent_identifier from path
                parts = filename.split('/')
                if len(parts) != 2:
                    continue

                agent_identifier = parts[0]

                file_path = hf_hub_download_with_backoff(
                    repo_id=REVIEW_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=token
                )
                day_metadata = load_jsonl(file_path)

                # Add agent_identifier to each review
                for review_meta in day_metadata:
                    review_meta['agent_identifier'] = agent_identifier
                    all_metadata.append(review_meta)

            except Exception as e:
                print(f"   Warning: Could not load {filename}: {str(e)}")

        print(f"✓ Loaded {len(all_metadata)} total reviews from last {LEADERBOARD_TIME_FRAME_DAYS} days")
        return all_metadata

    except Exception as e:
        print(f"✗ Error loading review metadata: {str(e)}")
        return []


def get_pr_status_from_metadata(review_meta):
    """
    Derive PR status from merged_at and closed_at fields.

    Returns:
        str: 'merged', 'closed', or 'open'
    """
    merged_at = review_meta.get('merged_at')
    closed_at = review_meta.get('closed_at')

    if merged_at:
        return 'merged'
    elif closed_at:
        return 'closed'
    else:
        return 'open'


def calculate_review_stats_from_metadata(metadata_list):
    """
    Calculate statistics from a list of review metadata.

    Returns:
        Dictionary with review metrics (total_reviews, merged_prs, acceptance_rate, etc.)
    """
    total_reviews = len(metadata_list)

    # Count merged PRs
    merged_prs = sum(1 for review_meta in metadata_list
                      if get_pr_status_from_metadata(review_meta) == 'merged')

    # Count rejected PRs
    rejected_prs = sum(1 for review_meta in metadata_list
                      if get_pr_status_from_metadata(review_meta) == 'closed')

    # Count pending PRs
    pending_prs = sum(1 for review_meta in metadata_list
                     if get_pr_status_from_metadata(review_meta) == 'open')

    # Calculate acceptance rate (exclude pending PRs)
    completed_prs = merged_prs + rejected_prs
    acceptance_rate = (merged_prs / completed_prs * 100) if completed_prs > 0 else 0

    return {
        'total_reviews': total_reviews,
        'merged_prs': merged_prs,
        'pending_prs': pending_prs,
        'acceptance_rate': round(acceptance_rate, 2),
    }


def calculate_monthly_metrics_by_agent():
    """
    Calculate monthly metrics for all agents for visualization.

    Returns:
        dict: {
            'agents': list of agent names,
            'months': list of month labels (e.g., '2025-01'),
            'data': {
                agent_name: {
                    'acceptance_rates': list of acceptance rates by month,
                    'total_reviews': list of review counts by month,
                    'merged_prs': list of merged PR counts by month,
                }
            }
        }
    """
    # Load agents
    agents = load_agents_from_hf()

    # Create mapping from agent_identifier to agent_name
    identifier_to_name = {agent.get('github_identifier'): agent.get('name') for agent in agents if agent.get('github_identifier')}

    # Load all review metadata
    all_metadata = load_review_metadata()

    if not all_metadata:
        return {'agents': [], 'months': [], 'data': {}}

    # Group by agent and month
    agent_month_data = defaultdict(lambda: defaultdict(list))

    for review_meta in all_metadata:
        agent_identifier = review_meta.get('agent_identifier')
        reviewed_at = review_meta.get('reviewed_at')

        if not agent_identifier or not reviewed_at:
            continue

        # Get agent_name from identifier
        agent_name = identifier_to_name.get(agent_identifier, agent_identifier)

        try:
            dt = datetime.fromisoformat(reviewed_at.replace('Z', '+00:00'))
            month_key = f"{dt.year}-{dt.month:02d}"
            agent_month_data[agent_name][month_key].append(review_meta)
        except Exception as e:
            print(f"Warning: Could not parse date '{reviewed_at}': {e}")
            continue

    # Get all unique months and sort them
    all_months = set()
    for agent_data in agent_month_data.values():
        all_months.update(agent_data.keys())
    months = sorted(list(all_months))

    # Calculate metrics for each agent and month
    result_data = {}
    for agent_name, month_dict in agent_month_data.items():
        acceptance_rates = []
        total_reviews_list = []
        merged_prs_list = []

        for month in months:
            reviews_in_month = month_dict.get(month, [])

            # Count merged PRs
            merged_count = sum(1 for review in reviews_in_month
                                if get_pr_status_from_metadata(review) == 'merged')

            # Count rejected PRs
            rejected_count = sum(1 for review in reviews_in_month
                                if get_pr_status_from_metadata(review) == 'closed')

            # Total reviews
            total_count = len(reviews_in_month)

            # Calculate acceptance rate (exclude pending PRs)
            completed_count = merged_count + rejected_count
            acceptance_rate = (merged_count / completed_count * 100) if completed_count > 0 else None

            acceptance_rates.append(acceptance_rate)
            total_reviews_list.append(total_count)
            merged_prs_list.append(merged_count)

        result_data[agent_name] = {
            'acceptance_rates': acceptance_rates,
            'total_reviews': total_reviews_list,
            'merged_prs': merged_prs_list,
        }

    agents_list = sorted(list(agent_month_data.keys()))

    return {
        'agents': agents_list,
        'months': months,
        'data': result_data
    }


def construct_leaderboard_from_metadata():
    """
    Construct leaderboard from stored review metadata.

    Returns:
        Dictionary of agent stats.
    """
    print("\n📊 Constructing leaderboard from review metadata...")

    # Load agents
    agents = load_agents_from_hf()
    if not agents:
        print("⚠️ No agents found")
        return {}

    print(f"✓ Loaded {len(agents)} agents")

    # Load all review metadata
    all_metadata = load_review_metadata()
    print(f"✓ Loaded {len(all_metadata)} review metadata entries")

    cache_dict = {}

    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('name', 'Unknown')

        # Filter metadata for this agent
        agent_metadata = [review for review in all_metadata if review.get("agent_identifier") == identifier]

        # Calculate stats
        stats = calculate_review_stats_from_metadata(agent_metadata)

        cache_dict[identifier] = {
            'name': agent_name,
            'name': agent_name,
            'website': agent.get('website', 'N/A'),
            'github_identifier': identifier,
            **stats
        }

    print(f"✓ Constructed cache with {len(cache_dict)} agent entries")

    return cache_dict


def save_leaderboard_data_to_hf(leaderboard_dict, monthly_metrics):
    """
    Save leaderboard data and monthly metrics to HuggingFace dataset as swe-review.json.

    Args:
        leaderboard_dict: Dictionary of agent stats from construct_leaderboard_from_metadata()
        monthly_metrics: Monthly metrics data from calculate_monthly_metrics_by_agent()

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi(token=token)
        filename = "swe-review.json"

        # Combine leaderboard and monthly metrics
        combined_data = {
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'leaderboard': leaderboard_dict,
            'monthly_metrics': monthly_metrics,
            'metadata': {
                'leaderboard_time_frame_days': LEADERBOARD_TIME_FRAME_DAYS
            }
        }

        # Save locally first
        with open(filename, 'w') as f:
            json.dump(combined_data, f, indent=2)

        try:
            # Upload to HuggingFace
            upload_file_with_backoff(
                api=api,
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=LEADERBOARD_REPO,
                repo_type="dataset"
            )
            print(f"✓ Saved leaderboard data to HuggingFace: {filename}")
            return True
        finally:
            # Always clean up local file
            if os.path.exists(filename):
                os.remove(filename)

    except Exception as e:
        print(f"✗ Error saving leaderboard data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# MAIN MINING FUNCTION
# =============================================================================

def mine_all_agents():
    """
    Mine review metadata for all agents within LEADERBOARD_TIME_FRAME_DAYS and save to HuggingFace.
    Uses ONE BigQuery query for ALL agents (most efficient approach).
    """
    # Load agent metadata from HuggingFace
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found in HuggingFace dataset")
        return
    
    # Extract all identifiers
    identifiers = [agent['github_identifier'] for agent in agents if agent.get('github_identifier')]
    if not identifiers:
        print("No valid agent identifiers found")
        return
    
    print(f"\n{'='*80}")
    print(f"Starting review metadata mining for {len(identifiers)} agents")
    print(f"Time frame: Last {LEADERBOARD_TIME_FRAME_DAYS} days")
    print(f"Data source: BigQuery + GitHub Archive (BATCHED QUERIES)")
    print(f"{'='*80}\n")
    
    # Initialize BigQuery client
    try:
        client = get_bigquery_client()
    except Exception as e:
        print(f"✗ Failed to initialize BigQuery client: {str(e)}")
        return
    
    # Define time range: past LEADERBOARD_TIME_FRAME_DAYS (excluding today)
    current_time = datetime.now(timezone.utc)
    end_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)
    
    try:
        # Use batched approach for better performance
        all_metadata = fetch_all_pr_metadata_batched(
            client, identifiers, start_date, end_date, batch_size=100
        )
    except Exception as e:
        print(f"✗ Error during BigQuery fetch: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Save results for each agent
    print(f"\n{'='*80}")
    print(f"💾 Saving results to HuggingFace for each agent...")
    print(f"{'='*80}\n")
    
    success_count = 0
    error_count = 0
    no_data_count = 0
    
    for i, agent in enumerate(agents, 1):
        identifier = agent.get('github_identifier')
        agent_name = agent.get('name', 'Unknown')
        
        if not identifier:
            print(f"[{i}/{len(agents)}] Skipping agent without identifier")
            error_count += 1
            continue
        
        metadata = all_metadata.get(identifier, [])
        
        print(f"[{i}/{len(agents)}] {agent_name} ({identifier}):")
        
        try:
            if metadata:
                print(f"   💾 Saving {len(metadata)} review records...")
                if save_review_metadata_to_hf(metadata, identifier):
                    success_count += 1
                else:
                    error_count += 1
            else:
                print(f"   No reviews found")
                no_data_count += 1
        
        except Exception as e:
            print(f"   ✗ Error saving {identifier}: {str(e)}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue
    
    # Calculate number of batches
    total_identifiers = len(identifiers)
    batch_size = 100
    num_batches = (total_identifiers + batch_size - 1) // batch_size  # Ceiling division

    print(f"\n{'='*80}")
    print(f"✅ Mining complete!")
    print(f"   Total agents: {len(agents)}")
    print(f"   Successfully saved: {success_count}")
    print(f"   No data (skipped): {no_data_count}")
    print(f"   Errors: {error_count}")
    print(f"   BigQuery batches executed: {num_batches} (batch size: {batch_size})")
    print(f"{'='*80}\n")

    # Construct and save leaderboard data
    print(f"\n{'='*80}")
    print(f"📊 Constructing and saving leaderboard data...")
    print(f"{'='*80}\n")

    try:
        # Construct leaderboard
        leaderboard_dict = construct_leaderboard_from_metadata()

        # Calculate monthly metrics
        print(f"\n📈 Calculating monthly metrics...")
        monthly_metrics = calculate_monthly_metrics_by_agent()

        # Save to HuggingFace
        print(f"\n💾 Saving leaderboard data to HuggingFace...")
        save_leaderboard_data_to_hf(leaderboard_dict, monthly_metrics)

        print(f"\n{'='*80}")
        print(f"✅ Leaderboard data saved successfully!")
        print(f"   Leaderboard entries: {len(leaderboard_dict)}")
        print(f"   Monthly data points: {len(monthly_metrics.get('months', []))} months")
        print(f"   Saved to: {LEADERBOARD_REPO}/swe-review.json")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n✗ Failed to construct/save leaderboard data: {str(e)}")
        import traceback
        traceback.print_exc()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mine_all_agents()