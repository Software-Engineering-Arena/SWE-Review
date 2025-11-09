import gradio as gr
from gradio_leaderboard import Leaderboard, ColumnFilter
import json
import os
import time
import tempfile
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from datasets import load_dataset, Dataset
import backoff
from dotenv import load_dotenv
import pandas as pd
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from google.cloud import bigquery

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/bot_metadata"  # HuggingFace dataset for agent metadata
REVIEW_METADATA_REPO = "SWE-Arena/review_metadata"  # HuggingFace dataset for review metadata
LEADERBOARD_REPO = "SWE-Arena/leaderboard_metadata"  # HuggingFace dataset for leaderboard data
LEADERBOARD_TIME_FRAME_DAYS = 180  # Time frame for constructing leaderboard
UPDATE_TIME_FRAME_DAYS = 30  # Time frame for mining new reviews

LEADERBOARD_COLUMNS = [
    ("Agent Name", "string"),
    ("Website", "string"),
    ("Total Reviews", "number"),
    ("Merged PRs", "number"),
    ("Acceptance Rate (%)", "number"),
]

# =============================================================================
# JSONL FILE OPERATIONS
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
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data


def save_jsonl(filename, data):
    """Save list of dictionaries to JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def cache_to_dict(cache_list):
    """Convert list of cache entries to dictionary by identifier."""
    return {entry['github_identifier']: entry for entry in cache_list}


def dict_to_cache(cache_dict):
    """Convert dictionary back to list of values."""
    return list(cache_dict.values())


def normalize_date_format(date_string):
    """
    Convert date strings to standardized ISO 8601 format with Z suffix.
    Handles both old format (2025-10-15T23:23:47.983068) and new format (2025-10-15T23:23:47Z).
    """
    if not date_string or date_string == 'N/A':
        return 'N/A'

    try:
        # Replace space with 'T' for ISO format compatibility
        date_string = date_string.replace(' ', 'T')

        # Fix incomplete timezone offset (+00 or -00 -> +00:00 or -00:00)
        if date_string[-3:-2] in ('+', '-') and ':' not in date_string[-3:]:
            date_string = date_string + ':00'

        # Parse the date string (handles both with and without microseconds)
        dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))

        # Convert to standardized format
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        print(f"Warning: Could not parse date '{date_string}': {e}")
        return date_string


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
    base=300,
    max_value=3600,
    giveup=lambda e: not is_rate_limit_error(e),
    on_backoff=lambda details: print(
        f"⏳ Rate limited. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/8..."
    )
)
def upload_large_folder_with_backoff(api, **kwargs):
    """Wrapper for api.upload_large_folder() with exponential backoff for rate limits."""
    return api.upload_large_folder(**kwargs)


@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    max_tries=8,
    base=300,
    max_value=3600,
    giveup=lambda e: not is_rate_limit_error(e),
    on_backoff=lambda details: print(
        f"⏳ Rate limited. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/8..."
    )
)
def list_repo_files_with_backoff(api, **kwargs):
    """Wrapper for api.list_repo_files() with exponential backoff for rate limits."""
    return api.list_repo_files(**kwargs)


@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    max_tries=8,
    base=300,
    max_value=3600,
    giveup=lambda e: not is_rate_limit_error(e),
    on_backoff=lambda details: print(
        f"⏳ Rate limited. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/8..."
    )
)
def hf_hub_download_with_backoff(**kwargs):
    """Wrapper for hf_hub_download() with exponential backoff for rate limits."""
    return hf_hub_download(**kwargs)


@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    max_tries=8,
    base=300,
    max_value=3600,
    giveup=lambda e: not is_rate_limit_error(e),
    on_backoff=lambda details: print(
        f"⏳ Rate limited. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/8..."
    )
)
def upload_file_with_backoff(api, **kwargs):
    """Wrapper for api.upload_file() with exponential backoff for rate limits."""
    return api.upload_file(**kwargs)


# =============================================================================
# BIGQUERY FUNCTIONS
# =============================================================================

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


def fetch_all_pr_metadata_batched(client, identifiers, start_date, end_date, batch_size=100, upload_immediately=True):
    """
    Fetch PR review metadata for ALL agents using BATCHED BigQuery queries.
    Splits agents into smaller batches to avoid performance issues with large queries.

    Args:
        client: BigQuery client instance
        identifiers: List of GitHub usernames/bot identifiers
        start_date: Start datetime (timezone-aware)
        end_date: End datetime (timezone-aware)
        batch_size: Number of agents to process per batch (default: 100)
        upload_immediately: If True, upload each batch to HuggingFace immediately after processing (default: True)

    Returns:
        Dictionary mapping agent identifier to list of PR metadata
    """
    print(f"\n🔍 Using BATCHED approach: {len(identifiers)} agents in batches of {batch_size}")

    # Log upload mode
    if upload_immediately:
        print(f"   📤 Upload mode: IMMEDIATE (upload after each batch)")
    else:
        print(f"   📤 Upload mode: DEFERRED (upload after all batches complete)")

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
            # Query this batch - process each agent in the batch
            batch_results = {}
            for identifier in batch_identifiers:
                review_rows = fetch_reviews_from_bigquery(client, identifier, start_date, end_date)

                # Extract metadata
                metadata_list = []
                seen_prs = set()
                for row in review_rows:
                    url = row.url
                    if url in seen_prs:
                        continue
                    seen_prs.add(url)

                    metadata = extract_review_metadata_from_bigquery(row)
                    metadata_list.append(metadata)

                if metadata_list:
                    all_metadata[identifier] = metadata_list
                    batch_results[identifier] = metadata_list

            successful_batches += 1
            print(f"   ✓ Batch {batch_num}/{total_batches} complete: {len(batch_identifiers)} agents processed")

            # Upload immediately after this batch if enabled
            if upload_immediately and batch_results:
                print(f"\n   📤 Uploading batch {batch_num}/{total_batches} results to HuggingFace...")
                upload_success = 0
                upload_errors = 0

                for identifier, metadata_list in batch_results.items():
                    if metadata_list:
                        if save_review_metadata_to_hf(metadata_list, identifier):
                            upload_success += 1
                        else:
                            upload_errors += 1

                print(f"   ✓ Batch {batch_num}/{total_batches} upload complete ({upload_success} agents uploaded, {upload_errors} errors)")

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


def fetch_reviews_from_bigquery(client, identifier, start_date, end_date):
    """
    Fetch PR review events from GitHub Archive for a SINGLE agent.

    NOTE: This function is designed for querying a single agent at a time.
    For querying multiple agents efficiently, use fetch_all_pr_metadata_batched() instead.

    Queries githubarchive.day.YYYYMMDD tables for PullRequestReviewEvent where
    actor.login matches the agent identifier, and joins with PR status.

    Args:
        client: BigQuery client instance
        identifier: GitHub username or bot identifier (e.g., 'amazon-inspector-beta[bot]')
        start_date: Start datetime (timezone-aware)
        end_date: End datetime (timezone-aware)

    Returns:
        List of review event rows with PR information including merged_at and closed_at
    """
    print(f"\n🔍 Querying BigQuery for reviews by {identifier}")
    print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Generate list of table names for each day in the range
    review_tables = []
    current_date = start_date
    while current_date < end_date:
        table_name = f"`githubarchive.day.{current_date.strftime('%Y%m%d')}`"
        review_tables.append(f"SELECT * FROM {table_name}")
        current_date += timedelta(days=1)
    review_union = " UNION ALL ".join(review_tables)

    # Generate status tables (lookback for PR status)
    status_start = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)
    status_tables = []
    current_date = status_start
    while current_date < end_date:
        table_name = f"`githubarchive.day.{current_date.strftime('%Y%m%d')}`"
        status_tables.append(f"SELECT * FROM {table_name}")
        current_date += timedelta(days=1)
    status_union = " UNION ALL ".join(status_tables)

    # Build comprehensive query with CTEs for PR status
    query = f"""
    WITH review_events AS (
        SELECT
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') as url,
            COALESCE(
                JSON_EXTRACT_SCALAR(payload, '$.review.submitted_at'),
                CAST(created_at AS STRING)
            ) as reviewed_at,
            actor.login as reviewer,
            created_at
        FROM (
            {review_union}
        )
        WHERE type = 'PullRequestReviewEvent'
        AND actor.login = @identifier
        AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') IS NOT NULL
    ),
    pr_status AS (
        SELECT
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') as url,
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at') as merged_at,
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.closed_at') as closed_at,
            created_at
        FROM (
            {status_union}
        )
        WHERE type = 'PullRequestEvent'
        AND JSON_EXTRACT_SCALAR(payload, '$.action') = 'closed'
        AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') IN (
            SELECT DISTINCT url FROM review_events
        )
        QUALIFY ROW_NUMBER() OVER (PARTITION BY url ORDER BY created_at DESC) = 1
    )
    SELECT DISTINCT
        re.url,
        re.reviewed_at,
        re.created_at,
        ps.merged_at,
        ps.closed_at
    FROM review_events re
    LEFT JOIN pr_status ps ON re.url = ps.url
    ORDER BY re.reviewed_at DESC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("identifier", "STRING", identifier)
        ]
    )

    print(f"   Querying {len(review_tables)} review tables and {len(status_tables)} status tables...")

    try:
        query_job = client.query(query, job_config=job_config)
        results = list(query_job.result())

        print(f"   ✓ Found {len(results)} review events")
        return results

    except Exception as e:
        print(f"   ✗ BigQuery error: {str(e)}")
        return []


def extract_review_metadata_from_bigquery(review_row):
    """
    Extract minimal PR review metadata from BigQuery row.

    Args:
        review_row: BigQuery row from PullRequestReviewEvent query

    Returns:
        Dictionary with review metadata containing:
        - url: PR URL
        - reviewed_at: Review timestamp
        - merged_at: Merge timestamp (if merged, else None)
        - closed_at: Close timestamp (if closed, else None)
    """
    url = review_row.url
    reviewed_at = review_row.reviewed_at or review_row.created_at
    merged_at = getattr(review_row, 'merged_at', None)
    closed_at = getattr(review_row, 'closed_at', None)

    # Convert to ISO format if datetime and normalize
    if hasattr(reviewed_at, 'isoformat'):
        reviewed_at = reviewed_at.isoformat()
    reviewed_at = normalize_date_format(reviewed_at) if reviewed_at else None

    if merged_at and hasattr(merged_at, 'isoformat'):
        merged_at = merged_at.isoformat()
    merged_at = normalize_date_format(merged_at) if merged_at else None

    if closed_at and hasattr(closed_at, 'isoformat'):
        closed_at = closed_at.isoformat()
    closed_at = normalize_date_format(closed_at) if closed_at else None

    return {
        'url': url,
        'reviewed_at': reviewed_at,
        'merged_at': merged_at,
        'closed_at': closed_at
    }


# =============================================================================
# GITHUB API OPERATIONS
# =============================================================================

def request_with_backoff(method, url, *, headers=None, params=None, json_body=None, data=None, max_retries=10, timeout=30):
    """
    Perform an HTTP request with exponential backoff and jitter for GitHub API.
    Retries on 403/429 (rate limits), 5xx server errors, and transient network exceptions.

    Returns the final requests.Response on success or non-retryable status, or None after exhausting retries.
    """
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = requests.request(
                method,
                url,
                headers=headers or {},
                params=params,
                json=json_body,
                data=data,
                timeout=timeout
            )

            status = resp.status_code

            # Success
            if 200 <= status < 300:
                return resp

            # Rate limits or server errors -> retry with backoff
            if status in (403, 429) or 500 <= status < 600:
                wait = None

                # Prefer Retry-After when present
                retry_after = resp.headers.get('Retry-After') or resp.headers.get('retry-after')
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except Exception:
                        wait = None

                # Fallback to X-RateLimit-Reset when 403/429
                if wait is None and status in (403, 429):
                    reset_hdr = resp.headers.get('X-RateLimit-Reset') or resp.headers.get('x-ratelimit-reset')
                    if reset_hdr:
                        try:
                            reset_timestamp = int(float(reset_hdr))
                            wait = max(reset_timestamp - time.time() + 2, 1)
                        except Exception:
                            wait = None

                # Final fallback: exponential backoff with jitter
                if wait is None:
                    wait = delay + random.uniform(0, 0.5)

                # Cap individual wait to avoid extreme sleeps
                wait = max(1.0, min(wait, 120.0))
                print(f"GitHub API {status}. Backing off {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
                continue

            # Non-retryable error; return response for caller to handle
            return resp

        except requests.RequestException as e:
            # Network error -> retry with backoff
            wait = delay + random.uniform(0, 0.5)
            wait = max(1.0, min(wait, 60.0))
            print(f"Request error: {e}. Retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
            delay = min(delay * 2, 60.0)

    print(f"Exceeded max retries for {url}")
    return None


def validate_github_username(identifier):
    """Verify that a GitHub identifier exists with backoff-aware requests."""
    try:
        url = f'https://api.github.com/users/{identifier}'
        response = request_with_backoff('GET', url, max_retries=1)
        if response is None:
            return False, "Validation error: network/rate limit exhausted"
        if response.status_code == 200:
            return True, "Username is valid"
        elif response.status_code == 404:
            return False, "GitHub identifier not found"
        else:
            return False, f"Validation error: HTTP {response.status_code}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def extract_review_metadata(pr):
    """
    Extract minimal PR review metadata for efficient storage.
    Only keeps essential fields: url, reviewed_at, merged_at, closed_at.
    Note: agent_name is not stored as it's inferred from the folder structure.

    Status can be derived from the timestamps:
    - merged_at: Timestamp if PR was merged, None otherwise
    - closed_at: Timestamp if PR was closed (either merged or just closed), None otherwise

    Merged PR = PR that was merged (merged_at is not None)
    Rejected PR = PR that was closed without merging (closed_at is not None but merged_at is None)
    Open PR = PR still open (both merged_at and closed_at are None)
    """
    # Extract PR metadata from search results
    # The GitHub search API returns PR data from /search/issues endpoint
    url = pr.get('url')
    created_at = pr.get('created_at')
    closed_at = pr.get('closed_at')

    # Check if PR has pull_request field (indicates it's a PR, not an issue)
    pull_request_data = pr.get('pull_request', {})
    merged_at = pull_request_data.get('merged_at') if pull_request_data else None

    return {
        'url': url,
        'reviewed_at': created_at,  # When the PR was created (agent reviewed it)
        'merged_at': merged_at,
        'closed_at': closed_at
    }


def get_pr_status_from_metadata(review_meta):
    """
    Derive PR status from merged_at and closed_at fields.

    Args:
        review_meta: Dictionary containing merged_at and closed_at fields

    Returns:
        str: 'merged', 'closed', or 'open'
    """
    merged_at = review_meta.get('merged_at')
    closed_at = review_meta.get('closed_at')

    # If merged_at is set (not None and not False), PR is merged
    if merged_at:
        return 'merged'
    # If closed_at is set but not merged, PR is closed without merging
    elif closed_at:
        return 'closed'
    # Otherwise, PR is still open
    else:
        return 'open'


def calculate_review_stats_from_metadata(metadata_list):
    """
    Calculate statistics from a list of review metadata (lightweight objects).
    Works with minimal metadata: url, reviewed_at, merged_at, closed_at.

    Returns a dictionary with comprehensive review metrics.

    Acceptance Rate is calculated as:
        merged PRs / (merged PRs + rejected PRs) * 100

    Merged PRs = PRs that were merged (merged_at is not None)
    Rejected PRs = PRs that were closed without merging (closed_at is not None but merged_at is None)
    Pending PRs = PRs still open (both merged_at and closed_at are None) - excluded from acceptance rate
    """
    total_reviews = len(metadata_list)

    # Count merged PRs (merged_at is set)
    merged_prs = sum(1 for review_meta in metadata_list
                      if get_pr_status_from_metadata(review_meta) == 'merged')

    # Count rejected PRs (closed without merging)
    rejected_prs = sum(1 for review_meta in metadata_list
                      if get_pr_status_from_metadata(review_meta) == 'closed')

    # Count pending PRs (still open)
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


def calculate_monthly_metrics_by_agent(top_n=None):
    """
    Calculate monthly metrics for all agents (or top N agents) for visualization.
    Loads data directly from SWE-Arena/review_metadata dataset.

    Args:
        top_n: If specified, only return metrics for the top N agents by total reviews.
               Agents are ranked by their total review count across all months.

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
    # Load ALL agents from HuggingFace agents repo
    agents = load_agents_from_hf()

    # Create mapping from agent_identifier to agent_name
    identifier_to_name = {agent.get('github_identifier'): agent.get('name') for agent in agents if agent.get('github_identifier')}

    # Load all review metadata from review_metadata dataset
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

            # Count merged PRs (merged_at is set)
            merged_count = sum(1 for review in reviews_in_month
                                if get_pr_status_from_metadata(review) == 'merged')

            # Count rejected PRs (closed without merging)
            rejected_count = sum(1 for review in reviews_in_month
                                if get_pr_status_from_metadata(review) == 'closed')

            # Total reviews created in this month
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

    # Filter to top N agents if specified
    agents_list = sorted(list(agent_month_data.keys()))
    if top_n is not None and top_n > 0:
        # Calculate total reviews for each agent across all months
        agent_totals = []
        for agent_name in agents_list:
            total_reviews = sum(result_data[agent_name]['total_reviews'])
            agent_totals.append((agent_name, total_reviews))

        # Sort by total reviews (descending) and take top N
        agent_totals.sort(key=lambda x: x[1], reverse=True)
        top_agents = [agent_name for agent_name, _ in agent_totals[:top_n]]

        # Filter result_data to only include top agents
        result_data = {agent: result_data[agent] for agent in top_agents if agent in result_data}
        agents_list = top_agents

    return {
        'agents': agents_list,
        'months': months,
        'data': result_data
    }


# =============================================================================
# REVIEW METADATA STORAGE & RETRIEVAL
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

    This function APPENDS new metadata and DEDUPLICATES by URL.
    Uses batch upload to avoid rate limit (uploads entire folder in single commit).

    Args:
        metadata_list: List of review metadata dictionaries
        agent_identifier: GitHub identifier of the agent (used as folder name)
    """
    import tempfile
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

                # Merge and deduplicate by URL
                existing_by_url = {meta['url']: meta for meta in existing_metadata if meta.get('url')}
                new_by_url = {meta['url']: meta for meta in day_metadata if meta.get('url')}

                # Update with new data (new data overwrites old)
                existing_by_url.update(new_by_url)
                merged_metadata = list(existing_by_url.values())

                # Save to temp directory
                save_jsonl(local_filename, merged_metadata)
                print(f"   Prepared {len(merged_metadata)} reviews for {filename}")

            # Upload entire folder using upload_large_folder (optimized for large files)
            # Note: upload_large_folder creates multiple commits automatically and doesn't support custom commit_message
            print(f"📤 Uploading {len(grouped)} files...")
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
        print(f"✗ Error saving review metadata: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def load_review_metadata():
    """
    Load review metadata from the last LEADERBOARD_TIME_FRAME_DAYS.

    Structure: [agent_identifier]/YYYY.MM.DD.jsonl

    Returns:
        List of dictionaries with 'agent_identifier' added to each review metadata.
        Only includes reviews from the last LEADERBOARD_TIME_FRAME_DAYS.
    """
    # Calculate cutoff date based on LEADERBOARD_TIME_FRAME_DAYS
    current_time = datetime.now(timezone.utc)
    cutoff_date = current_time - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    try:
        api = HfApi()
        token = get_hf_token()

        # List all files in the repository
        files = list_repo_files_with_backoff(api=api, repo_id=REVIEW_METADATA_REPO, repo_type="dataset")

        # Filter for files matching the pattern: [agent_identifier]/YYYY.MM.DD.jsonl
        # AND within the time frame (parse date from filename)
        time_frame_files = []
        for f in files:
            if f.endswith('.jsonl'):
                parts = f.split('/')
                if len(parts) == 2:  # [agent_identifier]/YYYY.MM.DD.jsonl
                    filename = parts[1]
                    # Parse date from filename: YYYY.MM.DD.jsonl
                    try:
                        date_part = filename.replace('.jsonl', '')  # Get YYYY.MM.DD
                        date_components = date_part.split('.')
                        if len(date_components) == 3:
                            file_year, file_month, file_day = map(int, date_components)
                            file_date = datetime(file_year, file_month, file_day, tzinfo=timezone.utc)

                            # Only include files within the time frame
                            if file_date >= cutoff_date:
                                time_frame_files.append(f)
                    except Exception:
                        # If we can't parse the date, skip this file
                        continue

        print(f"📥 Loading review metadata from last {LEADERBOARD_TIME_FRAME_DAYS} days ({len(time_frame_files)} daily files across all agents)...")

        all_metadata = []
        agent_identifiers_found = set()

        for filename in time_frame_files:
            try:
                # Extract agent_identifier from path (first part)
                # Format: agent_identifier/YYYY.MM.DD.jsonl
                parts = filename.split('/')
                if len(parts) != 2:
                    print(f"   Warning: Unexpected filename format: {filename}")
                    continue

                agent_identifier = parts[0]
                agent_identifiers_found.add(agent_identifier)

                file_path = hf_hub_download_with_backoff(
                    repo_id=REVIEW_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=token
                )
                day_metadata = load_jsonl(file_path)

                # Add agent_identifier and filter by time frame (double-check)
                filtered_count = 0
                for review_meta in day_metadata:
                    # Validate review date is within time frame
                    reviewed_at = review_meta.get('reviewed_at')
                    if reviewed_at:
                        try:
                            dt = datetime.fromisoformat(reviewed_at.replace('Z', '+00:00'))
                            if dt < cutoff_date:
                                continue  # Skip reviews older than time frame
                        except Exception:
                            pass  # Keep reviews with unparseable dates

                    review_meta['agent_identifier'] = agent_identifier
                    all_metadata.append(review_meta)
                    filtered_count += 1

                print(f"   ✓ Loaded {filtered_count} reviews from {filename}")
            except Exception as e:
                print(f"   Warning: Could not load {filename}: {str(e)}")

        print(f"✓ Loaded {len(all_metadata)} total reviews from last {LEADERBOARD_TIME_FRAME_DAYS} days")

        return all_metadata

    except Exception as e:
        print(f"✗ Error loading review metadata from last {LEADERBOARD_TIME_FRAME_DAYS} days: {str(e)}")
        return []


def get_latest_review_date_for_agent(agent_identifier):
    """
    Get the latest review creation date for an agent from stored metadata.
    Used for incremental updates - only fetch reviews newer than this date.

    Structure: [agent_identifier]/YYYY.MM.DD.jsonl

    Args:
        agent_identifier: GitHub identifier of the agent

    Returns:
        datetime or None if no existing reviews found.
    """
    try:
        api = HfApi()
        token = get_hf_token()

        # List all files in the repository
        files = list_repo_files_with_backoff(api=api, repo_id=REVIEW_METADATA_REPO, repo_type="dataset")

        # Filter for files in this agent's folder
        # New structure: [agent_identifier]/YYYY.MM.DD.jsonl
        agent_pattern = f"{agent_identifier}/"
        agent_files = [f for f in files if f.startswith(agent_pattern) and f.endswith('.jsonl')]

        if not agent_files:
            return None

        # Find latest created_at across all files
        latest_date = None
        for filename in agent_files:
            try:
                file_path = hf_hub_download_with_backoff(
                    repo_id=REVIEW_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=token
                )
                metadata = load_jsonl(file_path)

                for review_meta in metadata:
                    reviewed_at = review_meta.get("reviewed_at")
                    if reviewed_at:
                        try:
                            dt = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
                            if latest_date is None or dt > latest_date:
                                latest_date = dt
                        except Exception:
                            continue
            except Exception:
                continue

        return latest_date

    except Exception:
        return None


def get_daily_files_last_time_frame(agent_identifier):
    """
    Get list of daily file paths for an agent from the configured time frame.

    Args:
        agent_identifier: GitHub identifier of the agent

    Returns:
        List of file paths in format: [agent_identifier]/YYYY.MM.DD.jsonl
    """
    try:
        api = HfApi()
        token = get_hf_token()

        # Calculate date range using configured time frame
        today = datetime.now(timezone.utc)
        cutoff_date = today - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

        # List all files in the repository
        files = list_repo_files_with_backoff(api=api, repo_id=REVIEW_METADATA_REPO, repo_type="dataset")

        # Filter for files in this agent's folder
        agent_pattern = f"{agent_identifier}/"
        agent_files = [f for f in files if f.startswith(agent_pattern) and f.endswith('.jsonl')]

        # Filter by date range (extract date from filename)
        recent_files = []
        for filename in agent_files:
            try:
                # Extract date from filename: YYYY.MM.DD.jsonl
                parts = filename.split('/')
                if len(parts) != 2:
                    continue

                date_part = parts[1].replace('.jsonl', '')  # Get YYYY.MM.DD
                date_components = date_part.split('.')
                if len(date_components) != 3:
                    continue

                file_year, file_month, file_day = map(int, date_components)
                file_date = datetime(file_year, file_month, file_day, tzinfo=timezone.utc)

                # Include if within configured time frame
                if cutoff_date <= file_date <= today:
                    recent_files.append(filename)
            except Exception:
                continue

        return recent_files

    except Exception as e:
        print(f"Error getting daily files: {str(e)}")
        return []




# =============================================================================
# HUGGINGFACE DATASET OPERATIONS
# =============================================================================

def load_agents_from_hf():
    """Load all agent metadata JSON files from HuggingFace dataset."""
    try:
        api = HfApi()
        agents = []

        # List all files in the repository
        files = list_repo_files_with_backoff(api=api, repo_id=AGENTS_REPO, repo_type="dataset")

        # Filter for JSON files only
        json_files = [f for f in files if f.endswith('.json')]

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
                        continue

                    # Extract github_identifier from filename (e.g., "claude[bot].json" -> "claude[bot]")
                    filename_identifier = json_file.replace('.json', '')

                    # Add or override github_identifier to match filename
                    agent_data['github_identifier'] = filename_identifier

                    agents.append(agent_data)

            except Exception as e:
                print(f"Warning: Could not load {json_file}: {str(e)}")
                continue

        print(f"✓ Loaded {len(agents)} agents from HuggingFace")
        return agents

    except Exception as e:
        print(f"Could not load agents from HuggingFace: {str(e)}")
        return None




def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


def upload_with_retry(api, path_or_fileobj, path_in_repo, repo_id, repo_type, token, max_retries=5):
    """
    Upload file to HuggingFace with exponential backoff retry logic.

    Args:
        api: HfApi instance
        path_or_fileobj: Local file path to upload
        path_in_repo: Target path in the repository
        repo_id: Repository ID
        repo_type: Type of repository (e.g., "dataset")
        token: HuggingFace token
        max_retries: Maximum number of retry attempts

    Returns:
        True if upload succeeded, raises exception if all retries failed
    """
    delay = 2.0  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token
            )
            if attempt > 0:
                print(f"   ✓ Upload succeeded on attempt {attempt + 1}/{max_retries}")
            return True

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = delay + random.uniform(0, 1.0)
                print(f"   ⚠️ Upload failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                print(f"   ⏳ Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                delay = min(delay * 2, 60.0)  # Exponential backoff, max 60s
            else:
                print(f"   ✗ Upload failed after {max_retries} attempts: {str(e)}")
                raise


def save_agent_to_hf(data):
    """Save a new agent to HuggingFace dataset as {identifier}.json in root."""
    try:
        api = HfApi()
        token = get_hf_token()

        if not token:
            raise Exception("No HuggingFace token found. Please set HF_TOKEN in your Space settings.")

        identifier = data['github_identifier']
        filename = f"{identifier}.json"

        # Save locally first
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        try:
            # Upload to HuggingFace (root directory)
            upload_with_retry(
                api=api,
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=AGENTS_REPO,
                repo_type="dataset",
                token=token
            )
            print(f"✓ Saved agent to HuggingFace: {filename}")
            return True
        finally:
            # Always clean up local file, even if upload fails
            if os.path.exists(filename):
                os.remove(filename)

    except Exception as e:
        print(f"✗ Error saving agent: {str(e)}")
        return False


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
        api = HfApi()
        token = get_hf_token()

        if not token:
            raise Exception("No HuggingFace token found. Please set HF_TOKEN in your Space settings.")

        filename = "swe-review.json"

        # Combine leaderboard and monthly metrics
        combined_data = {
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'leaderboard': leaderboard_dict,
            'monthly_metrics': monthly_metrics,
            'metadata': {
                'leaderboard_time_frame_days': LEADERBOARD_TIME_FRAME_DAYS,
                'update_time_frame_days': UPDATE_TIME_FRAME_DAYS
            }
        }

        # Save locally first
        with open(filename, 'w') as f:
            json.dump(combined_data, f, indent=2)

        try:
            # Upload to HuggingFace
            upload_with_retry(
                api=api,
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=LEADERBOARD_REPO,
                repo_type="dataset",
                token=token
            )
            print(f"✓ Saved leaderboard data to HuggingFace: {filename}")
            return True
        finally:
            # Always clean up local file, even if upload fails
            if os.path.exists(filename):
                os.remove(filename)

    except Exception as e:
        print(f"✗ Error saving leaderboard data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def load_leaderboard_data_from_hf():
    """
    Load leaderboard data and monthly metrics from HuggingFace dataset.

    Returns:
        dict: Dictionary with 'leaderboard', 'monthly_metrics', and 'last_updated' keys
              Returns None if file doesn't exist or error occurs
    """
    try:
        token = get_hf_token()
        filename = "swe-review.json"

        # Download file
        file_path = hf_hub_download_with_backoff(
            repo_id=LEADERBOARD_REPO,
            filename=filename,
            repo_type="dataset",
            token=token
        )

        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        last_updated = data.get('last_updated', 'Unknown')
        print(f"✓ Loaded leaderboard data from HuggingFace (last updated: {last_updated})")

        return data

    except Exception as e:
        print(f"⚠️ Could not load leaderboard data from HuggingFace: {str(e)}")
        return None


def save_leaderboard_and_metrics_to_hf():
    """
    Creates a comprehensive JSON file with both leaderboard stats and monthly metrics.
    If the file exists, it will be overwritten.

    Returns:
        bool: True if successful, False otherwise
    """
    import io

    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi(token=token)

        print(f"\n{'='*80}")
        print(f"📊 Preparing leaderboard and metrics data for upload...")
        print(f"{'='*80}\n")

        # Get leaderboard data from review metadata
        print("   Constructing leaderboard data from review metadata...")
        leaderboard_data = construct_leaderboard_from_metadata()

        # Get monthly metrics data (all agents, not just top N)
        print("   Calculating monthly metrics from review metadata...")
        monthly_metrics = calculate_monthly_metrics_by_agent(top_n=None)

        # Combine into a single structure
        combined_data = {
            "leaderboard": leaderboard_data,
            "monthly_metrics": monthly_metrics,
            "metadata": {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "time_frame_days": LEADERBOARD_TIME_FRAME_DAYS,
                "total_agents": len(leaderboard_data)
            }
        }

        print(f"   Leaderboard entries: {len(leaderboard_data)}")
        print(f"   Monthly metrics for: {len(monthly_metrics['agents'])} agents")
        print(f"   Time frame: {LEADERBOARD_TIME_FRAME_DAYS} days")

        # Convert to JSON and create file-like object
        json_content = json.dumps(combined_data, indent=2)
        file_like_object = io.BytesIO(json_content.encode('utf-8'))

        # Upload to HuggingFace (will overwrite if exists)
        print(f"\n🤗 Uploading to {LEADERBOARD_REPO}...")
        upload_file_with_backoff(
            api=api,
            path_or_fileobj=file_like_object,
            path_in_repo="swe-review.json",
            repo_id=LEADERBOARD_REPO,
            repo_type="dataset",
            token=token,
            commit_message=f"Update leaderboard data - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )

        print(f"   ✓ Successfully uploaded swe-review.json")
        print(f"{'='*80}\n")

        return True

    except Exception as e:
        print(f"✗ Error saving leaderboard and metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        return False



# =============================================================================
# DATA MANAGEMENT
# =============================================================================

def mine_all_agents():
    """
    Mine review metadata for all agents within UPDATE_TIME_FRAME_DAYS and save to HuggingFace.
    Uses BATCHED BigQuery queries for all agents (efficient approach).
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
    print(f"Time frame: Last {UPDATE_TIME_FRAME_DAYS} days")
    print(f"Data source: BigQuery + GitHub Archive (BATCHED QUERIES)")
    print(f"{'='*80}\n")

    # Initialize BigQuery client
    try:
        client = get_bigquery_client()
    except Exception as e:
        print(f"✗ Failed to initialize BigQuery client: {str(e)}")
        return

    # Define time range: past UPDATE_TIME_FRAME_DAYS (excluding today)
    current_time = datetime.now(timezone.utc)
    end_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=UPDATE_TIME_FRAME_DAYS)

    try:
        # Use batched approach for better performance
        # upload_immediately=True means each batch uploads to HuggingFace right after BigQuery completes
        all_metadata = fetch_all_pr_metadata_batched(
            client, identifiers, start_date, end_date, batch_size=100, upload_immediately=True
        )

        # Calculate summary statistics
        total_prs = sum(len(metadata_list) for metadata_list in all_metadata.values())
        agents_with_data = sum(1 for metadata_list in all_metadata.values() if metadata_list)

        print(f"\n{'='*80}")
        print(f"✅ BigQuery mining and upload complete!")
        print(f"   Total agents: {len(agents)}")
        print(f"   Agents with data: {agents_with_data}")
        print(f"   Total PRs found: {total_prs}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"✗ Error during BigQuery fetch: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # After mining is complete, save leaderboard and metrics to HuggingFace
    print(f"📤 Uploading leaderboard and metrics data...")
    if save_leaderboard_and_metrics_to_hf():
        print(f"✓ Leaderboard and metrics successfully uploaded to {LEADERBOARD_REPO}")
    else:
        print(f"⚠️ Failed to upload leaderboard and metrics data")


def construct_leaderboard_from_metadata():
    """
    Construct leaderboard from stored review metadata instead of fetching all reviews.
    Much more memory-efficient and faster.

    Returns dictionary of agent stats.
    """
    print("📊 Constructing leaderboard from review metadata...")

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
        bot_metadata = [review for review in all_metadata if review.get("agent_identifier") == identifier]

        # Calculate stats
        stats = calculate_review_stats_from_metadata(bot_metadata)

        cache_dict[identifier] = {
            'name': agent_name,
            'name': agent_name,  # Store both for compatibility
            'website': agent.get('website', 'N/A'),
            'github_identifier': identifier,
            **stats
        }

    print(f"✓ Constructed cache with {len(cache_dict)} agent entries")

    return cache_dict


# =============================================================================
# UI FUNCTIONS
# =============================================================================

def create_monthly_metrics_plot(top_n=5):
    """
    Create a Plotly figure with dual y-axes showing:
    - Left y-axis: Acceptance Rate (%) as line curves
    - Right y-axis: Total Reviews created as bar charts

    Each agent gets a unique color for both their line and bars.

    Args:
        top_n: Number of top agents to show (default: 5)
    """
    # Try loading from saved dataset first
    saved_data = load_leaderboard_data_from_hf()

    if saved_data and 'monthly_metrics' in saved_data:
        metrics = saved_data['monthly_metrics']
        print(f"📈 Loaded monthly metrics from saved dataset")

        # Apply top_n filter if specified
        if top_n is not None and top_n > 0 and metrics.get('agents'):
            # Calculate total reviews for each agent
            agent_totals = []
            for agent_name in metrics['agents']:
                agent_data = metrics['data'].get(agent_name, {})
                total_reviews = sum(agent_data.get('total_reviews', []))
                agent_totals.append((agent_name, total_reviews))

            # Sort by total reviews and take top N
            agent_totals.sort(key=lambda x: x[1], reverse=True)
            top_agents = [agent_name for agent_name, _ in agent_totals[:top_n]]

            # Filter metrics to only include top agents
            metrics = {
                'agents': top_agents,
                'months': metrics['months'],
                'data': {agent: metrics['data'][agent] for agent in top_agents if agent in metrics['data']}
            }
    else:
        # Fallback: calculate from metadata if saved data doesn't exist
        print(f"📈 Saved data not available, calculating monthly metrics from metadata...")
        metrics = calculate_monthly_metrics_by_agent(top_n=top_n)

    if not metrics['agents'] or not metrics['months']:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=None,
            xaxis_title=None,
            height=500
        )
        return fig

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Generate unique colors for many agents using HSL color space
    def generate_color(index, total):
        """Generate distinct colors using HSL color space for better distribution"""
        hue = (index * 360 / total) % 360
        saturation = 70 + (index % 3) * 10  # Vary saturation slightly
        lightness = 45 + (index % 2) * 10   # Vary lightness slightly
        return f'hsl({hue}, {saturation}%, {lightness}%)'

    agents = metrics['agents']
    months = metrics['months']
    data = metrics['data']

    # Generate colors for all agents
    agent_colors = {agent: generate_color(idx, len(agents)) for idx, agent in enumerate(agents)}

    # Add traces for each agent
    for idx, agent_name in enumerate(agents):
        color = agent_colors[agent_name]
        agent_data = data[agent_name]

        # Add line trace for acceptance rate (left y-axis)
        acceptance_rates = agent_data['acceptance_rates']
        # Filter out None values for plotting
        x_acceptance = [month for month, rate in zip(months, acceptance_rates) if rate is not None]
        y_acceptance = [rate for rate in acceptance_rates if rate is not None]

        if x_acceptance and y_acceptance:  # Only add trace if there's data
            fig.add_trace(
                go.Scatter(
                    x=x_acceptance,
                    y=y_acceptance,
                    name=agent_name,
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    legendgroup=agent_name,
                    showlegend=(top_n is not None and top_n <= 10),  # Show legend for top N agents
                    hovertemplate='<b>Agent: %{fullData.name}</b><br>' +
                                 'Month: %{x}<br>' +
                                 'Acceptance Rate: %{y:.2f}%<br>' +
                                 '<extra></extra>'
                ),
                secondary_y=False
            )

        # Add bar trace for total reviews (right y-axis)
        # Only show bars for months where agent has reviews
        x_bars = []
        y_bars = []
        for month, count in zip(months, agent_data['total_reviews']):
            if count > 0:  # Only include months with reviews
                x_bars.append(month)
                y_bars.append(count)

        if x_bars and y_bars:  # Only add trace if there's data
            fig.add_trace(
                go.Bar(
                    x=x_bars,
                    y=y_bars,
                    name=agent_name,
                    marker=dict(color=color, opacity=0.6),
                    legendgroup=agent_name,
                    showlegend=False,  # Hide duplicate legend entry (already shown in Scatter)
                    hovertemplate='<b>Agent: %{fullData.name}</b><br>' +
                                 'Month: %{x}<br>' +
                                 'Total Reviews: %{y}<br>' +
                                 '<extra></extra>',
                    offsetgroup=agent_name  # Group bars by agent for proper spacing
                ),
                secondary_y=True
            )

    # Update axes labels
    fig.update_xaxes(title_text=None)
    fig.update_yaxes(
        title_text="<b>Acceptance Rate (%)</b>",
        range=[0, 100],
        secondary_y=False,
        showticklabels=True,
        tickmode='linear',
        dtick=10,
        showgrid=True
    )
    fig.update_yaxes(title_text="<b>Total Reviews</b>", secondary_y=True)

    # Update layout
    show_legend = (top_n is not None and top_n <= 10)
    fig.update_layout(
        title=None,
        hovermode='closest',  # Show individual agent info on hover
        barmode='group',
        height=600,
        showlegend=show_legend,
        margin=dict(l=50, r=150 if show_legend else 50, t=50, b=50)  # More right margin when legend is shown
    )

    return fig


def get_leaderboard_dataframe():
    """
    Load leaderboard from saved dataset and convert to pandas DataFrame for display.
    Falls back to constructing from metadata if saved data is not available.
    Returns formatted DataFrame sorted by total reviews.
    """
    # Try loading from saved dataset first
    saved_data = load_leaderboard_data_from_hf()

    if saved_data and 'leaderboard' in saved_data:
        cache_dict = saved_data['leaderboard']
        print(f"📊 Loaded leaderboard from saved dataset (last updated: {saved_data.get('last_updated', 'Unknown')})")
    else:
        # Fallback: construct from metadata if saved data doesn't exist
        print(f"📊 Saved data not available, constructing leaderboard from metadata...")
        cache_dict = construct_leaderboard_from_metadata()

    print(f"📊 Cache dict size: {len(cache_dict)}")

    if not cache_dict:
        print("⚠️ WARNING: cache_dict is empty!")
        # Return empty DataFrame with correct columns if no data
        column_names = [col[0] for col in LEADERBOARD_COLUMNS]
        return pd.DataFrame(columns=column_names)

    rows = []
    filtered_count = 0
    for identifier, data in cache_dict.items():
        total_reviews = data.get('total_reviews', 0)
        print(f"   Agent '{identifier}': {total_reviews} reviews")

        # Filter out agents with zero total reviews
        if total_reviews == 0:
            filtered_count += 1
            continue

        # Only include display-relevant fields
        rows.append([
            data.get('name', 'Unknown'),
            data.get('website', 'N/A'),
            total_reviews,
            data.get('merged_prs', 0),
            data.get('acceptance_rate', 0.0),
        ])

    print(f"📉 Filtered out {filtered_count} agents with 0 reviews")
    print(f"📈 Leaderboard will show {len(rows)} agents")

    # Create DataFrame
    column_names = [col[0] for col in LEADERBOARD_COLUMNS]
    df = pd.DataFrame(rows, columns=column_names)

    # Ensure numeric types
    numeric_cols = ["Total Reviews", "Merged PRs", "Acceptance Rate (%)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Sort by Total Reviews descending
    if "Total Reviews" in df.columns and not df.empty:
        df = df.sort_values(by="Total Reviews", ascending=False).reset_index(drop=True)

    print(f"✅ Final DataFrame shape: {df.shape}")
    print("="*60 + "\n")

    return df


def submit_agent(identifier, agent_name, developer, website):
    """
    Submit a new agent to the leaderboard.
    Validates input, saves submission, and fetches PR metadata (memory-efficient).
    """
    # Validate required fields
    if not identifier or not identifier.strip():
        return "❌ GitHub identifier is required", get_leaderboard_dataframe()
    if not agent_name or not agent_name.strip():
        return "❌ Agent name is required", get_leaderboard_dataframe()
    if not developer or not developer.strip():
        return "❌ Developer name is required", get_leaderboard_dataframe()
    if not website or not website.strip():
        return "❌ Website URL is required", get_leaderboard_dataframe()

    # Clean inputs
    identifier = identifier.strip()
    agent_name = agent_name.strip()
    developer = developer.strip()
    website = website.strip()

    # Validate GitHub identifier
    is_valid, message = validate_github_username(identifier)
    if not is_valid:
        return f"❌ {message}", get_leaderboard_dataframe()

    # Check for duplicates by loading agents from HuggingFace
    agents = load_agents_from_hf()
    if agents:
        existing_names = {agent['github_identifier'] for agent in agents}
        if identifier in existing_names:
            return f"⚠️ Agent with identifier '{identifier}' already exists", get_leaderboard_dataframe()

    # Create submission
    submission = {
        'name': agent_name,
        'developer': developer,
        'github_identifier': identifier,
        'website': website,
    }

    # Save to HuggingFace
    if not save_agent_to_hf(submission):
        return "❌ Failed to save submission", get_leaderboard_dataframe()

    # Reconstruct and save leaderboard data with new agent
    try:
        print(f"📊 Reconstructing leaderboard with new agent...")
        leaderboard_dict = construct_leaderboard_from_metadata()
        monthly_metrics = calculate_monthly_metrics_by_agent()
        save_leaderboard_data_to_hf(leaderboard_dict, monthly_metrics)
        print(f"✓ Leaderboard data updated")
    except Exception as e:
        print(f"⚠️ Failed to update leaderboard data: {str(e)}")

    # Return success message - data will be populated by daily incremental updates
    return f"✅ Successfully submitted {agent_name}! Review data will be populated by the next daily incremental update.", get_leaderboard_dataframe()


# =============================================================================
# GRADIO APPLICATION
# =============================================================================

print(f"\n🚀 Starting SWE Agent PR Leaderboard")
print(f"   Leaderboard time frame: {LEADERBOARD_TIME_FRAME_DAYS} days ({LEADERBOARD_TIME_FRAME_DAYS // 30} months)")
print(f"   Mining update frequency: Every {UPDATE_TIME_FRAME_DAYS} days\n")

# Start APScheduler for monthly PR mining at 12:00 AM UTC every 1st of the month
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    mine_all_agents,
    trigger=CronTrigger(day=1, hour=0, minute=0),  # 12:00 AM UTC every 1st of the month
    id='monthly_review_mining',
    name='Monthly Review Mining',
    replace_existing=True
)
scheduler.start()
print(f"\n{'='*80}")
print(f"✓ Scheduler initialized successfully")
print(f"⛏️  Mining schedule: Every 1st of the month at 12:00 AM UTC")
print(f"📥 On startup: Only loads cached data from HuggingFace (no mining)")
print(f"{'='*80}\n")

# Create Gradio interface
with gr.Blocks(title="SWE Agent Review Leaderboard", theme=gr.themes.Soft()) as app:
    total_months = LEADERBOARD_TIME_FRAME_DAYS // 30

    gr.Markdown("# 🏆 SWE Agent Review Leaderboard")
    gr.Markdown(f"Track and compare GitHub PR review acceptance statistics for SWE agents")
    
    with gr.Tabs():

        # Leaderboard Tab
        with gr.Tab("📊 Leaderboard"):
            gr.Markdown(f"*All statistics are based on reviews from the last {total_months} months*")
            leaderboard_table = Leaderboard(
                value=pd.DataFrame(columns=[col[0] for col in LEADERBOARD_COLUMNS]),  # Empty initially
                datatype=LEADERBOARD_COLUMNS,
                search_columns=["Agent Name", "Website"],
                filter_columns=[
                    ColumnFilter(
                        "Acceptance Rate (%)",
                        min=0,
                        max=100,
                        default=[0, 100],
                        type="slider",
                        label="Acceptance Rate (%)"
                    )
                ]
            )

            # Load leaderboard data when app starts
            app.load(
                fn=get_leaderboard_dataframe,
                inputs=[],
                outputs=[leaderboard_table]
            )

            # Monthly Metrics Section
            gr.Markdown("---")  # Divider
            gr.Markdown("### 📈 Monthly Performance - Top 5 Agents")
            gr.Markdown("*Shows acceptance rate trends and review volumes for the most active agents*")

            monthly_metrics_plot = gr.Plot(label="Monthly Metrics")

            # Load monthly metrics when app starts
            app.load(
                fn=lambda: create_monthly_metrics_plot(),
                inputs=[],
                outputs=[monthly_metrics_plot]
            )


        # Submit Agent Tab
        with gr.Tab("➕ Submit Agent"):
            
            gr.Markdown("### Submit Your Agent")
            gr.Markdown("Fill in the details below to add your agent to the leaderboard. Make sure you're logged in to HuggingFace CLI on your machine.")
            
            with gr.Row():
                with gr.Column():
                    github_input = gr.Textbox(
                        label="GitHub Identifier*",
                        placeholder="Your agent username (e.g., my-agent-bot)"
                    )
                    name_input = gr.Textbox(
                        label="Agent Name*",
                        placeholder="Your agent's display name"
                    )
                
                with gr.Column():
                    developer_input = gr.Textbox(
                        label="Developer*",
                        placeholder="Your developer or team name"
                    )
                    website_input = gr.Textbox(
                        label="Website",
                        placeholder="https://your-agent-website.com"
                    )
            
            submit_button = gr.Button(
                "Submit Agent",
                variant="primary"
            )
            submission_status = gr.Textbox(
                label="Submission Status",
                interactive=False
            )
            
            # Event handler
            submit_button.click(
                fn=submit_agent,
                inputs=[github_input, name_input, developer_input, website_input],
                outputs=[submission_status, leaderboard_table]
            )


# Launch application
if __name__ == "__main__":
    app.launch()