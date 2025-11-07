import gradio as gr
from gradio_leaderboard import Leaderboard
import json
import os
import time
import tempfile
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset, Dataset
import threading
from dotenv import load_dotenv
import pandas as pd
import random
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from google.cloud import bigquery

# Load environment variables
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='SWE Agent Review Leaderboard')
args = parser.parse_args()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/swe_agents"  # HuggingFace dataset for agent metadata
REVIEW_METADATA_REPO = "SWE-Arena/review_metadata"  # HuggingFace dataset for review metadata
LEADERBOARD_TIME_FRAME_DAYS = 30  # Time frame for leaderboard

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
        # Parse the date string (handles both with and without microseconds)
        if '.' in date_string:
            # Old format with microseconds
            dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        else:
            # Already in correct format or GitHub format
            return date_string
        
        # Convert to standardized format
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        print(f"Warning: Could not parse date '{date_string}': {e}")
        return date_string


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
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.url') as url,
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


def fetch_pr_status_from_bigquery(client, urls, start_date, end_date):
    """
    Fetch PR status (merged/closed) from GitHub Archive PullRequestEvent.

    For each PR URL, looks for PullRequestEvent with action='closed' to determine
    if the PR was merged or just closed.

    Args:
        client: BigQuery client instance
        urls: List of PR URLs to check status for
        start_date: Start datetime (should cover review period and after)
        end_date: End datetime (should be recent/current)

    Returns:
        Dictionary mapping PR URL to status dict:
        {
            'url': {
                'status': 'merged'|'closed'|'open',
                'merged': bool,
                'closed_at': timestamp or None
            }
        }
    """
    if not urls:
        return {}

    print(f"\n🔍 Querying BigQuery for PR status ({len(urls)} PRs)...")

    # Extract repo and PR number from URLs
    # URL format: https://github.com/owner/repo/pull/123
    pr_info = []
    for url in urls:
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
            JSON_EXTRACT_SCALAR(payload, '$.pull_request.url') as url,
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
            url = row.url

            merged = row.merged if row.merged is not None else False
            closed_at = row.closed_at or row.merged_at

            # Convert to ISO format if datetime
            if hasattr(closed_at, 'isoformat'):
                closed_at = closed_at.isoformat()

            status = 'merged' if merged else 'closed'

            status_map[url] = {
                'status': status,
                'merged': merged,
                'closed_at': closed_at
            }

        # Mark remaining PRs as open
        for url in urls:
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
        return {url: {'status': 'open', 'merged': False, 'closed_at': None} for url in urls}


def extract_review_metadata_from_bigquery(review_row, status_info):
    """
    Extract minimal PR review metadata from BigQuery row and status info.

    Args:
        review_row: BigQuery row from PullRequestReviewEvent query
        status_info: Status dictionary from fetch_pr_status_from_bigquery

    Returns:
        Dictionary with review metadata
    """
    url = review_row.url
    pr_number = review_row.pr_number
    reviewed_at = review_row.reviewed_at or review_row.created_at

    # Convert to ISO format if datetime
    if hasattr(reviewed_at, 'isoformat'):
        reviewed_at = reviewed_at.isoformat()

    return {
        'url': url,
        'reviewed_at': reviewed_at,
        'pr_status': status_info['status'],
        'merged_at': status_info['merged'],
        'closed_at': status_info['closed_at'],
        'url': url,
        'review_id': f"pr_{pr_number}"
    }


# =============================================================================
# GITHUB API OPERATIONS
# =============================================================================

def request_with_backoff(method, url, *, headers=None, params=None, json_body=None, data=None, max_retries=10, timeout=30, token_pool=None, token=None):
    """
    Perform an HTTP request with exponential backoff and jitter for GitHub API.
    Retries on 403/429 (rate limits), 5xx server errors, and transient network exceptions.

    Args:
        token_pool: Optional TokenPool instance for rate limit tracking
        token: Optional token string to mark as rate-limited if 403/429 occurs

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
                reset_timestamp = None

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

                # Mark token as rate-limited if we have token pool and token
                if status in (403, 429) and token_pool and token:
                    token_pool.mark_rate_limited(token, reset_timestamp)

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

def get_github_tokens():
    """Get all GitHub tokens from environment variables (all vars starting with GITHUB_TOKEN)."""
    tokens = []
    for key, value in os.environ.items():
        if key.startswith('GITHUB_TOKEN') and value:
            tokens.append(value)

    if not tokens:
        print("Warning: No GITHUB_TOKEN found. API rate limits: 60/hour (authenticated: 5000/hour)")
    else:
        print(f"✓ Loaded {len(tokens)} GitHub token(s) for rotation")

    return tokens


def get_github_token():
    """Get first GitHub token from environment variables (backward compatibility)."""
    tokens = get_github_tokens()
    return tokens[0] if tokens else None


class TokenPool:
    """
    Hybrid token pool with parallel execution and round-robin fallback.

    Splits tokens into two pools:
    - Parallel pool (50%): For concurrent API calls to maximize throughput
    - Round-robin pool (50%): Backup pool for rate limit fallback

    Features:
    - Automatic fallback when parallel tokens hit rate limits
    - Rate limit tracking with timestamp-based recovery
    - Thread-safe token management
    - Real-time statistics monitoring
    """
    def __init__(self, tokens):
        import threading

        self.all_tokens = tokens if tokens else [None]
        self.lock = threading.Lock()

        # Split tokens into parallel and round-robin pools (50/50)
        total_tokens = len(self.all_tokens)
        split_point = max(1, total_tokens // 2)

        self.parallel_tokens = self.all_tokens[:split_point]
        self.roundrobin_tokens = self.all_tokens[split_point:] if total_tokens > 1 else self.all_tokens

        # Round-robin index for fallback pool
        self.roundrobin_index = 0

        # Rate limit tracking: {token: reset_timestamp}
        self.parallel_rate_limited = set()
        self.roundrobin_rate_limited = set()
        self.rate_limit_resets = {}

        # Statistics
        self.stats = {
            'parallel_calls': 0,
            'roundrobin_calls': 0,
            'fallback_triggers': 0
        }

        print(f"📊 Token Pool Initialized:")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Parallel pool: {len(self.parallel_tokens)} tokens")
        print(f"   Round-robin pool: {len(self.roundrobin_tokens)} tokens")

    def _cleanup_expired_rate_limits(self):
        """Remove tokens from rate-limited sets if their reset time has passed."""
        current_time = time.time()
        expired_tokens = [
            token for token, reset_time in self.rate_limit_resets.items()
            if current_time >= reset_time
        ]

        for token in expired_tokens:
            self.parallel_rate_limited.discard(token)
            self.roundrobin_rate_limited.discard(token)
            del self.rate_limit_resets[token]
            if expired_tokens:
                print(f"   ✓ Recovered {len(expired_tokens)} token(s) from rate limit")

    def get_parallel_token(self):
        """Get an available token from the parallel pool."""
        with self.lock:
            self._cleanup_expired_rate_limits()

            # Find first non-rate-limited parallel token
            for token in self.parallel_tokens:
                if token not in self.parallel_rate_limited:
                    self.stats['parallel_calls'] += 1
                    return token

            return None

    def get_roundrobin_token(self):
        """Get the next available token from round-robin pool."""
        with self.lock:
            self._cleanup_expired_rate_limits()

            # Try all tokens in round-robin order
            attempts = 0
            while attempts < len(self.roundrobin_tokens):
                token = self.roundrobin_tokens[self.roundrobin_index]
                self.roundrobin_index = (self.roundrobin_index + 1) % len(self.roundrobin_tokens)

                if token not in self.roundrobin_rate_limited:
                    self.stats['roundrobin_calls'] += 1
                    return token

                attempts += 1

            return None

    def get_next_token(self):
        """
        Get next available token, trying parallel pool first, then falling back to round-robin.

        Returns:
            Token string or None if all tokens are rate-limited
        """
        # Try parallel pool first
        token = self.get_parallel_token()
        if token:
            return token

        # Fallback to round-robin pool
        with self.lock:
            self.stats['fallback_triggers'] += 1

        token = self.get_roundrobin_token()
        if not token:
            print("   ⚠️ All tokens are rate-limited, waiting...")

        return token

    def get_headers(self):
        """Get headers with the next available token."""
        token = self.get_next_token()
        return {'Authorization': f'token {token}'} if token else {}

    def mark_rate_limited(self, token, reset_timestamp=None):
        """
        Mark a token as rate-limited with optional reset timestamp.

        Args:
            token: The token to mark as rate-limited
            reset_timestamp: Unix timestamp when rate limit resets (optional)
        """
        if not token:
            return

        with self.lock:
            # Determine which pool the token belongs to
            if token in self.parallel_tokens:
                self.parallel_rate_limited.add(token)
            if token in self.roundrobin_tokens:
                self.roundrobin_rate_limited.add(token)

            # Store reset timestamp if provided
            if reset_timestamp:
                self.rate_limit_resets[token] = reset_timestamp
                reset_time = datetime.fromtimestamp(reset_timestamp, tz=timezone.utc)
                print(f"   ⏰ Token rate-limited until {reset_time.strftime('%H:%M:%S')} UTC")

    def get_available_parallel_tokens(self):
        """Get list of all available (non-rate-limited) parallel tokens."""
        with self.lock:
            self._cleanup_expired_rate_limits()
            return [t for t in self.parallel_tokens if t not in self.parallel_rate_limited]

    def get_stats(self):
        """Get token pool usage statistics."""
        with self.lock:
            return {
                'parallel_calls': self.stats['parallel_calls'],
                'roundrobin_calls': self.stats['roundrobin_calls'],
                'fallback_triggers': self.stats['fallback_triggers'],
                'parallel_rate_limited': len(self.parallel_rate_limited),
                'roundrobin_rate_limited': len(self.roundrobin_rate_limited)
            }

    def print_stats(self):
        """Print token pool usage statistics."""
        stats = self.get_stats()
        total_calls = stats['parallel_calls'] + stats['roundrobin_calls']

        print(f"\n📊 Token Pool Statistics:")
        print(f"   Total API calls: {total_calls}")
        if total_calls > 0:
            print(f"   Parallel calls: {stats['parallel_calls']} ({stats['parallel_calls']/total_calls*100:.1f}%)")
            print(f"   Round-robin calls: {stats['roundrobin_calls']} ({stats['roundrobin_calls']/total_calls*100:.1f}%)")
        print(f"   Fallback triggers: {stats['fallback_triggers']}")
        print(f"   Currently rate-limited: {stats['parallel_rate_limited']} parallel, {stats['roundrobin_rate_limited']} round-robin")


def validate_github_username(identifier):
    """Verify that a GitHub identifier exists with backoff-aware requests."""
    try:
        token = get_github_token()
        headers = {'Authorization': f'token {token}'} if token else {}
        url = f'https://api.github.com/users/{identifier}'
        response = request_with_backoff('GET', url, headers=headers, max_retries=1)
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


def fetch_reviews_with_time_partition(base_query, start_date, end_date, token_pool, prs_by_url, depth=0):
    """
    Fetch reviews within a specific time range using time-based partitioning.
    Recursively splits the time range if hitting the 1000-result limit.
    Supports splitting by day, hour, minute, and second as needed.

    Args:
        depth: Current recursion depth (for tracking)

    Returns the number of reviews found in this time partition.
    """
    # Calculate time difference
    time_diff = end_date - start_date
    total_seconds = time_diff.total_seconds()

    # Determine granularity and format dates accordingly
    if total_seconds >= 86400:  # >= 1 day
        # Use day granularity (YYYY-MM-DD)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
    elif total_seconds >= 3600:  # >= 1 hour but < 1 day
        # Use hour granularity (YYYY-MM-DDTHH:MM:SSZ)
        start_str = start_date.strftime('%Y-%m-%dT%H:00:00Z')
        end_str = end_date.strftime('%Y-%m-%dT%H:59:59Z')
    elif total_seconds >= 60:  # >= 1 minute but < 1 hour
        # Use minute granularity (YYYY-MM-DDTHH:MM:SSZ)
        start_str = start_date.strftime('%Y-%m-%dT%H:%M:00Z')
        end_str = end_date.strftime('%Y-%m-%dT%H:%M:59Z')
    else:  # < 1 minute
        # Use second granularity (YYYY-MM-DDTHH:MM:SSZ)
        start_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Add date range to query (use created for PR search)
    query = f'{base_query} created:{start_str}..{end_str}'

    indent = "  " + "  " * depth
    print(f"{indent}Searching range {start_str} to {end_str}...")

    page = 1
    per_page = 100
    total_in_partition = 0

    while True:
        url = 'https://api.github.com/search/issues'  # Use issues endpoint for PR search
        params = {
            'q': query,
            'per_page': per_page,
            'page': page,
            'sort': 'created',
            'order': 'asc'
        }
        token = token_pool.get_next_token()
        headers = {'Authorization': f'token {token}'} if token else {}

        try:
            response = request_with_backoff('GET', url, headers=headers, params=params, token_pool=token_pool, token=token)
            if response is None:
                print(f"{indent}  Error: retries exhausted for range {start_str} to {end_str}")
                return total_in_partition

            if response.status_code != 200:
                print(f"{indent}  Error: HTTP {response.status_code} for range {start_str} to {end_str}")
                return total_in_partition

            data = response.json()
            total_count = data.get('total_count', 0)
            items = data.get('items', [])

            if not items:
                break

            # Add PR reviews to global dict (keyed by PR URL)
            for pr in items:
                url = pr.get('url')
                pr_number = pr.get('number')
                # Use PR URL as unique key (more reliable than number alone)
                if url and url not in prs_by_url:
                    prs_by_url[url] = pr
                    total_in_partition += 1

            # Check if we hit the 1000-result limit
            if total_count > 1000 and page == 10:
                print(f"{indent}  ⚠️ Hit 1000-result limit ({total_count} total). Splitting time range...")

                # Determine how to split based on time range duration
                if total_seconds < 2:  # Less than 2 seconds - can't split further
                    print(f"{indent}  ⚠️ Cannot split further (range < 2 seconds). Some results may be missing.")
                    break

                elif total_seconds < 120:  # Less than 2 minutes - split by seconds
                    # Split into 2-4 parts depending on range
                    num_splits = min(4, max(2, int(total_seconds / 30)))
                    split_duration = time_diff / num_splits
                    split_dates = [start_date + split_duration * i for i in range(num_splits + 1)]

                    total_from_splits = 0
                    for i in range(num_splits):
                        split_start = split_dates[i]
                        split_end = split_dates[i + 1]
                        # Avoid overlapping ranges (add 1 second to start)
                        if i > 0:
                            split_start = split_start + timedelta(seconds=1)

                        count = fetch_reviews_with_time_partition(
                            base_query, split_start, split_end, token_pool, prs_by_url, depth + 1
                        )
                        total_from_splits += count

                    return total_from_splits

                elif total_seconds < 7200:  # Less than 2 hours - split by minutes
                    # Split into 2-4 parts
                    num_splits = min(4, max(2, int(total_seconds / 1800)))
                    split_duration = time_diff / num_splits
                    split_dates = [start_date + split_duration * i for i in range(num_splits + 1)]

                    total_from_splits = 0
                    for i in range(num_splits):
                        split_start = split_dates[i]
                        split_end = split_dates[i + 1]
                        # Avoid overlapping ranges (add 1 minute to start)
                        if i > 0:
                            split_start = split_start + timedelta(minutes=1)

                        count = fetch_reviews_with_time_partition(
                            base_query, split_start, split_end, token_pool, prs_by_url, depth + 1
                        )
                        total_from_splits += count

                    return total_from_splits

                elif total_seconds < 172800:  # Less than 2 days - split by hours
                    # Split into 2-4 parts
                    num_splits = min(4, max(2, int(total_seconds / 43200)))
                    split_duration = time_diff / num_splits
                    split_dates = [start_date + split_duration * i for i in range(num_splits + 1)]

                    total_from_splits = 0
                    for i in range(num_splits):
                        split_start = split_dates[i]
                        split_end = split_dates[i + 1]
                        # Avoid overlapping ranges (add 1 hour to start)
                        if i > 0:
                            split_start = split_start + timedelta(hours=1)

                        count = fetch_reviews_with_time_partition(
                            base_query, split_start, split_end, token_pool, prs_by_url, depth + 1
                        )
                        total_from_splits += count

                    return total_from_splits

                else:  # 2+ days - split by days
                    days_diff = time_diff.days

                    # Use aggressive splitting for large ranges or deep recursion
                    # Split into 4 parts if range is > 30 days, otherwise split in half
                    if days_diff > 30 or depth > 5:
                        # Split into 4 parts for more aggressive partitioning
                        quarter_diff = time_diff / 4
                        split_dates = [
                            start_date,
                            start_date + quarter_diff,
                            start_date + quarter_diff * 2,
                            start_date + quarter_diff * 3,
                            end_date
                        ]

                        total_from_splits = 0
                        for i in range(4):
                            split_start = split_dates[i]
                            split_end = split_dates[i + 1]
                            # Avoid overlapping ranges
                            if i > 0:
                                split_start = split_start + timedelta(days=1)

                            count = fetch_reviews_with_time_partition(
                                base_query, split_start, split_end, token_pool, prs_by_url, depth + 1
                            )
                            total_from_splits += count

                        return total_from_splits
                    else:
                        # Binary split for smaller ranges
                        mid_date = start_date + time_diff / 2

                        # Recursively fetch both halves
                        count1 = fetch_reviews_with_time_partition(
                            base_query, start_date, mid_date, token_pool, prs_by_url, depth + 1
                        )
                        count2 = fetch_reviews_with_time_partition(
                            base_query, mid_date + timedelta(days=1), end_date, token_pool, prs_by_url, depth + 1
                        )

                        return count1 + count2

            # Normal pagination: check if there are more pages
            if len(items) < per_page or page >= 10:
                break

            page += 1
            time.sleep(0.5)  # Courtesy delay between pages

        except Exception as e:
            print(f"{indent}  Error fetching range {start_str} to {end_str}: {str(e)}")
            return total_in_partition

    if total_in_partition > 0:
        print(f"{indent}  ✓ Found {total_in_partition} reviews in range {start_str} to {end_str}")

    return total_in_partition


def fetch_reviews_parallel(query_patterns, start_date, end_date, token_pool, prs_by_url):
    """
    Fetch reviews for multiple query patterns in parallel using available parallel tokens.

    This function uses ThreadPoolExecutor to execute multiple query patterns concurrently,
    with each pattern using a dedicated token from the parallel pool. Falls back to
    sequential execution if insufficient parallel tokens are available.

    Args:
        query_patterns: List of query pattern strings (e.g., ['is:pr author:bot1', 'is:pr reviewed-by:bot1'])
        start_date: Start datetime for time range
        end_date: End datetime for time range
        token_pool: TokenPool instance for token management
        prs_by_url: Dictionary to collect PRs by URL (shared across patterns)

    Returns:
        Total number of PRs found across all patterns
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # Check how many parallel tokens are available
    available_tokens = token_pool.get_available_parallel_tokens()

    if len(available_tokens) < 2 or len(query_patterns) < 2:
        # Not enough tokens or patterns for parallelization, use sequential
        print(f"   ⚠️ Sequential execution: {len(available_tokens)} parallel tokens available for {len(query_patterns)} patterns")
        total_found = 0
        for pattern in query_patterns:
            pattern_prs = {}
            count = fetch_reviews_with_time_partition(
                pattern, start_date, end_date, token_pool, pattern_prs, depth=0
            )
            # Merge pattern results into global dict
            with threading.Lock():
                for url, pr in pattern_prs.items():
                    if url not in prs_by_url:
                        prs_by_url[url] = pr
            total_found += count
        return total_found

    # Use parallel execution
    print(f"   🚀 Parallel execution: {len(available_tokens)} parallel tokens for {len(query_patterns)} patterns")

    # Thread-safe lock for updating prs_by_url
    lock = threading.Lock()

    def fetch_pattern(pattern):
        """Fetch reviews for a single pattern (runs in parallel)."""
        pattern_prs = {}
        try:
            count = fetch_reviews_with_time_partition(
                pattern, start_date, end_date, token_pool, pattern_prs, depth=0
            )
            return pattern, pattern_prs, count
        except Exception as e:
            print(f"   Error fetching pattern '{pattern}': {str(e)}")
            return pattern, {}, 0

    # Execute patterns in parallel
    max_workers = min(len(query_patterns), len(available_tokens))
    total_found = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all patterns
        future_to_pattern = {
            executor.submit(fetch_pattern, pattern): pattern
            for pattern in query_patterns
        }

        # Collect results as they complete
        for future in as_completed(future_to_pattern):
            pattern = future_to_pattern[future]
            try:
                _, pattern_prs, count = future.result()

                # Merge results into global dict (thread-safe)
                with lock:
                    for url, pr in pattern_prs.items():
                        if url not in prs_by_url:
                            prs_by_url[url] = pr

                total_found += count
                print(f"   ✓ Pattern '{pattern}' completed: {count} PRs found")

            except Exception as e:
                print(f"   ✗ Pattern '{pattern}' failed: {str(e)}")

    return total_found


def extract_review_metadata(pr):
    """
    Extract minimal PR review metadata for efficient storage.
    Only keeps essential fields: url, reviewed_at, pr_status, merged_at, closed_at.
    Note: agent_name is not stored as it's inferred from the folder structure.

    PR status:
    - pr_status: 'open', 'merged', or 'closed'
    - merged_at: True if PR was merged, False otherwise
    - closed_at: Date when PR was closed/merged (if applicable)

    Merged PR = PR that was merged after agent review
    Rejected PR = PR that was closed without merging after agent review
    """
    # Extract PR metadata from search results
    # The GitHub search API returns PR data from /search/issues endpoint
    url = pr.get('url')
    pr_number = pr.get('number')
    created_at = pr.get('created_at')
    closed_at = pr.get('closed_at')
    state = pr.get('state', 'open')  # open or closed

    # Check if PR has pull_request field (indicates it's a PR, not an issue)
    pull_request_data = pr.get('pull_request', {})

    # For initial extraction, we don't know if merged yet
    # This will be updated by update_pr_status function
    merged_at = pull_request_data.get('merged_at') is not None if pull_request_data else False

    # Determine initial status
    if merged_at:
        status = 'merged'
    elif state == 'closed':
        status = 'closed'
    else:
        status = 'open'

    return {
        'url': url,
        'reviewed_at': created_at,  # When the PR was created (agent reviewed it)
        'pr_status': status,
        'merged_at': merged_at,
        'closed_at': closed_at,
        'review_id': f"pr_{pr_number}"  # Use PR number for deduplication
    }


def update_pr_status(metadata_list, token_pool):
    """
    Update PR status for reviews to get current merged/closed state.

    For each PR associated with a review, fetch current status from GitHub API.
    Updates metadata_list in-place with PR status information.

    Args:
        metadata_list: List of review metadata dictionaries
        token_pool: TokenPool instance for rotating tokens

    Returns:
        Updated metadata_list with current PR status
    """
    if not metadata_list:
        return metadata_list

    # Track unique PRs to avoid duplicate API calls
    url_to_status = {}
    updated_count = 0

    for metadata in metadata_list:
        url = metadata.get('url')
        if not url:
            continue

        # Skip if already fetched for this PR
        if url in url_to_status:
            status_info = url_to_status[url]
            metadata['pr_status'] = status_info['status']
            metadata['merged_at'] = status_info['merged']
            metadata['closed_at'] = status_info['closed_at']
            continue

        try:
            # Convert HTML URL to API URL
            # https://github.com/owner/repo/pull/123 -> https://api.github.com/repos/owner/repo/pulls/123
            parts = url.replace('https://github.com/', '').split('/')
            if len(parts) >= 4:
                owner, repo, pull_word, pr_number = parts[0], parts[1], parts[2], parts[3]
                api_url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}'

                token = token_pool.get_next_token()
                headers = {'Authorization': f'token {token}'} if token else {}
                response = request_with_backoff('GET', api_url, headers=headers, max_retries=3, token_pool=token_pool, token=token)

                if response and response.status_code == 200:
                    pr_data = response.json()
                    state = pr_data.get('state', 'open')
                    merged = pr_data.get('merged', False)
                    closed_at = pr_data.get('closed_at')
                    merged_at = pr_data.get('merged_at')

                    # Determine final status
                    if merged:
                        status = 'merged'
                    elif state == 'closed':
                        status = 'closed'
                    else:
                        status = 'open'

                    status_info = {
                        'status': status,
                        'merged': merged,
                        'closed_at': closed_at or merged_at
                    }

                    # Cache and update
                    url_to_status[url] = status_info
                    metadata['pr_status'] = status
                    metadata['merged_at'] = merged
                    metadata['closed_at'] = closed_at or merged_at
                    updated_count += 1

                # Small delay to avoid rate limiting
                time.sleep(0.1)

        except Exception as e:
            print(f"   Warning: Could not check PR status for {url}: {e}")
            continue

    if updated_count > 0:
        print(f"   ✓ Updated status for {updated_count} unique PRs")

    return metadata_list




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

            # Count merged PRs (merged)
            merged_count = sum(1 for review in reviews_in_month
                                if review.get('pr_status') == 'merged')

            # Count rejected PRs (closed without merging)
            rejected_count = sum(1 for review in reviews_in_month
                                if review.get('pr_status') == 'closed')

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

    This function APPENDS new metadata and DEDUPLICATES by review_id.
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
        files = api.list_repo_files(repo_id=REVIEW_METADATA_REPO, repo_type="dataset")

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

                file_path = hf_hub_download(
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

        # DEBUG: Show unique agent identifiers found in review folders
        if agent_identifiers_found:
            print(f"📋 Agent identifiers found in review metadata folders:")
            for identifier in sorted(agent_identifiers_found):
                count = sum(1 for r in all_metadata if r.get('agent_identifier') == identifier)
                print(f"   - '{identifier}': {count} reviews")

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
        files = api.list_repo_files(repo_id=REVIEW_METADATA_REPO, repo_type="dataset")

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
                file_path = hf_hub_download(
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
        files = api.list_repo_files(repo_id=REVIEW_METADATA_REPO, repo_type="dataset")

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




def fetch_review_current_status(review_url, token):
    """
    Fetch the current revert status of a single review from GitHub API.

    Args:
        token: GitHub API token
        token: GitHub API token

    Returns:
        Dictionary with updated is_reverted and revert_at, or None if failed
    """
    try:
        # Convert HTML URL to API URL
        # https://github.com/owner/repo/reviews/123 -> https://api.github.com/repos/owner/repo/reviews/123
        parts = review_url.replace('https://github.com/', '').split('/')
        if len(parts) < 4:
            return None

        owner, repo, review_word, review_number = parts[0], parts[1], parts[2], parts[3]
        api_url = f'https://api.github.com/repos/{owner}/{repo}/reviews/{review_number}'

        headers = {'Authorization': f'token {token}'} if token else {}
        response = request_with_backoff('GET', api_url, headers=headers, max_retries=3)

        if response is None or response.status_code != 200:
            return None

        review_data = response.json()
        state = review_data.get('state')
        state_reason = review_data.get('state_reason')
        closed_at = review_data.get('closed_at')

        return {
            'state': state,
            'state_reason': state_reason,
            'closed_at': closed_at
        }

    except Exception as e:
        print(f"   Error fetching review status for {review_url}: {str(e)}")
        return None


def refresh_review_status_for_agent(agent_identifier, token):
    """
    Refresh status for all open reviews from the last month for an agent.
    Only updates reviews that are still open (state="open" or no state_reason).

    This implements the smart update strategy:
    - Skip reviews that are already closed/resolved
    - Fetch current status for open reviews
    - Update and save back to daily files

    Args:
        agent_identifier: GitHub identifier of the agent
        token: GitHub API token

    Returns:
        Tuple: (total_checked, updated_count)
    """
    print(f"\n🔄 Refreshing open reviews for {agent_identifier} (last month)...")

    try:
        # Get daily files from configured time frame
        recent_files = get_daily_files_last_time_frame(agent_identifier)

        if not recent_files:
            print(f"   No recent files found for {agent_identifier}")
            return (0, 0)

        print(f"   Found {len(recent_files)} daily files to check")

        total_checked = 0
        updated_count = 0

        # Process each file
        for filename in recent_files:
            try:
                # Download file
                file_path = hf_hub_download(
                    repo_id=REVIEW_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=get_hf_token()
                )
                reviews = load_jsonl(file_path)

                if not reviews:
                    continue

                updated_reviews = []
                file_had_updates = False

                # Check each review
                for review in reviews:
                    # Skip if already closed (has a state_reason)
                    if review.get("is_reverted"):
                        updated_reviews.append(review)
                        continue

                    # Review may have been reverted, check status
                    review_url = review.get("url")

                    if not review_url:
                        updated_reviews.append(review)
                        continue

                    current_status = fetch_review_current_status(review_url, token)

                    if current_status:
                        # Check if status changed (now closed)
                        if current_status['state'] == 'closed':
                            print(f"   ✓ Review status changed: {review_url}")
                            review['state'] = current_status['state']
                            review['state_reason'] = current_status['state_reason']
                            review['closed_at'] = current_status['closed_at']
                            updated_count += 1
                            file_had_updates = True

                    updated_reviews.append(review)
                    time.sleep(0.1)  # Rate limiting courtesy delay

                # Save file if there were updates
                if file_had_updates:
                    # Extract filename components for local save
                    parts = filename.split('/')
                    local_filename = parts[-1]  # Just YYYY.MM.DD.jsonl

                    # Save locally
                    save_jsonl(local_filename, updated_reviews)

                    try:
                        # Upload back to HuggingFace
                        api = HfApi()
                        upload_with_retry(
                            api=api,
                            path_or_fileobj=local_filename,
                            path_in_repo=filename,
                            repo_id=REVIEW_METADATA_REPO,
                            repo_type="dataset",
                            token=get_hf_token()
                        )
                        print(f"   💾 Updated {filename}")
                    finally:
                        # Always clean up local file, even if upload fails
                        if os.path.exists(local_filename):
                            os.remove(local_filename)

            except Exception as e:
                print(f"   Warning: Could not process {filename}: {str(e)}")
                continue

        print(f"   ✅ Refresh complete: {total_checked} open reviews checked, {updated_count} updated")
        return (total_checked, updated_count)

    except Exception as e:
        print(f"   ✗ Error refreshing reviews for {agent_identifier}: {str(e)}")
        return (0, 0)


# =============================================================================
# HUGGINGFACE DATASET OPERATIONS
# =============================================================================

def load_agents_from_hf():
    """Load all agent metadata JSON files from HuggingFace dataset."""
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

                    # Extract github_identifier from filename (e.g., "claude[bot].json" -> "claude[bot]")
                    filename_identifier = json_file.replace('.json', '')

                    # Add or override github_identifier to match filename
                    agent_data['github_identifier'] = filename_identifier

                    # DEBUG: Log the identifier being used
                    print(f"   ✓ Loaded agent: '{filename_identifier}' -> {agent_data.get('name', 'Unknown')}")

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




# =============================================================================
# DATA MANAGEMENT
# =============================================================================

def update_all_agents_incremental():
    """
    Weekly scheduled task for incremental review mining and statistics update.

    Strategy:
    1. Update PR status for all existing metadata (last LEADERBOARD_TIME_FRAME_DAYS - 7)
    2. Fetch last week's new reviews
    3. Save all updated/new metadata back to HuggingFace
    4. Reload statistics from updated metadata
    """
    print(f"\n{'='*80}")
    print(f"🕛 Weekly Incremental Update started at {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*80}")

    try:
        # Fetch and update reviews
        fetch_and_update_weekly_reviews()

        # Reload statistics from updated metadata
        print(f"\n📋 Reloading statistics from updated review metadata...")
        construct_leaderboard_from_metadata()

        print(f"\n{'='*80}")
        print(f"📊 Update Summary:")
        print(f"   ✓ Updated existing review statuses")
        print(f"   ✓ Fetched last week's new reviews")
        print(f"   ✓ Statistics reloaded")
        print(f"{'='*80}")

        print(f"\n✅ Weekly Incremental Update completed at {datetime.now(timezone.utc).isoformat()}")

    except Exception as e:
        print(f"✗ Weekly update failed: {str(e)}")
        import traceback
        traceback.print_exc()


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

    # Debug: Check what agent_identifiers exist in review metadata
    if all_metadata:
        review_identifiers = set(r.get('agent_identifier') for r in all_metadata if r.get('agent_identifier'))
        print(f"   Unique agent_identifiers in reviews: {review_identifiers}")
    else:
        print("⚠️ No review metadata loaded!")

    cache_dict = {}

    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('name', 'Unknown')

        # Filter metadata for this agent
        agent_metadata = [review for review in all_metadata if review.get("agent_identifier") == identifier]

        # Debug output
        if len(agent_metadata) > 0:
            print(f"   ✓ Agent '{identifier}' matched {len(agent_metadata)} reviews")

        # Calculate stats
        stats = calculate_review_stats_from_metadata(agent_metadata)

        cache_dict[identifier] = {
            'agent_name': agent_name,
            'website': agent.get('website', 'N/A'),
            'github_identifier': identifier,
            **stats
        }

    print(f"✓ Constructed cache with {len(cache_dict)} agent entries")

    return cache_dict


# =============================================================================
# UI FUNCTIONS
# =============================================================================

def create_monthly_metrics_plot(top_n=None):
    """
    Create a Plotly figure with dual y-axes showing:
    - Left y-axis: Acceptance Rate (%) as line curves
    - Right y-axis: Total Reviews created as bar charts

    Each agent gets a unique color for both their line and bars.

    Args:
        top_n: If specified, only show metrics for the top N agents by total reviews.
    """
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
    fig.update_yaxes(title_text="<b>Acceptance Rate (%)</b>", range=[0, 100], secondary_y=False)
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
    Construct leaderboard from review metadata and convert to pandas DataFrame for display.
    Returns formatted DataFrame sorted by retention rate.
    """
    print("\n" + "="*60)
    print("🔍 DEBUG: get_leaderboard_dataframe() called")
    print("="*60)

    # Construct leaderboard from metadata
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
            data.get('agent_name', 'Unknown'),
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

    # Sort by Acceptance Rate (%) descending
    if "Acceptance Rate (%)" in df.columns and not df.empty:
        df = df.sort_values(by="Acceptance Rate (%)", ascending=False).reset_index(drop=True)

    print(f"✅ Final DataFrame shape: {df.shape}")
    print("="*60 + "\n")

    return df


def submit_agent(identifier, agent_name, organization, description, website):
    """
    Submit a new agent to the leaderboard.
    Validates input, saves submission, and fetches PR metadata (memory-efficient).
    """
    # Validate required fields
    if not identifier or not identifier.strip():
        return "❌ GitHub identifier is required", get_leaderboard_dataframe()
    if not agent_name or not agent_name.strip():
        return "❌ Agent name is required", get_leaderboard_dataframe()
    if not organization or not organization.strip():
        return "❌ Organization name is required", get_leaderboard_dataframe()
    if not website or not website.strip():
        return "❌ Website URL is required", get_leaderboard_dataframe()

    # Clean inputs
    identifier = identifier.strip()
    agent_name = agent_name.strip()
    organization = organization.strip()
    description = description.strip()
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
        'agent_name': agent_name,
        'organization': organization,
        'github_identifier': identifier,
        'description': description,
        'website': website,
    }

    # Save to HuggingFace
    if not save_agent_to_hf(submission):
        return "❌ Failed to save submission", get_leaderboard_dataframe()

    # Return success message - data will be populated by daily incremental updates
    return f"✅ Successfully submitted {agent_name}! Review data will be populated by the next daily incremental update.", get_leaderboard_dataframe()


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

def fetch_and_update_weekly_reviews():
    """
    Fetch and update reviews with comprehensive status checking using BigQuery.

    Strategy:
    1. For each agent:
       - Examine ALL open reviews from last LEADERBOARD_TIME_FRAME_DAYS - 7 for their closed_at status
       - Update PR status for all existing metadata using BigQuery (last LEADERBOARD_TIME_FRAME_DAYS - 7)
       - Fetch new reviews from last week using BigQuery
       - Save all updated/new metadata back to HuggingFace
    """
    # Initialize BigQuery client
    try:
        client = get_bigquery_client()
    except Exception as e:
        print(f"✗ Failed to initialize BigQuery client: {str(e)}")
        return

    # Load all agents
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found in HuggingFace dataset")
        return

    # Calculate date range
    today_utc = datetime.now(timezone.utc)
    today_midnight = datetime.combine(today_utc.date(), datetime.min.time(), tzinfo=timezone.utc)
    last_week_midnight = today_midnight - timedelta(days=7)
    cutoff_date = today_midnight - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS - 7)

    print(f"📅 Time Range Configuration:")
    print(f"   Last week 12am UTC: {last_week_midnight.isoformat()}")
    print(f"   Today 12am UTC: {today_midnight.isoformat()}")
    print(f"   Cutoff for existing reviews: {cutoff_date.isoformat()}")
    print(f"   Examining reviews from: {cutoff_date.date()} to {today_midnight.date()}")

    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('name', 'Unknown')

        if not identifier:
            print(f"Warning: Skipping agent without identifier: {agent}")
            continue

        try:
            print(f"\n{'='*60}")
            print(f"Processing: {agent_name} ({identifier})")
            print(f"{'='*60}")

            # Step 1: Load all existing metadata within timeframe
            print(f"📊 Loading existing metadata from last {LEADERBOARD_TIME_FRAME_DAYS - 1} days...")
            all_metadata = load_review_metadata()
            agent_metadata = [r for r in all_metadata if r.get("agent_identifier") == identifier]

            # Filter to last LEADERBOARD_TIME_FRAME_DAYS - 1 days (from cutoff to today)
            recent_metadata = []
            for review in agent_metadata:
                reviewed_at = review.get('reviewed_at', '')
                if reviewed_at:
                    try:
                        review_date = datetime.fromisoformat(reviewed_at.replace('Z', '+00:00'))
                        if cutoff_date <= review_date < today_midnight:
                            recent_metadata.append(review)
                    except Exception as e:
                        print(f"   Warning: Could not parse date '{reviewed_at}': {e}")
                        continue

            print(f"   ✓ Loaded {len(recent_metadata)} existing reviews from timeframe")

            # Step 2: Update PR status for existing reviews using BigQuery
            if recent_metadata:
                print(f"🔍 Updating PR status for {len(recent_metadata)} existing reviews using BigQuery...")
                # Extract PR URLs from existing metadata
                urls = [r.get('url') for r in recent_metadata if r.get('url')]
                if urls:
                    # Fetch status from BigQuery
                    extended_end_date = today_utc
                    status_map = fetch_pr_status_from_bigquery(client, urls, cutoff_date, extended_end_date)

                    # Update metadata with new status
                    for review in recent_metadata:
                        url = review.get('url')
                        if url and url in status_map:
                            status_info = status_map[url]
                            review['pr_status'] = status_info['status']
                            review['merged_at'] = status_info['merged']
                            review['closed_at'] = status_info['closed_at']

                    print(f"   ✓ Updated PR status for existing reviews")

            # Step 3: Fetch NEW reviews from last week to today using BigQuery
            print(f"🔍 Fetching new reviews from {last_week_midnight.isoformat()} to {today_midnight.isoformat()} using BigQuery...")

            review_rows = fetch_reviews_from_bigquery(client, identifier, last_week_midnight, today_midnight)

            # Extract unique PR URLs and fetch status
            urls = list(set([row.url for row in review_rows if row.url]))
            print(f"   Found {len(review_rows)} review events across {len(urls)} unique PRs")

            # Fetch PR status for new reviews
            extended_end_date = today_utc
            status_map = fetch_pr_status_from_bigquery(client, urls, last_week_midnight, extended_end_date)

            # Extract metadata for new reviews
            weekly_metadata = []
            seen_prs = set()
            for row in review_rows:
                url = row.url
                if url in seen_prs:
                    continue
                seen_prs.add(url)

                status_info = status_map.get(url, {
                    'status': 'open',
                    'merged': False,
                    'closed_at': None
                })

                metadata = extract_review_metadata_from_bigquery(row, status_info)
                metadata['agent_identifier'] = identifier
                weekly_metadata.append(metadata)

            print(f"   ✓ Found {len(weekly_metadata)} unique PRs in 7-day window")

            # Step 4: Combine and save all metadata
            all_updated_metadata = recent_metadata + weekly_metadata

            if all_updated_metadata:
                print(f"💾 Saving {len(all_updated_metadata)} total reviews to HuggingFace...")
                save_review_metadata_to_hf(all_updated_metadata, identifier)
                print(f"✓ Updated {identifier}: {len(recent_metadata)} existing (status checked) + {len(weekly_metadata)} new = {len(all_updated_metadata)} total")
            else:
                print(f"   No reviews to save for {identifier}")

        except Exception as e:
            print(f"✗ Error processing {identifier}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


# =============================================================================
# GRADIO APPLICATION
# =============================================================================

# Start APScheduler for weekly updates at 12:00 AM UTC every Monday
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    update_all_agents_incremental,
    trigger=CronTrigger(day_of_week='mon', hour=0, minute=0),  # 12:00 AM UTC every Monday
    id='weekly_review_mining',
    name='Weekly Regular Review Mining',
    replace_existing=True
)
scheduler.start()
print("✓ Scheduler started: Weekly Incremental Update at 12:00 AM UTC every Monday (updates existing metadata + mines last week's reviews)")

# Create Gradio interface
with gr.Blocks(title="SWE Agent Review Leaderboard", theme=gr.themes.Soft()) as app:

    gr.Markdown("# 🏆 SWE Agent Review Leaderboard")
    gr.Markdown("Track and compare GitHub PR review acceptance statistics for SWE agents (last month)")
    
    with gr.Tabs():

        # Leaderboard Tab
        with gr.Tab("📊 Leaderboard"):
            gr.Markdown("*All statistics are based on reviews from the last month*")
            leaderboard_table = Leaderboard(
                value=pd.DataFrame(columns=[col[0] for col in LEADERBOARD_COLUMNS]),  # Empty initially
                datatype=LEADERBOARD_COLUMNS,
                search_columns=["Agent Name", "Website"],
                filter_columns=["Acceptance Rate (%)"]
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
                fn=lambda: create_monthly_metrics_plot(top_n=5),
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
                    organization_input = gr.Textbox(
                        label="Organization*",
                        placeholder="Your organization or team name"
                    )
                    description_input = gr.Textbox(
                        label="Description",
                        placeholder="Brief description of your agent",
                        lines=3
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
                inputs=[github_input, name_input, organization_input, description_input, website_input],
                outputs=[submission_status, leaderboard_table]
            )


# Launch application
if __name__ == "__main__":
    app.launch()