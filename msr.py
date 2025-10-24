"""
Minimalist Review Metadata Mining Script
Mines PR review metadata from GitHub and saves to HuggingFace dataset.
"""

import json
import os
import time
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv
import random

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
                from datetime import datetime, timezone
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


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


# =============================================================================
# GITHUB API FUNCTIONS
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


def fetch_reviews_with_time_partition(base_query, start_date, end_date, token_pool, prs_by_url, depth=0):
    """
    Fetch reviews within a specific time range using time-based partitioning.
    Recursively splits the time range if hitting the 1000-result limit.
    Supports splitting by day, hour, minute, and second as needed.

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
                pr_url = pr.get('html_url')
                if pr_url and pr_url not in prs_by_url:
                    prs_by_url[pr_url] = pr
                    total_in_partition += 1

            # Check if we hit the 1000-result limit
            if total_count > 1000 and page == 10:
                print(f"{indent}  ⚠️ Hit 1000-result limit ({total_count} total). Splitting time range...")

                # Determine how to split based on time range duration
                if total_seconds < 2:  # Less than 2 seconds - can't split further
                    print(f"{indent}  ⚠️ Cannot split further (range < 2 seconds). Some results may be missing.")
                    break

                elif total_seconds < 120:  # Less than 2 minutes - split by seconds
                    num_splits = min(4, max(2, int(total_seconds / 30)))
                    split_duration = time_diff / num_splits
                    split_dates = [start_date + split_duration * i for i in range(num_splits + 1)]

                    total_from_splits = 0
                    for i in range(num_splits):
                        split_start = split_dates[i]
                        split_end = split_dates[i + 1]
                        if i > 0:
                            split_start = split_start + timedelta(seconds=1)

                        count = fetch_reviews_with_time_partition(
                            base_query, split_start, split_end, token_pool, prs_by_url, depth + 1
                        )
                        total_from_splits += count

                    return total_from_splits

                elif total_seconds < 7200:  # Less than 2 hours - split by minutes
                    num_splits = min(4, max(2, int(total_seconds / 1800)))
                    split_duration = time_diff / num_splits
                    split_dates = [start_date + split_duration * i for i in range(num_splits + 1)]

                    total_from_splits = 0
                    for i in range(num_splits):
                        split_start = split_dates[i]
                        split_end = split_dates[i + 1]
                        if i > 0:
                            split_start = split_start + timedelta(minutes=1)

                        count = fetch_reviews_with_time_partition(
                            base_query, split_start, split_end, token_pool, prs_by_url, depth + 1
                        )
                        total_from_splits += count

                    return total_from_splits

                elif total_seconds < 172800:  # Less than 2 days - split by hours
                    num_splits = min(4, max(2, int(total_seconds / 43200)))
                    split_duration = time_diff / num_splits
                    split_dates = [start_date + split_duration * i for i in range(num_splits + 1)]

                    total_from_splits = 0
                    for i in range(num_splits):
                        split_start = split_dates[i]
                        split_end = split_dates[i + 1]
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
            lock = threading.Lock()
            with lock:
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
    Only keeps essential fields: html_url, reviewed_at, pr_status, pr_merged, pr_closed_at.

    PR status:
    - pr_status: 'open', 'merged', or 'closed'
    - pr_merged: True if PR was merged, False otherwise
    - pr_closed_at: Date when PR was closed/merged (if applicable)
    """
    pr_url = pr.get('html_url')
    pr_number = pr.get('number')
    created_at = pr.get('created_at')
    closed_at = pr.get('closed_at')
    state = pr.get('state', 'open')  # open or closed

    # Check if PR has pull_request field (indicates it's a PR, not an issue)
    pull_request_data = pr.get('pull_request', {})
    pr_merged = pull_request_data.get('merged_at') is not None if pull_request_data else False

    # Determine initial status
    if pr_merged:
        status = 'merged'
    elif state == 'closed':
        status = 'closed'
    else:
        status = 'open'

    return {
        'html_url': pr_url,
        'reviewed_at': created_at,  # When the PR was created (agent reviewed it)
        'pr_status': status,
        'pr_merged': pr_merged,
        'pr_closed_at': closed_at,
        'pr_url': pr_url,  # Store PR URL for tracking
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
    pr_url_to_status = {}
    updated_count = 0

    for metadata in metadata_list:
        pr_url = metadata.get('pr_url')
        if not pr_url:
            continue

        # Skip if already fetched for this PR
        if pr_url in pr_url_to_status:
            status_info = pr_url_to_status[pr_url]
            metadata['pr_status'] = status_info['status']
            metadata['pr_merged'] = status_info['merged']
            metadata['pr_closed_at'] = status_info['closed_at']
            continue

        try:
            # Convert HTML URL to API URL
            # https://github.com/owner/repo/pull/123 -> https://api.github.com/repos/owner/repo/pulls/123
            parts = pr_url.replace('https://github.com/', '').split('/')
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
                    pr_url_to_status[pr_url] = status_info
                    metadata['pr_status'] = status
                    metadata['pr_merged'] = merged
                    metadata['pr_closed_at'] = closed_at or merged_at
                    updated_count += 1

                # Small delay to avoid rate limiting
                time.sleep(0.1)

        except Exception as e:
            print(f"   Warning: Could not check PR status for {pr_url}: {e}")
            continue

    if updated_count > 0:
        print(f"   ✓ Updated status for {updated_count} unique PRs")

    return metadata_list


def fetch_all_reviews_metadata(identifier, agent_name, token_pool):
    """
    Fetch PR reviews associated with a GitHub user or bot for the past LEADERBOARD_TIME_FRAME_DAYS.
    Returns lightweight metadata instead of full review objects.

    This function employs time-based partitioning to navigate GitHub's 1000-result limit per query.
    It searches using the query pattern:
    - reviewed-by:{identifier} (PR reviews by the agent)

    After fetching reviews, it updates PR status to determine if PRs were merged or closed.

    Args:
        identifier: GitHub username or bot identifier
        agent_name: Human-readable name of the agent for metadata purposes
        token_pool: TokenPool instance for rotating tokens

    Returns:
        List of dictionaries containing minimal PR review metadata with PR status
    """

    # Define query pattern for PR reviews
    query_patterns = [f'is:pr reviewed-by:{identifier}']

    # Use a dict to deduplicate PRs by URL
    prs_by_url = {}

    # Define time range: past LEADERBOARD_TIME_FRAME_DAYS (excluding today)
    current_time = datetime.now(timezone.utc)
    end_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)  # 12:00 AM UTC today
    start_date = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    print(f"\n🔍 Searching for PRs reviewed by {identifier}")
    print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (today excluded)")
    print(f"   Query patterns: {len(query_patterns)}")

    overall_start_time = time.time()

    # Use parallel execution if multiple patterns and sufficient tokens
    if len(query_patterns) > 1:
        reviews_found = fetch_reviews_parallel(
            query_patterns,
            start_date,
            end_date,
            token_pool,
            prs_by_url
        )
    else:
        # Single pattern, use sequential
        reviews_found = fetch_reviews_with_time_partition(
            query_patterns[0],
            start_date,
            end_date,
            token_pool,
            prs_by_url
        )

    overall_duration = time.time() - overall_start_time
    print(f"   ✓ All patterns complete: {len(prs_by_url)} unique PRs found")
    print(f"   ⏱️ Total time: {overall_duration:.1f} seconds")

    all_prs = list(prs_by_url.values())

    print(f"\n✅ COMPLETE: Found {len(all_prs)} unique PRs reviewed by {identifier}")
    print(f"📦 Extracting minimal metadata and updating PR status...")

    # Extract metadata for each PR review
    metadata_list = [extract_review_metadata(pr) for pr in all_prs]

    # Update PR status to get current merged/closed state
    print(f"🔍 Updating PR status for reviewed PRs...")
    metadata_list = update_pr_status(metadata_list, token_pool)

    # Calculate memory savings
    import sys
    original_size = sys.getsizeof(str(all_prs))
    metadata_size = sys.getsizeof(str(metadata_list))
    savings_pct = ((original_size - metadata_size) / original_size * 100) if original_size > 0 else 0

    print(f"💾 Memory efficiency: {original_size // 1024}KB → {metadata_size // 1024}KB (saved {savings_pct:.1f}%)")

    return metadata_list


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


def upload_with_retry(api, path_or_fileobj, path_in_repo, repo_id, repo_type, token, max_retries=5):
    """
    Upload file to HuggingFace with exponential backoff retry logic.
    """
    delay = 2.0

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
                delay = min(delay * 2, 60.0)
            else:
                print(f"   ✗ Upload failed after {max_retries} attempts: {str(e)}")
                raise


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
    """
    tokens = get_github_tokens()
    token_pool = TokenPool(tokens)

    # Load agent metadata from HuggingFace
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found in HuggingFace dataset")
        return

    print(f"\n{'='*80}")
    print(f"Starting review metadata mining for {len(agents)} agents")
    print(f"Time frame: Last {LEADERBOARD_TIME_FRAME_DAYS} days")
    print(f"{'='*80}\n")

    # Mine each agent
    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('agent_name', 'Unknown')

        if not identifier:
            print(f"Warning: Skipping agent without identifier: {agent}")
            continue

        try:
            print(f"\n{'='*80}")
            print(f"Processing: {agent_name} ({identifier})")
            print(f"{'='*80}")

            # Fetch review metadata
            metadata = fetch_all_reviews_metadata(identifier, agent_name, token_pool)

            if metadata:
                print(f"💾 Saving {len(metadata)} review records...")
                save_review_metadata_to_hf(metadata, identifier)
                print(f"✓ Successfully processed {agent_name}")
            else:
                print(f"   No reviews found for {agent_name}")

        except Exception as e:
            print(f"✗ Error processing {identifier}: {str(e)}")
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
