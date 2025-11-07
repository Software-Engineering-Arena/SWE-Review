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

def fetch_all_pr_metadata_single_query(client, identifiers, start_date, end_date):
    """
    Fetch PR review metadata for ALL agents using ONE comprehensive BigQuery query.
    
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
    
    # Generate table UNION statements for PR status (look back 1 year to catch all closures)
    status_start_date = end_date - timedelta(days=365)
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
            
            # Convert datetime objects to ISO strings
            reviewed_at = row.reviewed_at
            if hasattr(reviewed_at, 'isoformat'):
                reviewed_at = reviewed_at.isoformat()
                
            merged_at = row.merged_at
            if hasattr(merged_at, 'isoformat'):
                merged_at = merged_at.isoformat()
                
            closed_at = row.closed_at
            if hasattr(closed_at, 'isoformat'):
                closed_at = closed_at.isoformat()
            
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

        api = HfApi()

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

            # Upload entire folder in a single commit
            print(f"   📤 Uploading {len(grouped)} files in single batch commit...")
            api.upload_large_folder(
                folder_path=temp_dir,
                repo_id=REVIEW_METADATA_REPO,
                repo_type="dataset",
                token=token,
                commit_message=f"Update: {agent_identifier} ({len(grouped)} daily files, {len(metadata_list)} total reviews)"
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
    print(f"Data source: BigQuery + GitHub Archive (ONE QUERY FOR ALL AGENTS)")
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
        all_metadata = fetch_all_pr_metadata_single_query(
            client, identifiers, start_date, end_date
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
        agent_name = agent.get('name', agent.get('agent_name', 'Unknown'))
        
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
    
    print(f"\n{'='*80}")
    print(f"✅ Mining complete!")
    print(f"   Total agents: {len(agents)}")
    print(f"   Successfully saved: {success_count}")
    print(f"   No data (skipped): {no_data_count}")
    print(f"   Errors: {error_count}")
    print(f"   BigQuery queries executed: 1")
    print(f"{'='*80}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mine_all_agents()