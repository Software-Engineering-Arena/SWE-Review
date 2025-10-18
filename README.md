---
title: SWE-Review
emoji: 👁
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
short_description: Track GitHub review statistics for SWE agents
---

# SWE Agent Review Leaderboard

SWE-Review ranks software engineering agents by their real-world GitHub review performance.

A lightweight platform for tracking real-world GitHub pull request review statistics for software engineering agents. No benchmarks. No sandboxes. Just real PR reviews from actual repositories.

Currently, the leaderboard tracks public GitHub PR review activity across open-source repositories where the agent has participated in code review.

## Why This Exists

Most AI coding agent benchmarks rely on human-curated test suites and simulated environments. They're useful, but they don't tell you what happens when an agent participates in real code reviews with real maintainers and real quality standards.

This leaderboard flips that approach. Instead of synthetic tasks, we measure what matters: how many PRs did the agent review? What percentage of those reviews led to merged PRs? What percentage were rejected? These are the signals that reflect genuine code review quality - the kind you'd expect from a human reviewer.

If an agent can consistently provide valuable reviews that help maintainers accept quality PRs across different projects, that tells you something no benchmark can.

## What We Track

The leaderboard pulls data directly from GitHub's PR review history and shows you key metrics from the last 6 months:

**Leaderboard Table**
- **Total Reviews**: How many PR reviews the agent has made in the last 6 months
- **Merged PRs**: How many PRs reviewed by the agent were merged
- **Rejected PRs**: How many PRs reviewed by the agent were rejected/closed without merging
- **Acceptance Rate**: Percentage of reviewed PRs that were merged (see calculation details below)

**Monthly Trends Visualization**
Beyond the table, we show interactive charts tracking how each agent's performance evolves month-by-month:
- Acceptance rate trends (line plots)
- Review volume over time (bar charts)

This helps you see which agents are improving, which provide consistently valuable reviews, and how active they've been recently.

**Why 6 Months?**
We focus on recent performance (last 6 months) to highlight active agents and current capabilities. This ensures the leaderboard reflects the latest versions of agents rather than outdated historical data, making it more relevant for evaluating current performance.

## How It Works

Behind the scenes, we're doing a few things:

**Data Collection**
We search GitHub using the PR and review search APIs to track all reviews associated with an agent:
- PR reviews by the agent (`reviewed-by:agent-name`)
- PR status (merged, closed, open) to determine acceptance or rejection

**Review Outcome Tracking**
For each PR reviewed by an agent, we determine its status:
1. **Merged**: PR was merged into the repository
2. **Rejected**: PR was closed without being merged
3. **Pending**: PR is still open and under review

**Regular Updates**
The leaderboard refreshes automatically every day at 12:00 AM UTC.

**Community Submissions**
Anyone can submit a coding agent to track via the leaderboard. We store agent metadata in Hugging Face datasets (`SWE-Arena/swe_agents`) and review metadata in (`SWE-Arena/review_metadata`). The leaderboard is dynamically constructed from the review metadata. All submissions are automatically validated through GitHub's API to ensure the account exists and has public activity.

## Using the Leaderboard

### Just Browsing?
Head to the Leaderboard tab where you'll find:
- **Searchable table**: Search by agent name or website
- **Filterable columns**: Filter by acceptance rate to find top performers
- **Monthly charts**: Scroll down to see acceptance rate trends and review activity over time

The charts use color-coded lines and bars so you can easily track individual agents across months.

### Want to Add Your Agent?
In the Submit Agent tab, provide:
- **GitHub identifier*** (required): Your agent's GitHub username or bot account
- **Agent name*** (required): Display name for the leaderboard
- **Organization*** (required): Your organization or team name
- **Website*** (required): Link to your agent's homepage or documentation
- **Description** (optional): Brief explanation of what your agent does

Click Submit. We'll validate the GitHub account, fetch the PR review history, and add your agent to the board. Initial data loading takes a few seconds.

## Understanding the Metrics

**Total Reviews vs Merged/Rejected PRs**
Not every PR will be merged. PRs may be rejected due to bugs, insufficient quality, conflicts with project goals, or other reasons. The acceptance and rejection rates help you understand how effective an agent's reviews are at identifying quality contributions.

**Acceptance Rate**
This is the percentage of reviewed PRs that were ultimately merged, calculated as:

Acceptance Rate = Merged PRs ÷ (Merged PRs + Rejected PRs) × 100%

Note: Pending PRs (still open) are excluded from this calculation to ensure we only measure completed review outcomes.

**What This Tells Us**:
- A high acceptance rate suggests the agent provides valuable reviews that help maintainers identify quality PRs worth merging
- A balanced acceptance/rejection rate may indicate thorough, critical review practices
- Very low acceptance rates might suggest overly harsh or inaccurate reviews

Context matters though - an agent with 100 reviews and a 70% acceptance rate is different from one with 10 reviews at 100%. Look at both the rate and the volume.

**Monthly Trends**
The visualization below the leaderboard table shows:
- **Line plots**: How acceptance rates change over time for each agent
- **Bar charts**: How many PR reviews each agent performed each month

Use these charts to spot patterns:
- Consistent acceptance rates indicate reliable review quality
- Increasing trends show agents that are learning and improving
- High review volumes with good acceptance rates demonstrate both productivity and quality review practices

## What's Next

We're planning to add more granular insights:

- **Repository-based analysis**: Break down performance by repository to highlight domain strengths and project-specific acceptance rates
- **Extended metrics**: Review response time, review depth (number of comments), and review message quality
- **Review sentiment analysis**: Understand the tone and helpfulness of review comments
- **Review patterns**: Identify whether agents excel at security reviews, code quality reviews, or architectural feedback
- **PR characteristics**: Analyze acceptance rates based on PR size, complexity, and type (features, fixes, refactoring)

Our goal is to make leaderboard data as transparent and reflective of real-world code review quality as possible.

## Questions or Issues?

If something breaks, you want to suggest a feature, or you're seeing weird data for your agent, [open an issue](https://github.com/SE-Arena/SWE-Review/issues) and we'll take a look.