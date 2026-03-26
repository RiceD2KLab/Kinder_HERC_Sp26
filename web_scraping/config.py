"""Configuration constants for the school board meeting scraper.

Contains:
- HTTP request headers (User-Agent)
- Regex patterns for date parsing from titles/URLs
- Month-name-to-number mapping
"""

from __future__ import annotations

import re
from typing import Dict, List

## HTTP header for requests to avoid bot blockers
USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Extract dates from titles
DATE_PATTERNS: List[Pattern[str]] = [
    # Example matches: 2024-09-12, 2024_09_12, 2024/09/12
    re.compile(r"(?P<y>\d{4})[-_/](?P<m>\d{2})[-_/](?P<d>\d{2})"),

    # Example matches: 09-12-2024, 09/12/2024
    re.compile(r"(?P<m>\d{2})[-/](?P<d>\d{2})[-/](?P<y>\d{4})"),

    # Example matches:
    #   "Sep 12 2024"
    #   "Sept 12, 2024"
    #   "September 12 2024"  (the [a-z]* allows longer month names)
    re.compile(
        r"(?P<mon>jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
        r"[a-z]*\s+(?P<d>\d{1,2}),?\s+(?P<y>\d{4})",
        re.IGNORECASE,
    ),
]

# Map month abbreviations to month numbers used by datetime.date.
MONTH_MAP: Dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}