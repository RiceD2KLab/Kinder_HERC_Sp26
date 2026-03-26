"""Data structures used across the web scraping pipeline.

Provides the Source dataclass representing a district + URL input pair.
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Source:
    """
    A single user-provided input source describing where meeting videos live.

    A "source" is the smallest unit of input the pipeline processes. It maps one district to
    one URL that *may* contain one or many meeting videos (playlist/index/vendor page).

    Parameters
    ----------
    district : str
        Human-readable district name (e.g., "Houston ISD").
        This becomes the subdirectory name under the output root:
            School Board Meetings/<district>/

    url : str
        URL to a playlist/video/index page/direct media/etc.
        Examples:
          - YouTube playlist: https://www.youtube.com/playlist?list=...
          - Legistar calendar: https://<district>.legistar.com/Calendar.aspx
          - District "Board Meetings" page: https://www.district.org/board/meetings
          - Direct MP4: https://cdn.site.org/meetings/2024-10-01.mp4
          - HLS stream: https://cdn.site.org/meetings/master.m3u8

    Returns
    -------
    Source
        A Source instance (dataclass) that is used by the pipeline.
    """
    district: str
    url: str