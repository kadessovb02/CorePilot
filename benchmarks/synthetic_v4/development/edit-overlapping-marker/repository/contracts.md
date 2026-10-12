# count_overlapping contract

Count every starting position where marker occurs, including overlapping matches. Matching is exact and case-sensitive. Preserve zero for empty/no-match text, Unicode strings, TypeError for nonstrings, and ValueError for an empty marker.

Behavior expectations are visible in tests/test_requirements.py; existing compatibility is in tests/test_compatibility.py.
