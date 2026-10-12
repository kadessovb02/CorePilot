# parse_assignment contract

Split only at the first equals sign. Return a two-string tuple with the stripped name and the exact remaining value, including further equals signs and spaces. Preserve TypeError for nonstrings and ValueError for missing separator or blank name.

Behavior expectations are visible in tests/test_requirements.py; existing compatibility is in tests/test_compatibility.py.
