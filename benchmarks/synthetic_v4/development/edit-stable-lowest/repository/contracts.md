# select_lowest contract

Return the first record attaining the minimum integer score, by original object identity; tied later records must not replace it. Preserve one-pass iterable/generator handling, input immutability, KeyError for absent score, TypeError for noninteger/bool scores, and ValueError for empty input.

Behavior expectations are visible in tests/test_requirements.py; existing compatibility is in tests/test_compatibility.py.
