# resolve_index contract

Convert a negative index by adding length exactly once, so -1 denotes the final position. Preserve positive indices, integer-only validation (bool rejected), negative-length ValueError, and out-of-range/empty IndexError.

Behavior expectations are visible in tests/test_requirements.py; existing compatibility is in tests/test_compatibility.py.
