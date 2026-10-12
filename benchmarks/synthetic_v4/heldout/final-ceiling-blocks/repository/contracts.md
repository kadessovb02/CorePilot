# block_count contract

Return the smallest integer number of whole blocks able to hold total items, rounding up any remainder without floating-point arithmetic. Preserve zero total, default block_size 4096, exact multiples, arbitrarily large integers, TypeError for noninteger/bool sizes and ValueError for negative total or nonpositive block size.

Behavior expectations are visible in tests/test_requirements.py; existing compatibility is in tests/test_compatibility.py.
