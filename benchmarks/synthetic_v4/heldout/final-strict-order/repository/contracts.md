# strictly_increasing contract

Return True only when every adjacent value is strictly greater than its predecessor; equality must return False. Preserve empty/singleton True, integer-only validation including rejection of bool, full validation before comparison, one-pass generators and unchanged input.

Behavior expectations are visible in tests/test_requirements.py; existing compatibility is in tests/test_compatibility.py.
