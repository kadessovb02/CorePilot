import math


class RemoteError(RuntimeError):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"remote status {status_code}")


def build_headers(api_token, request_id=None):
    if api_token is not None and not isinstance(api_token, str):
        raise TypeError("api_token must be a string or None")
    headers = {"Accept": "application/json"}
    if api_token is not None:
        headers["X-API-TOKEN"] = api_token
    if request_id is not None:
        headers["X-REQUEST-ID"] = request_id
    return headers


def retry_delay(attempt, base_delay=0.25, max_delay=4.0):
    if not isinstance(attempt, int):
        raise TypeError("attempt must be an integer")
    if attempt < 0:
        raise ValueError("attempt must be nonnegative")
    base_delay, max_delay = float(base_delay), float(max_delay)
    if not math.isfinite(base_delay) or not math.isfinite(max_delay) or min(base_delay, max_delay) < 0:
        raise ValueError("delays must be finite and nonnegative")
    return min(base_delay * (attempt + 1), max_delay)


def decode_response(status_code, payload):
    if status_code == 200:
        return payload
    raise RemoteError(status_code)
