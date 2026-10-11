import json


def validate_event(payload, seen_ids=()):
    if not isinstance(payload, dict):
        raise TypeError("event payload must be an object")
    identifier = payload.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("event id must be a nonempty string")
    if payload.get("kind") not in {"created", "restocked"}:
        raise ValueError("unsupported event kind")
    if identifier in seen_ids:
        raise ValueError("event id was already seen")
    return dict(payload)


def encode_event(event):
    return json.dumps(event)
