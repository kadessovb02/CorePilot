"""Stage-only response schemas; submissions never grant workspace capabilities."""
from copy import deepcopy


def _object(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


def _text(maximum, minimum=1):
    return {"type": "string", "minLength": minimum, "maxLength": maximum}


def _array(items, maximum):
    return {"type": "array", "items": items, "maxItems": maximum}


_PATH = _text(500)
_SUMMARY = _text(2000)
_PARAMETERS = {
    "select": _object({"read_paths": _array(_PATH, 12), "edit_paths": _array(_PATH, 6),
                       "new_test_paths": _array(_PATH, 2)}),
    "edit": _object({"patches": _array(_object({"path": _PATH, "old_text": _text(64000),
                                              "new_text": _text(64000, 0)}), 12),
                     "summary": _SUMMARY}, ["patches"]),
    "tests": _object({"tests": _array(_object({"path": _PATH, "content": _text(64000)}), 2),
                      "summary": _SUMMARY}, ["tests"]),
    "review": _object({"approved": {"type": "boolean"}, "summary": _SUMMARY,
                       "issues": _array(_text(500), 20)}),
    "inspect": _object({"answer": _text(6000), "references": _array(_object({
        "path": _PATH, "line": {"type": "integer", "minimum": 1}}), 12)}),
}


def stage_tool(stage):
    if stage not in _PARAMETERS:
        raise ValueError("Unknown controlled submission stage")
    return {"type": "function", "function": {"name": "submit_" + stage,
        "description": "Submit the complete " + stage + " stage result as data for host validation. "
                       "This function cannot read, edit files or execute commands.",
        "parameters": deepcopy(_PARAMETERS[stage])}}
