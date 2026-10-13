"""Host-selected, hash-bound unittest scaffolds for one new regression file.

The operator supplies the template; a model may replace only its placeholder
method. These helpers never write files or claim a behavioral test passed.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re

from .controlled_contract import ContractError
from .edit_regions import RegionError, replace_python_region, select_python_region
from .workspace import safe_path, source_manifest


def validate_scaffold_config(root: Path, value: dict, read_paths) -> dict:
    if not isinstance(value, dict) or set(value) != {"path", "identifier", "sha256"}:
        raise ValueError("regression_scaffold requires only path, identifier and sha256")
    path, identifier, digest = (value[key] for key in ("path", "identifier", "sha256"))
    if (not isinstance(path, str) or not path.endswith(".py") or "\\" in path
            or Path(path).as_posix() != path or any(part in {".", ".."} for part in path.split("/"))):
        raise ValueError("regression_scaffold.path must be a canonical relative Python path")
    target = safe_path(root, path, False)
    if not target.is_file() or path not in read_paths:
        raise ValueError("regression_scaffold must be an existing explicitly approved read_path")
    if (not isinstance(identifier, str) or not re.fullmatch(
            r"[A-Za-z_]\w*\.test_[A-Za-z0-9_]+", identifier, flags=re.ASCII)):
        raise ValueError("regression_scaffold.identifier must name one qualified unittest test method")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("regression_scaffold.sha256 must be an exact lowercase SHA256")
    return {"path": path, "identifier": identifier, "sha256": digest}


def select_regression_scaffold(root: Path, profile: dict, selection: dict) -> dict:
    """Bind an existing approved template and a separate, fixed new test path."""
    try:
        config = validate_scaffold_config(root, profile.get("regression_scaffold"), profile["read_paths"])
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    path = config["path"]
    manifest = source_manifest(root)
    if manifest.get(path) != config["sha256"] or path not in selection["read_paths"]:
        raise ContractError("Approved regression scaffold changed or was not selected for an actual read")
    destinations = selection["new_test_paths"]
    if len(destinations) != 1 or destinations[0] == path or path in selection["edit_paths"]:
        raise ContractError("Regression scaffold is protected; select exactly one separate new test file")
    safe_path(root, destinations[0])
    if not destinations[0].endswith('.py') or not any(
            destinations[0].startswith(test_root + '/') for test_root in profile.get('test_roots', ['tests'])):
        raise ContractError("Regression scaffold destination must be a selected Python test path")
    source = safe_path(root, path, False).read_bytes()
    if len(source) > 64_000 or hashlib.sha256(source).hexdigest() != config["sha256"]:
        raise ContractError("Approved regression scaffold bytes are oversized or stale")
    try:
        tree = ast.parse(source.decode("utf-8-sig"))
        region = select_python_region(source, config["identifier"])
    except (UnicodeError, SyntaxError, RegionError) as exc:
        raise ContractError("Approved regression scaffold must be valid UTF-8 Python") from exc
    modules = {alias.asname or alias.name for node in tree.body if isinstance(node, ast.Import)
               for alias in node.names if alias.name == "unittest"}
    bases = {alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
             and node.module == "unittest" for alias in node.names if alias.name == "TestCase"}
    class_name, method_name = config["identifier"].split(".")
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1 or not any(
            isinstance(base, ast.Name) and base.id in bases or
            isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name)
            and base.value.id in modules and base.attr == "TestCase" for base in classes[0].bases):
        raise ContractError("Approved scaffold must use an explicit unittest.TestCase class")
    methods = [node for node in classes[0].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name.startswith("test_")]
    if (len(methods) != 1 or methods[0].name != method_name or not isinstance(methods[0], ast.FunctionDef)
            or len(methods[0].body) != 1 or not isinstance(methods[0].body[0], ast.Pass)):
        raise ContractError("Approved scaffold must contain one synchronous placeholder test method with only pass")
    return {**config, "destination": destinations[0], "source": source,
            "content": source.decode("utf-8"), "region": region}


def apply_regression_scaffold(scaffold: dict, replacement: str) -> tuple[bytes, dict]:
    """Return a whole new test document while preserving its protected prefix."""
    source = scaffold["source"]
    if hashlib.sha256(source).hexdigest() != scaffold["sha256"]:
        raise ContractError("Regression scaffold snapshot hash is inconsistent")
    try:
        candidate, receipt = replace_python_region(source, scaffold["region"], replacement)
    except RegionError as exc:
        diagnostics = dict(getattr(exc, "diagnostics", None) or {})
        diagnostics["error_code"] = exc.code
        raise ContractError(str(exc), diagnostics=diagnostics) from exc
    return candidate, {**receipt, "template_path": scaffold["path"],
        "template_sha256": scaffold["sha256"], "destination": scaffold["destination"],
        "template_unchanged": True}
