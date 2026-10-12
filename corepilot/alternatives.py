"""Frozen repair bases and bounded isolated copies; loading never executes work."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from .workspace import BoundaryError, snapshot, source_manifest


def manifest_fingerprint(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def write_receipt(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise BoundaryError('Alternative receipt cannot be a symbolic link')
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as stream:
        temporary = Path(stream.name)
        try:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


class AlternativeStore:
    def __init__(self, runtime: Path, task_id: str):
        if len(task_id) != 32 or any(char not in '0123456789abcdef' for char in task_id):
            raise BoundaryError('Invalid alternative task identity')
        self.runtime = Path(runtime).resolve()
        self.directory = self.runtime / 'tasks' / task_id / 'alternatives'
        self.base = self.directory / 'base'
        self.receipt = self.directory / 'receipt.json'
        for part in (self.directory.parent, self.directory, self.base):
            if part.is_symlink():
                raise BoundaryError('Alternative storage cannot use symbolic links')

    def load(self) -> dict:
        """Read saved evidence only: no commands, model calls or file rewrites."""
        if self.receipt.is_symlink():
            raise BoundaryError('Alternative receipt cannot be a symbolic link')
        value = json.loads(self.receipt.read_text())
        if (value.get('version') != 1 or not isinstance(value.get('attempts'), list)
                or len(value['attempts']) > 2 or not isinstance(value.get('base_manifest'), dict)):
            raise BoundaryError('Invalid alternative receipt')
        if source_manifest(self.base) != value['base_manifest']:
            raise BoundaryError('Frozen alternative base changed')
        for row in value['attempts']:
            if not isinstance(row, dict) or not isinstance(row.get('workspace_id'), str):
                raise BoundaryError('Invalid alternative workspace')
            path = self.runtime / 'workspaces' / row['workspace_id']
            if (len(row['workspace_id']) != 32
                    or any(char not in '0123456789abcdef' for char in row['workspace_id']) or path.is_symlink()):
                raise BoundaryError('Invalid alternative workspace')
        return value

    def freeze(self, root: Path, checkpoint: dict) -> dict:
        if self.receipt.exists():
            return self.load()
        if self.base.exists():
            raise BoundaryError('Incomplete alternative base requires inspection')
        expected = source_manifest(root)
        copied = snapshot(root, self.base)
        if source_manifest(root) != expected or copied != expected or source_manifest(self.base) != expected:
            raise BoundaryError('Source changed while freezing alternative base')
        value = {'version': 1, 'base_manifest': expected,
                 'base_source_fingerprint': manifest_fingerprint(expected),
                 'checkpoint': checkpoint, 'attempts': []}
        write_receipt(self.receipt, value)
        return value

    def create(self, *, component: str, checks: dict, maximum: int = 2) -> dict:
        if component not in {'implementation', 'tests'} or type(maximum) is not int or not 0 <= maximum <= 2:
            raise BoundaryError('Alternative component or budget is invalid')
        value = self.load()
        if len(value['attempts']) >= maximum:
            raise BoundaryError('Sequential alternative budget reached')
        number = len(value['attempts']) + 1
        workspace_id = uuid.uuid4().hex
        target = self.runtime / 'workspaces' / workspace_id
        row = {'id': number, 'workspace_id': workspace_id, 'component': component,
               'base_source_fingerprint': value['base_source_fingerprint'],
               'checks': checks, 'status': 'preparing', 'rejection_reason_codes': []}
        value['attempts'].append(row)
        write_receipt(self.receipt, value)  # Persist intent before copying, never silently replay.
        copied = snapshot(self.base, target)
        if copied != value['base_manifest'] or source_manifest(target) != copied:
            raise BoundaryError('Alternative does not match its frozen base')
        row['status'] = 'running'
        write_receipt(self.receipt, value)
        return {**row, 'workspace': str(target)}

    def finalize(self, number: int, *, quality: dict, accepted: bool, reasons: list[str]) -> dict:
        value = self.load()
        if type(number) is not int or not 1 <= number <= len(value['attempts']):
            raise BoundaryError('Unknown alternative')
        row = value['attempts'][number - 1]
        if row['status'] not in {'running', 'verified_pending_review', 'rejected', 'accepted'}:
            raise BoundaryError('Incomplete alternative copy cannot be finalized')
        fingerprint = manifest_fingerprint(source_manifest(self.runtime / 'workspaces' / row['workspace_id']))
        if accepted and (quality.get('passed') is not True or reasons or quality.get('source_fingerprint') != fingerprint):
            raise BoundaryError('Accepted alternative needs current passing quality evidence')
        status = 'accepted' if accepted else 'verified_pending_review' if quality.get('passed') is True and not reasons else 'rejected'
        row.update(status=status, quality=quality,
                   rejection_reason_codes=list(reasons), accepted=accepted,
                   candidate_source_fingerprint=fingerprint)
        write_receipt(self.receipt, value)
        return row
