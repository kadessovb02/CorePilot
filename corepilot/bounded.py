"""Function-scoped execution with addressed repair and current quality gates."""
from __future__ import annotations

import ast
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time

from .alternatives import AlternativeStore, manifest_fingerprint
from .check_runner import regression_argv, structured_unittest
from .controlled import Controller, ControlledBlocked, _host_observations, _requests
from .controlled_contract import ContractError, explicit_edit_scope, prepare_tests, relative_path, validate_selection
from .edit_regions import select_python_region, replace_python_region, validate_python_style, RegionError
from .execution_quality import validate_differential_regression
from .regression_scaffold import apply_regression_scaffold, select_regression_scaffold
from .scope import explicit_function_scope, superseded_function_scope
from .tools import WorkspaceTools
from .workspace import BoundaryError, safe_path, snapshot, source_manifest


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _functions(tree, prefix=''):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            identifier = prefix + node.name
            yield identifier
            yield from _functions(node, identifier + '.')
        elif isinstance(node, ast.ClassDef):
            yield from _functions(node, prefix + node.name + '.')


def select_target(root: Path, request: str, profile: dict) -> dict:
    """Only an explicit unambiguous Python symbol may authorize a region edit."""
    manifest = source_manifest(root)
    named = set(re.findall(r'(?<![\w/])(?:[\w.-]+/)*[\w.-]+\.py\b', request))
    allowed = set(profile.get('allowed_edit_paths') or manifest)
    files = sorted(path for path in manifest if path in allowed and path.endswith('.py')
                   and not any(path == item or path.startswith(item + '/') for item in profile['test_roots']))
    # Use the same already-admitted literal scope as file permissions. A
    # preserved helper's incidental name must not authorize a second target.
    try:
        requested_paths = explicit_edit_scope(request)
        explicit = explicit_function_scope(request)
        superseded = superseded_function_scope(request)
    except ContractError as exc:
        raise ControlledBlocked('Explicit source scope rejected: ' + str(exc)) from exc
    if requested_paths is not None:
        files = [path for path in files if path in requested_paths]
    if superseded:
        raise ControlledBlocked('Earlier function-body authorization was superseded; state a fresh explicit function-body scope')
    if explicit is not None:
        path = relative_path(explicit['source_path'])
        if path not in files:
            raise ControlledBlocked('Explicit function target is unavailable or outside configured editable source paths')
        source = safe_path(root, path, False).read_bytes()
        if len(source) > 64_000:
            raise ControlledBlocked('Explicit function target source exceeds the region size limit')
        try:
            return {**select_python_region(source, explicit['identifier']), 'path': path}
        except RegionError as exc:
            raise ControlledBlocked('Explicit function target does not identify one current source symbol: ' + str(exc)) from exc
    if named & set(files):
        files = sorted(named & set(files))
    if len(files) > 80:
        raise ControlledBlocked('Name the target file; region lookup is limited to 80 source files')
    candidates = []
    for path in files:
        source = safe_path(root, path, False).read_bytes()
        if len(source) > 64_000:
            continue
        try:
            tree = ast.parse(source.decode('utf-8-sig'))
        except (SyntaxError, UnicodeError, ValueError, RecursionError):
            continue
        for identifier in _functions(tree):
            if re.search(r'(?<![\w.])' + re.escape(identifier) + r'(?![\w.])', request):
                candidates.append((path, identifier, source))
    if len(candidates) != 1:
        raise ControlledBlocked('Name exactly one editable Python function and its file; target is missing or ambiguous')
    path, identifier, source = candidates[0]
    try:
        region = select_python_region(source, identifier)
    except RegionError as exc:
        raise ControlledBlocked(str(exc)) from exc
    return {**region, 'path': path}


class BoundedController(Controller):
    def __init__(self, engine, task, tools):
        super().__init__(engine, task, tools)
        self.execution = self.state.setdefault('execution', {
            'version': 1, 'alternatives': [], 'rejections': [], 'attempt_signatures': [],
            'next_action': 'initial', 'elapsed_seconds': 0})
        self.alternative_store = AlternativeStore(engine.settings.runtime, task['id'])

    def rejected(self, stage, message, error):
        raw = message.get('content') or json.dumps(message.get('tool_calls', []), sort_keys=True)
        signature = _sha(json.dumps({'stage': stage, 'source': self.state.get('read_manifest'),
            'response': _sha(raw.encode('utf-8'))}, sort_keys=True).encode())
        previous = self.execution.setdefault('rejected_submissions', [])
        repeated = any(row['fingerprint'] == signature for row in previous)
        diagnostics = error.diagnostics if isinstance(error.diagnostics, dict) else {}
        truncated = message.get('_corepilot_finish_reason') == 'length'
        code = ('model_output_token_budget_reached' if truncated else
                diagnostics.get('error_code', 'stage_format_rejected'))
        row = {'fingerprint': signature, 'stage': stage, 'category': 'budget' if truncated else 'format', 'reason_code': code,
               'repeated': repeated, 'step': self.task['step']}
        previous.append(row)
        self.event('repair_decision', 'Rejected stage proposal' + (' repeated unchanged' if repeated else ''), row)
        if repeated:
            raise ControlledBlocked('Identical invalid stage response repeated under unchanged source; checkpoint retained')

    async def ask(self, stage, context, **kwargs):
        try:
            return await super().ask(stage, context, **kwargs)
        except ControlledBlocked as exc:
            reason = str(exc)
            if reason.startswith(('Controlled model-call budget reached', 'Controlled context exceeds budget')):
                self.rejection('budget', ['model_call_budget_reached' if self.task['step'] >= self.task['max_steps'] else 'context_budget_reached'])
            raise

    def context(self, **extra):
        value = super().context(**extra)
        value['bounded_regression_contract'] = True
        value['repair_component'] = self.execution.get('next_action', 'initial')
        quality = self.execution.get('quality', {})
        if quality and not quality.get('passed'):
            value['quality_failure_codes'] = self.failures(quality)
            regression = quality.get('regression', {})
            value['regression_failure_untrusted'] = {key: regression.get(key)
                for key in ('passed', 'reason_codes', 'semantic_scope')}
            for arm in ('candidate', 'baseline'):
                check = regression.get(arm, {})
                value['regression_failure_untrusted'][arm] = {key: check[key] for key in
                    ('exit_code', 'tests_run', 'failure_count', 'error_count', 'failures', 'errors') if key in check}
        return value

    async def refresh_context(self, observed_checks=None):
        from .project_context import ensure_project_context, project_context_view
        manifest = source_manifest(self.root)
        package = await asyncio.to_thread(ensure_project_context, self.root,
            self.engine.settings.runtime / 'project-context' / (self.task['id'] + '.json'),
            manifest=manifest, observed_checks=observed_checks if observed_checks is not None else {
                key: self.profile.get(key, []) for key in ('checks', 'behavior_checks', 'compatibility_checks')})
        self.fresh(manifest, 'Source changed while preparing versioned context')
        self.execution['project_context'] = project_context_view(package['package'],
            '\n\n'.join(_requests(self.task)), max_chars=4500)
        self.event('navigation', 'Prepared versioned region context', {
            'source_fingerprint': manifest_fingerprint(manifest),
            'model_calls': 0, 'refreshed': package['refreshed']})

    def model_observations(self):
        # Raw full reads remain paired in the checkpoint. A marked bounded view
        # avoids regenerating the whole file and repeating all test output.
        observations = _host_observations(self.evidence)
        target = self.execution.get('target')
        result = []
        for row in observations:
            value = row['result']
            path = value.get('path')
            if target and path == target['path']:
                result.append({'operation': 'read_file', 'arguments': row['arguments'],
                    'result': {'path': path, 'view': 'selected_function_only',
                               'file_sha256': self.state['read_manifest'].get(path),
                               'identifier': target['identifier'], 'content': target['content']}})
            elif isinstance(value.get('content'), str):
                text = value['content']
                maximum = 2400 if path in self.state.get('authored_tests', {}) else 1800
                result.append({**row, 'result': {**value, 'content': text[:maximum],
                    'view': 'bounded_excerpt', 'excerpt_truncated': len(text) > maximum}})
        return result[:12]

    def rejection(self, component: str, reasons: list[str], *, quality=None):
        row = {'component': component, 'reason_codes': list(reasons),
               'source_fingerprint': manifest_fingerprint(source_manifest(self.root)),
               'step': self.task['step']}
        if quality is not None:
            row['quality'] = quality
        self.execution['rejections'].append(row)
        self.event('repair_decision', 'Addressed repair: ' + component, row)

    async def selection(self):
        self.phase('select')
        request = '\n\n'.join(_requests(self.task))
        target = select_target(self.root, request, self.profile)
        manifest = source_manifest(self.root)
        guidance = [path for path in ('AGENTS.md', 'CLAUDE.md') if path in manifest]
        named = set(re.findall(r'(?<![\w/])(?:[\w.-]+/)*[\w.-]+\.py\b', request))
        test_paths = [path for path in sorted(named) if path not in manifest
                      and any(path.startswith(root + '/') for root in self.profile['test_roots'])]
        fixed = self.state.get('fixed_new_test_paths')
        if fixed is not None:
            test_paths = list(fixed)
        if not test_paths:
            test_paths = [self.profile['test_roots'][0] + '/test_corepilot_' +
                          re.sub(r'\W+', '_', target['identifier']) + '_' + self.task['id'][:8] + '.py']
        if len(test_paths) != 1:
            raise ControlledBlocked('Region execution needs exactly one fixed regression test file')
        read_paths = list(dict.fromkeys([target['path'], *self.profile['read_paths'],
                                        *sorted(named & set(manifest)), *self.state.get('authored_tests', {})]))
        selection = validate_selection(self.root, {'read_paths': read_paths,
            'edit_paths': [target['path']], 'new_test_paths': test_paths},
            test_roots=self.profile['test_roots'], allowed_edit_paths=self.profile['allowed_edit_paths'],
            read_only=False, request=request, authored_test_paths=tuple(self.state.get('authored_tests', {})),
            authored_sha256=self.state.get('authored_tests', {}), fixed_test_paths=fixed)
        self.state['selection'] = selection
        self.state.setdefault('fixed_new_test_paths', test_paths)
        self.execution['target'] = target
        await self.read([*guidance, *read_paths])
        await self.refresh_context()
        self.fresh(self.state['read_manifest'], 'Source changed while preparing versioned context')

    async def edits(self):
        self.phase('edit')
        await self.read([*self.contents, *self.state.get('authored_tests', {})])
        target = self.execution['target']
        original = safe_path(self.root, target['path'], False).read_bytes()
        region = select_python_region(original, target['identifier'])
        self.execution['target'] = {**region, 'path': target['path']}
        def validate(result):
            self.fresh(self.state['read_manifest'], 'Region source changed before replacement')
            try:
                candidate, receipt = replace_python_region(original, region, result['content'])
            except RegionError as exc:
                diagnostics = deepcopy(getattr(exc, 'diagnostics', None) or {})
                diagnostics['error_code'] = exc.code
                raise ContractError(str(exc), diagnostics=diagnostics) from exc
            # Never retry the same rejected proposal under unchanged evidence.
            signature = _sha(json.dumps({'component': 'implementation', 'source': self.state['start_manifest'],
                'proposal': _sha(candidate)}, sort_keys=True).encode())
            if signature in self.execution['attempt_signatures']:
                raise ContractError('Identical implementation attempt already failed; change the approach')
            changed = candidate != original
            if not changed and self.state['read_manifest'].get(target['path']) == self.state['start_manifest'].get(target['path']):
                raise ContractError('Initial region replacement must make a real change')
            return candidate, receipt, signature
        result, prepared = await self.ask('edit', self.context(
            selected_region={key: value for key, value in region.items() if key not in {'content'}},
            current_region=region['content'], project_context_untrusted=self.execution.get('project_context', {})),
            validate=validate, document_path=target['path'], region=region)
        candidate, receipt, signature = prepared
        self.execution['current_proposal_signature'] = signature
        self.execution.setdefault('component_signatures', {})['implementation'] = signature
        if candidate != original:
            self.fresh(self.state['read_manifest'])
            applied, _ = await self.host('apply_patch', {'path': target['path'],
                'old_text': original.decode('utf-8'), 'new_text': candidate.decode('utf-8')})
            if 'error' in applied:
                raise ControlledBlocked('Host could not apply the hash-checked function replacement')
            self.task['last_edit_step'] = self.task['tool_operations']
        updated = select_python_region(candidate, target['identifier'])
        self.execution['target'] = {**updated, 'path': target['path']}
        self.execution['last_region_action'] = {'path': target['path'], 'identifier': target['identifier'],
            'context': region['content'], 'replacement': result['content'], 'receipt': receipt,
            'before_sha256': _sha(original), 'after_sha256': _sha(candidate)}
        self.state['summaries']['edit'] = 'Host applied a function-scoped proposal.'
        self.event('region_edit', 'Validated function replacement', self.execution['last_region_action'])
        await self.refresh_context()

    async def tests(self):
        if self.profile.get('regression_scaffold') is None:
            await super().tests()
        else:
            await self.scaffold_tests()
        path = self.state['selection']['new_test_paths'][0]
        signature = _sha(json.dumps({'component': 'tests', 'source': source_manifest(self.root),
            'test': self.state['authored_tests'][path]}, sort_keys=True).encode())
        if signature in self.execution['attempt_signatures']:
            raise ControlledBlocked('Identical test attempt already failed; checkpoint retained')
        self.execution['current_proposal_signature'] = signature
        self.execution.setdefault('component_signatures', {})['tests'] = signature

    async def scaffold_tests(self):
        self.phase('tests')
        await self.read([*self.contents, *self.state.get('authored_tests', {})])
        scaffold = select_regression_scaffold(self.root, self.profile, self.state['selection'])
        destination = scaffold['destination']
        target = self.execution['target']
        implementation = select_python_region(safe_path(self.root, target['path'], False).read_bytes(),
                                              target['identifier'])
        self.fresh(self.state['read_manifest'], 'Implementation changed while preparing scaffold test context')
        def validate(result):
            self.fresh(self.state['read_manifest'], 'Source became stale before scaffold test application')
            current = select_regression_scaffold(self.root, self.profile, self.state['selection'])
            if current['source'] != scaffold['source']:
                raise ContractError('Approved regression scaffold changed before replacement')
            candidate, receipt = apply_regression_scaffold(scaffold, result['content'])
            prepared = prepare_tests(self.root, [{'path': destination, 'content': candidate.decode('utf-8')}],
                selected_paths=[destination], test_roots=self.profile['test_roots'],
                authored_test_paths=tuple(self.state.get('authored_tests', {})),
                authored_sha256=self.state.get('authored_tests', {}))
            return prepared, receipt
        _, (prepared, receipt) = await self.ask('tests', self.context(
            current_implementation_untrusted={'path': target['path'], 'identifier': target['identifier'],
                'sha256': implementation['file_sha256'], 'region_sha256': implementation['region_sha256'],
                'content': implementation['content']},
            fixed_output_path=destination, regression_scaffold_untrusted={
                'path': scaffold['path'], 'sha256': scaffold['sha256'], 'identifier': scaffold['identifier'],
                'template': scaffold['content'], 'current_region': scaffold['region']['content']}),
            validate=validate, document_path=destination, region=scaffold['region'])
        self.fresh(self.state['read_manifest'], 'Source became stale before scaffold test application')
        content = prepared[destination]
        changed = self.contents.get(destination) != content
        result, _ = await self.host('write_file', {'path': destination, 'content': content})
        if 'error' in result:
            raise ControlledBlocked('Prevalidated scaffold regression test could not be written')
        self.state['authored_tests'][destination] = _sha(content.encode('utf-8'))
        if changed:
            self.task['last_edit_step'] = self.task['tool_operations']
        self.state['summaries']['tests'] = 'Host supplied a new test file from the approved scaffold.'
        self.event('regression_scaffold', 'Validated placeholder-method replacement in a new test file', receipt)
        self.engine.store.save(self.task)

    async def execute_check(self, argv, *, category, tools=None, expected=None):
        selected_tools = tools or self.tools
        expected = expected or source_manifest(selected_tools.root)
        if selected_tools is self.tools:
            result, _ = await self.host('run_command', {'argv': argv, 'timeout': self.engine.settings.command_timeout})
        else:
            previous_tools = self.tools
            try:
                self.tools = selected_tools
                result, _ = await self.host('run_command', {'argv': argv, 'timeout': self.engine.settings.command_timeout})
            finally:
                self.tools = previous_tools
        if source_manifest(selected_tools.root) != expected:
            raise ControlledBlocked('A read-only quality check changed its source')
        record = {'argv': argv, 'category': category, 'exit_code': result.get('exit_code'),
            'timed_out': result.get('timed_out', False), 'output': result.get('output', result.get('error', '')),
            'sandbox': result.get('sandbox'), 'operation': self.task['tool_operations'],
            'source_fingerprint': manifest_fingerprint(expected)}
        self.task.setdefault('checks', []).append(record)
        self.task['last_check_step'] = self.task['tool_operations']
        self.engine.store.save(self.task)
        return record

    async def regression(self, current_manifest):
        path = self.state['selection']['new_test_paths'][0]
        actual = safe_path(self.root, path, False).read_bytes()
        if _sha(actual) != self.state['authored_tests'].get(path):
            return {'passed': False, 'reason_codes': ['authored_test_hash_changed']}
        argv = regression_argv([path])
        check_id = 'authored-regression'
        candidate = structured_unittest(await self.execute_check(argv, category='regression_candidate'))
        candidate.update(check_id=check_id, authored_test_sha256=_sha(actual))
        # One isolated pristine copy per actual authored-test version. The
        # generated test is identical; only the implementation is reverted.
        directory = self.engine.settings.runtime / 'tasks' / self.task['id'] / 'regression' / _sha(actual)
        if not directory.exists():
            copied = await asyncio.to_thread(snapshot, Path(self.task['baseline']), directory)
            if copied != self.state['start_manifest']:
                raise ControlledBlocked('Pristine regression baseline changed')
            test = safe_path(directory, path)
            test.parent.mkdir(parents=True, exist_ok=True)
            test.write_bytes(actual)
        baseline_manifest = source_manifest(directory)
        wanted = {**self.state['start_manifest'], path: _sha(actual)}
        if baseline_manifest != wanted:
            raise ControlledBlocked('Pristine regression copy does not match original source and identical test')
        baseline_tools = WorkspaceTools(directory, self.engine.settings.root,
            command_timeout=self.engine.settings.command_timeout,
            source_root=Path(self.task['source_path']), read_only_commands=True)
        try:
            baseline = structured_unittest(await self.execute_check(argv, category='regression_baseline',
                tools=baseline_tools, expected=baseline_manifest))
        finally:
            await baseline_tools.cancel()
        baseline.update(check_id=check_id, authored_test_sha256=_sha(actual))
        gate = validate_differential_regression(baseline, candidate,
            expected_source_sha256=manifest_fingerprint(current_manifest),
            expected_test_sha256=_sha(actual), expected_baseline_source_sha256=manifest_fingerprint(baseline_manifest),
            test_source=actual.decode('utf-8'), target_identifier=self.execution['target']['identifier'],
            expectations=self.profile.get('expectations'))
        return {**gate, 'baseline': baseline, 'candidate': candidate,
                'authored_test_sha256': _sha(actual)}

    async def quality(self):
        self.phase('check')
        expected = source_manifest(self.root)
        self.scope(expected)
        selected = self.state['selection']['edit_paths'][0]
        base_bytes = safe_path(Path(self.task['baseline']), selected, False).read_bytes()
        now_bytes = safe_path(self.root, selected, False).read_bytes()
        try:
            base_region = select_python_region(base_bytes, self.execution['target']['identifier'])
            now_region = select_python_region(now_bytes, self.execution['target']['identifier'])
            recreated, _ = replace_python_region(base_bytes, base_region, now_region['content'])
            region_ok = recreated == now_bytes
        except RegionError:
            region_ok = False
        scope = {'passed': region_ok, 'reason_codes': [] if region_ok else ['region_or_public_interface_changed']}
        style = validate_python_style(base_bytes, now_bytes)
        source_style = deepcopy(style)
        for path in self.state.get('authored_tests', {}):
            test_style = validate_python_style(b'', safe_path(self.root, path, False).read_bytes())
            if not test_style['passed']:
                style = {**test_style, 'component': 'tests', 'source_passed': source_style['passed']}
        quality = {'scope': scope, 'style': style, 'source_style': source_style}
        records = []
        for category, key in (('behavior', 'behavior_checks'), ('compatibility', 'compatibility_checks'), ('integration', 'checks')):
            commands = self.profile.get(key, [])
            checks = [await self.execute_check(argv, category=category, expected=expected) for argv in commands]
            records.extend(checks)
            quality[category] = {'passed': bool(checks) and all(check['exit_code'] == 0 and not check['timed_out'] for check in checks),
                'checks': checks, 'reason_codes': [] if checks and all(check['exit_code'] == 0 and not check['timed_out'] for check in checks)
                else [category + '_check_failed' if checks else category + '_checks_not_configured']}
        quality['regression'] = await self.regression(expected)
        self.fresh(expected, 'Source changed during quality admission')
        quality['passed'] = all(quality[key]['passed'] for key in ('behavior', 'compatibility', 'integration', 'scope', 'style', 'regression'))
        quality['source_fingerprint'] = manifest_fingerprint(expected)
        self.state['checks'] = records
        self.state['verified_manifest'] = expected
        self.execution['quality'] = quality
        self.event('quality', 'Observed behavior, compatibility, style, scope and regression gates', quality)
        await self.refresh_context(records)
        return quality

    @staticmethod
    def repair_component(quality):
        if not quality['scope']['passed'] or not quality['behavior']['passed'] or not quality['compatibility']['passed']:
            return 'implementation'
        if not quality['style']['passed']:
            return quality['style'].get('component', 'implementation')
        if not quality['integration']['passed'] and quality['regression']['passed']:
            return 'environment'
        return 'tests'

    def failures(self, quality):
        return [code for key in ('behavior', 'compatibility', 'integration', 'scope', 'style', 'regression')
                for code in quality[key].get('reason_codes', []) if not quality[key]['passed']]

    async def alternative(self, component, quality):
        reasons = self.failures(quality)
        self.rejection(component, reasons, quality=quality)
        signature = self.execution.get('component_signatures', {}).get(component)
        if signature:
            self.execution['attempt_signatures'].append(signature)
        checkpoint = {'selection': deepcopy(self.state['selection']),
            'authored_tests': deepcopy(self.state['authored_tests']), 'target': deepcopy(self.execution['target'])}
        frozen = await asyncio.to_thread(self.alternative_store.freeze, self.root, checkpoint)
        if len(frozen['attempts']) >= self.engine.settings.controlled['max_repairs']:
            raise ControlledBlocked('Two-alternative/shared repair budget exhausted; all failures retained')
        if self.task['step'] >= self.task['max_steps']:
            self.rejection('budget', ['model_call_budget_reached'])
            raise ControlledBlocked('Shared model-call budget reached; no alternative was started')
        checks = {key: self.profile.get(key, []) for key in ('behavior_checks', 'compatibility_checks')}
        row = await asyncio.to_thread(self.alternative_store.create, component=component, checks=checks,
                                      maximum=self.engine.settings.controlled['max_repairs'])
        base_state = frozen['checkpoint']
        self.state['selection'] = deepcopy(base_state['selection'])
        self.state['authored_tests'] = deepcopy(base_state['authored_tests'])
        self.execution['target'] = deepcopy(base_state['target'])
        self.state['round'] = row['id']
        self.execution['alternatives'].append(row)
        self.task['workspace'] = row['workspace']
        self.root = Path(row['workspace'])
        self.tools = WorkspaceTools(self.root, self.engine.settings.root,
            command_timeout=self.engine.settings.command_timeout,
            source_root=Path(self.task['source_path']), read_only_commands=True)
        self.engine.tools[self.task['id']] = self.tools
        # Every alternative starts as an exact copy of the same frozen base.
        # Proven components are then carried forward explicitly, with their
        # evidence recorded; a failed repair can never overwrite them.
        preserved = self.execution.get('preserved_components', {})
        for path, value in preserved.items():
            if value['component'] == component:
                continue
            content = value['content'].encode('utf-8')
            if _sha(content) != value['sha256']:
                raise ControlledBlocked('Preserved component content does not match its verified hash')
            target = safe_path(self.root, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            if _sha(target.read_bytes()) != value['sha256']:
                raise ControlledBlocked('Preserved component changed during transfer')
            if value['component'] == 'tests':
                self.state['authored_tests'][path] = value['sha256']
        row['preserved_components'] = {path: {key: value[key] for key in ('component', 'sha256', 'verified_source_fingerprint')}
                                        for path, value in preserved.items() if value['component'] != component}
        self.execution['target'] = {**select_python_region(
            safe_path(self.root, self.execution['target']['path'], False).read_bytes(),
            self.execution['target']['identifier']), 'path': self.execution['target']['path']}
        self.contents = {}
        await self.read([*self.state['selection']['read_paths'], *self.state['authored_tests']])
        self.state['feedback'].append({'checks': [check for key in ('behavior', 'compatibility')
            for check in quality[key]['checks']], 'component': component, 'reason_codes': reasons,
            'regression': {key: value for key, value in quality['regression'].items() if key not in {'baseline', 'candidate'}}})
        self.execution['next_action'] = component
        self.event('alternative', 'Started isolated alternative from the same frozen state', {
            'id': row['id'], 'component': component, 'base_source_fingerprint': row['base_source_fingerprint'],
            'shared_model_calls_used': self.task['step'], 'shared_max_calls': self.task['max_steps']})

    async def run(self):
        started = time.monotonic()
        self.pair_pending()
        self.state.setdefault('start_manifest', source_manifest(Path(self.task['baseline'])))
        self.state.setdefault('authored_tests', {})
        if (not self.profile.get('checks') or not self.profile.get('behavior_checks') or not self.profile.get('compatibility_checks')
                or self.engine.settings.controlled.get('artifact_protocol') != 'documents'):
            raise ControlledBlocked('Region execution requires documents plus operator-owned behavior and compatibility checks')
        if not getattr(self.tools, 'read_only_commands', False):
            raise ControlledBlocked('Region execution requires read-only sandboxed checks')
        check_contract = {key: deepcopy(self.profile.get(key, [])) for key in
                          ('checks', 'behavior_checks', 'compatibility_checks', 'test_roots', 'allowed_edit_paths', 'read_paths', 'expectations')}
        if 'regression_scaffold' in self.profile:
            check_contract['regression_scaffold'] = deepcopy(self.profile['regression_scaffold'])
        budget_contract = {'max_calls': self.task['max_steps'], 'max_tokens': self.engine.settings.max_tokens,
            'context_chars': self.engine.settings.context_chars,
            'task_seconds': self.engine.settings.resource_guard['max_task_seconds'],
            'max_alternatives': self.engine.settings.controlled['max_repairs']}
        if (self.execution.setdefault('check_contract', check_contract) != check_contract
                or self.execution.setdefault('budget_contract', budget_contract) != budget_contract):
            raise ControlledBlocked('Saved bounded checks or budgets changed; start a new task instead of replaying')
        try:
            await self.selection()
            action = self.execution.get('next_action', 'initial')
            while True:
                if action in {'initial', 'implementation'}:
                    await self.edits()
                    if action == 'initial':
                        self.execution['next_action'] = 'tests'
                        self.engine.store.save(self.task)
                if action in {'initial', 'tests'}:
                    await self.tests()
                if action == 'review':
                    self.fresh(self.state['verified_manifest'], 'Saved review admission is stale')
                    quality = self.execution['quality']
                else:
                    self.execution['next_action'] = 'quality'
                    self.engine.store.save(self.task)
                    quality = await self.quality()
                if all(quality[key]['passed'] for key in ('behavior', 'compatibility', 'scope', 'source_style')):
                    path = self.execution['target']['path']
                    content = safe_path(self.root, path, False).read_bytes()
                    self.execution.setdefault('preserved_components', {})[path] = {
                        'component': 'implementation', 'content': content.decode('utf-8'), 'sha256': _sha(content),
                        'verified_source_fingerprint': quality['source_fingerprint']}
                if quality['regression']['passed']:
                    for path, digest in self.state['authored_tests'].items():
                        self.execution.setdefault('preserved_components', {})[path] = {
                            'component': 'tests', 'content': safe_path(self.root, path, False).read_bytes().decode('utf-8'),
                            'sha256': digest, 'verified_source_fingerprint': quality['source_fingerprint']}
                current_alternative = self.execution['alternatives'][-1] if self.execution['alternatives'] else None
                if current_alternative:
                    saved = self.alternative_store.finalize(current_alternative['id'], quality=quality,
                        accepted=False, reasons=self.failures(quality))
                    current_alternative.update(saved)
                if not quality['passed']:
                    component = self.repair_component(quality)
                    if component == 'environment':
                        self.rejection(component, ['integration_cause_unproven'], quality=quality)
                        raise ControlledBlocked('Integration check failed outside the identified component; operator diagnosis required')
                    await self.alternative(component, quality)
                    action = component
                    continue
                self.phase('review')
                self.execution['next_action'] = 'review'
                self.engine.store.save(self.task)
                review = await self.ask('review', self.context(
                    diff=self.engine.diff(self.task['id'])['diff'], checks=self.state['checks'],
                    quality={key: {'passed': quality[key]['passed']} for key in ('behavior', 'compatibility', 'style', 'scope', 'regression')}))
                self.fresh(self.state['verified_manifest'], 'Source changed after the full quality check')
                self.state['review'] = review
                self.event('review', review['summary'], review)
                if not review['approved']:
                    self.rejection('implementation', ['review_rejected'])
                    if current_alternative:
                        current_alternative.update(self.alternative_store.finalize(current_alternative['id'],
                            quality=quality, accepted=False, reasons=['review_rejected']))
                    rejected = deepcopy(quality)
                    rejected['behavior'] = {'passed': False, 'checks': [], 'reason_codes': ['review_rejected']}
                    await self.alternative('implementation', rejected)
                    action = 'implementation'
                    continue
                if current_alternative:
                    current_alternative.update(self.alternative_store.finalize(current_alternative['id'],
                        quality=quality, accepted=True, reasons=[]))
                self.execution['next_action'] = 'completed'
                self.task['answer'] = ('Function-scoped change passed configured behavior and compatibility checks, '
                    'style/scope gates and the same regression test on original and corrected source.\n\n' + review['summary'])
                self.phase('completed')
                self.task.update(status='completed', error=None)
                self.event('status', 'Completed after all current quality gates and review', {'status': 'completed'})
                self.event('diff', 'Workspace changes available', self.engine.diff(self.task['id']))
                break
        except (ControlledBlocked, ContractError, RegionError, BoundaryError):
            self.execution.setdefault('rejections', []).append({'component': 'budget' if self.task['step'] >= self.task['max_steps'] else 'format',
                'reason_codes': ['interrupted_before_full_admission'], 'step': self.task['step']})
            raise
        finally:
            self.execution['elapsed_seconds'] += time.monotonic() - started
            self.engine.store.save(self.task)
