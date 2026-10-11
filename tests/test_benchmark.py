"""Comparison control and oracle isolation with synthetic data, never inference."""
import asyncio
import json
import runpy
import shutil
import sys
from pathlib import Path

import httpx
import pytest

from corepilot import benchmark
from corepilot.benchmark_resources import ResourceThresholds, public_sample, summarize
from corepilot.config import Settings
from corepilot.workspace import source_manifest


def safe(**changes):
    return public_sample({"supported": True, "platform": "darwin", "pressure_level": 1,
                          "available_estimate_gib": 4.0, "used_swap_gib": 2.0,
                          "compressor_gib": 1.0, "rss_complete": True, "own_rss_gib": 0.1, **changes})


class Monitor:
    thresholds = ResourceThresholds()

    def __init__(self, samples=None, *, abort=False):
        self.samples = list(samples or [safe()])
        self.sample_count = 0
        self.abort = abort
        self.stopped = False

    async def sample(self):
        self.sample_count += 1
        return self.samples.pop(0) if len(self.samples) > 1 else self.samples[0]

    async def watch(self, stop_event, *, baseline, on_unsafe, interval):
        if self.abort:
            await on_unsafe(["memory_pressure_warning"], safe(pressure_level=2))
        await stop_event.wait()
        self.stopped = True
        return {"status": "unsafe" if self.abort else "stopped",
                "reasons": ["memory_pressure_warning"] if self.abort else [],
                "samples": [safe()], "summary": summarize([safe()], baseline)}


@pytest.fixture
def suite(tmp_path):
    folder = tmp_path / "synthetic-suite"
    repo = folder / "repository"
    repo.mkdir(parents=True)
    (repo / "module.py").write_text("value = 1\n")
    (folder / "oracle.py").write_text("# Synthetic oracle marker; tests mock execution.\n")
    cases = [{"id": "edit_alpha", "prompt": "Fix the synthetic value", "read_only": False,
              "required_reads": ["module.py"], "oracle": "alpha", "category": "edit", "max_calls": 12, "timeout_seconds": 90},
             {"id": "inspect_beta", "prompt": "Explain module.py without editing", "read_only": True,
              "required_reads": ["module.py"], "oracle": "beta", "category": "inspection", "max_calls": 20, "timeout_seconds": 130}]
    (folder / "cases.json").write_text(json.dumps(cases))
    return folder


def result(case_id, mode, *, passed=True, seconds=1, **overrides):
    return {"id": case_id, "mode": mode, "seconds": seconds, "completed": passed,
            "task_status": "completed" if passed else "interrupted", "oracle_passed": passed,
            "usable_passed": passed, "http_attempts": 2, "timeout": False, "safety_abort": False,
            "safety_reasons": [], **overrides}


@pytest.fixture
def no_rest(monkeypatch):
    waits = []
    async def sleep(seconds):
        waits.append(seconds)
    monkeypatch.setattr(benchmark.asyncio, "sleep", sleep)
    return waits


def test_blocked_admission_never_constructs_engine_or_calls_inference(tmp_path, suite, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Blocked resource admission started model/engine execution")
    monkeypatch.setattr(benchmark, "AgentEngine", forbidden)
    monkeypatch.setattr(benchmark, "MeteredModel", forbidden)
    monkeypatch.setattr(benchmark, "run_case", forbidden)
    monitor = Monitor([safe(pressure_level=2)])
    output = tmp_path / "report.json"
    report = asyncio.run(benchmark.compare(Settings(root=tmp_path / "state"), suite, output, monitor=monitor))
    assert report["records"] == [] and report["stop_reasons"] == ["memory_pressure_warning"]
    assert report["unattempted_runs"] == 4
    assert report["summary"]["paired"]["median_seconds_both_usable"] == {"single": None, "specialists": None}
    assert report == json.loads(output.read_text())


def test_ab_ba_pairs_receive_identical_case_budgets_and_original_source_is_unchanged(tmp_path, suite, monkeypatch, no_rest):
    calls = []
    initial = source_manifest(suite / "repository")
    async def run(settings, selected, case, mode, monitor, baseline, **caps):
        calls.append((case["id"], mode, caps))
        assert selected == suite and baseline["pressure_level"] == 1
        return result(case["id"], mode)
    monkeypatch.setattr(benchmark, "run_case", run)
    report = asyncio.run(benchmark.compare(Settings(root=tmp_path / "state"), suite, tmp_path / "report.json",
                                          monitor=Monitor(), rest_seconds=10, request_pause_seconds=0))
    assert [(case, mode) for case, mode, _ in calls] == [("edit_alpha", "single"), ("edit_alpha", "specialists"),
                                                       ("inspect_beta", "specialists"), ("inspect_beta", "single")]
    assert calls[0][2] == calls[1][2] == {"timeout": 90, "max_calls": 12, "max_tokens": 768, "context_chars": 14000, "request_pause_seconds": 0}
    assert calls[2][2] == calls[3][2] == {"timeout": 130, "max_calls": 20, "max_tokens": 768, "context_chars": 14000, "request_pause_seconds": 0}
    assert report["settings"]["between_requests_seconds"] == 0
    assert report["summary"]["paired"]["both_usable"] == 2
    assert not report["stop_reasons"] and report["unattempted_runs"] == 0
    assert source_manifest(suite / "repository") == initial
    assert no_rest and all(delay == 10 for delay in no_rest)


@pytest.mark.parametrize("stop", ["timeout", "safety_abort"])
def test_timeout_or_resource_abort_prevents_all_future_model_calls(tmp_path, suite, monkeypatch, no_rest, stop):
    calls = []
    async def run(settings, selected, case, mode, monitor, baseline, **caps):
        calls.append((case["id"], mode))
        return result(case["id"], mode, passed=False, **{stop: True},
                      safety_reasons=["memory_pressure_warning"] if stop == "safety_abort" else [])
    monkeypatch.setattr(benchmark, "run_case", run)
    report = asyncio.run(benchmark.compare(Settings(root=tmp_path / "state"), suite, tmp_path / "report.json", monitor=Monitor()))
    assert calls == [("edit_alpha", "single")]
    assert len(report["records"]) == 1 and report["unattempted_runs"] == 3
    assert report["stop_reasons"] == (["memory_pressure_warning"] if stop == "safety_abort" else ["case_timeout_backend_idle_unconfirmed"])
    assert no_rest == []


def test_new_pressure_warning_between_paired_runs_blocks_the_second_mode(tmp_path, suite, monkeypatch, no_rest):
    calls = []
    async def run(settings, selected, case, mode, monitor, baseline, **caps):
        calls.append(mode)
        return result(case["id"], mode)
    monkeypatch.setattr(benchmark, "run_case", run)
    monitor = Monitor([safe(), safe(), safe(pressure_level=4)])
    report = asyncio.run(benchmark.compare(Settings(root=tmp_path / "state"), suite, tmp_path / "report.json", monitor=monitor))
    assert calls == ["single"] and report["stop_reasons"] == ["memory_pressure_critical"]
    assert report["unattempted_runs"] == 3


def test_speed_comparison_uses_only_pairs_with_both_usable_results():
    rows = [result("failed_fast", "single", passed=False, seconds=0.1), result("failed_fast", "specialists", seconds=50),
            result("ok", "single", seconds=20), result("ok", "specialists", seconds=30),
            result("both_failed", "single", passed=False, seconds=0.1), result("both_failed", "specialists", passed=False, seconds=0.1)]
    summary = benchmark.aggregate(rows)
    assert summary["single"]["median_seconds_all"] == 0.1
    assert summary["paired"]["median_seconds_both_usable"] == {"single": 20, "specialists": 30}
    assert summary["paired"]["both_usable"] == 1
    assert summary["paired"]["specialist_wins"] == 1 and summary["paired"]["ties"] == 2
    without_success = benchmark.aggregate(rows[:2])
    assert without_success["paired"]["median_seconds_both_usable"] == {"single": None, "specialists": None}


def test_aggregate_separates_edits_and_inspections_using_explicit_read_only_flag():
    rows = [result("edit", "single", passed=False, oracle_passed=True, read_only=False, category="inspection"),
            result("inspect", "single", read_only=True, category="edit"),
            result("legacy", "single"),
            result("edit", "specialists", read_only=False),
            result("inspect", "specialists", passed=False, read_only=True)]
    summary = benchmark.aggregate(rows)
    assert summary["single"]["by_kind"] == {
        "edits": {"attempted": 1, "oracle_passed": 1, "usable_passed": 0},
        "inspections": {"attempted": 1, "oracle_passed": 1, "usable_passed": 1}}
    assert summary["specialists"]["by_kind"] == {
        "edits": {"attempted": 1, "oracle_passed": 1, "usable_passed": 1},
        "inspections": {"attempted": 1, "oracle_passed": 0, "usable_passed": 0}}
    assert summary["single"]["attempted"] == 3


@pytest.mark.parametrize("reads", [["/outside.py"], ["../outside.py"], ["nested/../../outside.py"],
                                  [""], ["module.py", 4], [], "module.py"])
def test_case_required_reads_reject_unsafe_or_malformed_paths(suite, reads):
    rows = json.loads((suite / "cases.json").read_text())
    rows[0]["required_reads"] = reads
    (suite / "cases.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="required read paths"):
        benchmark.load_cases(suite)


@pytest.mark.parametrize("key,value", [("oracle", False), ("oracle", {}), ("oracle", " "),
                                      ("category", []), ("category", 3), ("category", "")])
def test_case_oracle_and_category_require_nonempty_strings(suite, key, value):
    rows = json.loads((suite / "cases.json").read_text())
    rows[0][key] = value
    (suite / "cases.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="oracle or category"):
        benchmark.load_cases(suite)


@pytest.mark.parametrize("key", ["max_calls", "timeout_seconds"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "12", None])
def test_case_execution_budgets_reject_boolean_noninteger_and_nonpositive_values(suite, key, value):
    rows = json.loads((suite / "cases.json").read_text())
    rows[0][key] = value
    (suite / "cases.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="execution budget"):
        benchmark.load_cases(suite)


def test_case_optional_budgets_and_oracle_identifier_remain_backward_compatible(suite):
    rows = json.loads((suite / "cases.json").read_text())
    del rows[0]["max_calls"]
    del rows[0]["timeout_seconds"]
    rows[0]["required_reads"] = ["nested/module.py"]
    (suite / "cases.json").write_text(json.dumps(rows))
    loaded = benchmark.load_cases(suite)
    assert loaded == rows
    assert loaded[0]["oracle"] != loaded[0]["id"]


def benchmark_cli(monkeypatch, arguments):
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py"))
    main = namespace["main"]
    calls = []
    async def compare(settings, suite, output, **options):
        calls.append((settings, suite, output, options))
        return {"summary": {}, "stop_reasons": []}
    monkeypatch.setitem(main.__globals__, "compare", compare)
    monkeypatch.setitem(main.__globals__, "ResourceMonitor", lambda **kwargs: "synthetic-monitor")
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", *map(str, arguments)])
    return main, calls


@pytest.mark.parametrize("indirect", [False, True])
def test_cli_rejects_raw_state_inside_checkout_even_through_external_symlink(tmp_path, monkeypatch, capsys, indirect):
    checkout = Path(__file__).resolve().parents[1]
    if indirect:
        link = tmp_path / "checkout-link"
        link.symlink_to(checkout, target_is_directory=True)
        state = link / "forbidden-benchmark-state"
    else:
        state = checkout / "forbidden-benchmark-state"
    main, calls = benchmark_cli(monkeypatch, ["--state", state, "--output", tmp_path / "report.json"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and not calls
    assert "outside the checkout" in capsys.readouterr().err
    assert not state.exists()


def test_cli_rejects_reused_state_before_comparison(tmp_path, monkeypatch, capsys):
    state = tmp_path / "existing-state"
    state.mkdir()
    main, calls = benchmark_cli(monkeypatch, ["--state", state, "--output", tmp_path / "report.json"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and not calls
    assert "new state directory" in capsys.readouterr().err


@pytest.mark.parametrize("value", [[], "synthetic metadata", 4, None])
def test_cli_rejects_nonobject_metadata_before_comparison(tmp_path, monkeypatch, capsys, value):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(value))
    state = tmp_path / "fresh-state"
    main, calls = benchmark_cli(monkeypatch, ["--state", state, "--metadata", metadata,
                                              "--output", tmp_path / "report.json"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and not calls
    assert "reviewed JSON object" in capsys.readouterr().err
    assert not state.exists()


def test_cli_forwards_reviewed_metadata_and_fresh_external_state(tmp_path, suite, monkeypatch):
    state = tmp_path / "fresh-state"
    metadata = tmp_path / "metadata.json"
    reviewed = {"model": "synthetic-model", "adapter_identity": "unknown", "trial": 1}
    metadata.write_text(json.dumps(reviewed))
    output = tmp_path / "report.json"
    main, calls = benchmark_cli(monkeypatch, ["--state", state, "--suite", suite, "--metadata", metadata,
                                              "--output", output])
    assert main() == 0 and len(calls) == 1
    settings, selected, destination, options = calls[0]
    assert settings.root == state and selected == suite and destination == output
    assert options["metadata"] == reviewed and options["monitor"] == "synthetic-monitor"
    assert not state.exists() and not output.exists()


def fake_runtime(monkeypatch, suite, *, hang=False, change=False):
    models, engines = [], []
    class Model:
        def __init__(self, settings, metrics):
            self.settings, self.closed = settings, False
            models.append(self)
        async def close(self):
            self.closed = True
    class Engine:
        def __init__(self, settings, model, role_models):
            self.settings, self.model, self.role_models = settings, model, role_models
            self.cancelled = 0
            self.closed = False
            self.jobs = {}
            engines.append(self)
        async def create(self, project, prompt, **options):
            assert project == "benchmark" and prompt.endswith("\n/no_think")
            assert options["memory"] is False
            assert self.settings.projects == {"benchmark": (suite / "repository").resolve()}
            assert not (self.settings.projects["benchmark"] / "oracle.py").exists()
            workspace = self.settings.root / "workspace"
            shutil.copytree(self.settings.projects["benchmark"], workspace)
            self.data = {"id": "synthetic-task", "workspace": str(workspace), "answer": "Synthetic answer",
                         "status": "running", "events": [], "checks": []}
            self.options = options
            async def job():
                try:
                    if hang:
                        await asyncio.Future()
                    if change:
                        (workspace / "module.py").write_text("value = 9\n")
                    self.data["status"] = "completed"
                except asyncio.CancelledError:
                    self.data["status"] = "cancelled"
            self.jobs[self.data["id"]] = asyncio.create_task(job())
            return {"id": self.data["id"]}
        async def cancel(self, task_id):
            self.cancelled += 1
            self.jobs[task_id].cancel()
            await asyncio.gather(self.jobs[task_id], return_exceptions=True)
            self.data["status"] = "cancelled"
        def task(self, task_id):
            return self.data
        def diff(self, task_id):
            return {"changed_files": ["module.py"] if change else []}
        async def close(self):
            self.closed = True
            for job in self.jobs.values():
                if not job.done():
                    job.cancel()
            await asyncio.gather(*self.jobs.values(), return_exceptions=True)
            await self.model.close()
            await self.role_models.close()
    monkeypatch.setattr(benchmark, "MeteredModel", Model)
    monkeypatch.setattr(benchmark, "AgentEngine", Engine)
    return models, engines


def test_run_case_same_model_caps_and_json_protocol_for_every_role(tmp_path, suite, monkeypatch):
    models, engines = fake_runtime(monkeypatch, suite)
    async def grade(*args):
        return {"passed": True, "failures": []}
    monkeypatch.setattr(benchmark, "grade", grade)
    async def run():
        settings = Settings(root=tmp_path / "state", model_name="synthetic-model")
        case = benchmark.load_cases(suite)[0]
        for mode in ("single", "specialists"):
            monitor = Monitor()
            record = await benchmark.run_case(settings, suite, case, mode, monitor, safe(),
                                               timeout=1, max_calls=24, max_tokens=768, context_chars=14000)
            assert record["usable_passed"] and monitor.stopped
        assert len(models) == 8 and all(model.closed for model in models)
        for model in models:
            config = model.settings
            assert (config.model_name, config.model_url, config.max_steps, config.max_tokens, config.context_chars, config.protocol) == (
                "synthetic-model", "http://127.0.0.1:8081/v1", 24, 768, 14000, "json")
            assert config.orchestration["role_max_steps"] == 24
            assert not config.orchestration["router"]["enabled"]
        assert engines[0].options["orchestrated"] is False and engines[1].options["orchestrated"] is True
        assert all(engine.closed for engine in engines)
    asyncio.run(run())


@pytest.mark.parametrize("abort", [False, True])
def test_run_case_cancels_job_and_stops_watcher_after_timeout_or_safety_abort(tmp_path, suite, monkeypatch, abort):
    models, engines = fake_runtime(monkeypatch, suite, hang=True)
    grades = []
    async def grade(*args):
        grades.append(True)
        return {"passed": False, "failures": ["unfinished"]}
    monkeypatch.setattr(benchmark, "grade", grade)
    monitor = Monitor(abort=abort)
    async def run():
        record = await benchmark.run_case(Settings(root=tmp_path / "state"), suite, benchmark.load_cases(suite)[0],
            "single", monitor, safe(), timeout=0.02, max_calls=24, max_tokens=768, context_chars=14000)
        assert not record["completed"] and not record["usable_passed"]
        assert record["safety_abort"] is abort and record["timeout"] is (not abort)
        assert engines[0].cancelled == 1 and engines[0].closed and monitor.stopped
        assert all(model.closed for model in models)
        assert "synthetic-task" not in json.dumps(record)
        assert record["resources"]["samples"] == 1
        assert not grades if abort else grades == [True]
    asyncio.run(run())


def test_external_runner_cancellation_stops_watch_and_closes_model_clients(tmp_path, suite, monkeypatch):
    models, engines = fake_runtime(monkeypatch, suite, hang=True)
    async def forbidden_grade(*args):
        pytest.fail("Cancelled generation reached external oracle")
    monkeypatch.setattr(benchmark, "grade", forbidden_grade)
    monitor = Monitor()
    async def run():
        task = asyncio.create_task(benchmark.run_case(Settings(root=tmp_path / "state"), suite,
            benchmark.load_cases(suite)[0], "single", monitor, safe(), timeout=60,
            max_calls=24, max_tokens=768, context_chars=14000))
        await asyncio.sleep(0)
        assert len(engines) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert engines[0].closed and monitor.stopped
        assert all(job.done() for job in engines[0].jobs.values())
        assert all(model.closed for model in models)
    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_creation_failure_or_cancellation_closes_clients_before_watch_starts(tmp_path, suite, monkeypatch, cancel):
    models, engines = fake_runtime(monkeypatch, suite)
    entered = asyncio.Event()

    async def create(self, *args, **kwargs):
        entered.set()
        if cancel:
            await asyncio.Future()
        raise RuntimeError("Synthetic setup failure")

    monkeypatch.setattr(benchmark.AgentEngine, "create", create)
    monitor = Monitor()
    async def run():
        task = asyncio.create_task(benchmark.run_case(Settings(root=tmp_path / "state"), suite,
            benchmark.load_cases(suite)[0], "single", monitor, safe(), timeout=60,
            max_calls=24, max_tokens=768, context_chars=14000))
        await entered.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await task
        assert engines[0].closed and all(model.closed for model in models)
        assert len(models) == 4 and not engines[0].jobs
        assert not monitor.stopped, "A resource watcher started despite task creation failing"
    asyncio.run(run())


@pytest.mark.parametrize("pause", [0, 3])
def test_metered_models_share_cooldown_and_time_only_requests_including_failed_requests(tmp_path, monkeypatch, pause):
    clock, starts, waits = [0.0], [], []
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: clock[0])

    async def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(benchmark.asyncio, "sleep", sleep)

    def response(request):
        starts.append(clock[0])
        if len(starts) == 3:
            clock[0] += 1
            return httpx.Response(503, json={"error": "Synthetic unavailable endpoint"})
        clock[0] += 2
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": '{"final":"Synthetic completion"}'},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4,
                      "prompt_tokens_details": {"cached_tokens": 2}}})

    async def run():
        metrics = benchmark.new_metrics()
        metrics["_request_pause_seconds"] = pause
        settings = Settings(root=tmp_path / "state", protocol="json", model_name="synthetic-model")
        models = [benchmark.MeteredModel(settings, metrics) for _ in range(2)]
        try:
            for model in models:
                await model.http.aclose()
                model.http = httpx.AsyncClient(transport=httpx.MockTransport(response),
                    event_hooks={"request": [model.requested], "response": [model.received]})
            messages = [{"role": "system", "content": "Synthetic instruction"},
                        {"role": "user", "content": "Synthetic request"}]
            assert (await models[0].complete(messages, tool_names=[]))["content"] == "Synthetic completion"
            assert (await models[1].complete(messages, tool_names=[]))["content"] == "Synthetic completion"
            with pytest.raises(httpx.HTTPStatusError):
                await models[0].complete(messages, tool_names=[])
            assert starts == [0, 2 + pause, 4 + 2 * pause] and waits == [pause, pause]
            assert metrics["inference_seconds"] == 5 and clock[0] == 5 + 2 * pause
            assert metrics["_last_completed"] == 5 + 2 * pause
            assert metrics["http_attempts"] == 3 and metrics["http_errors"] == 1
            assert metrics["usage_responses"] == 2
            assert (metrics["prompt_tokens"], metrics["completion_tokens"], metrics["cached_prompt_tokens"]) == (20, 8, 4)
        finally:
            for model in models:
                await model.close()
    asyncio.run(run())


def test_read_only_changed_workspace_cannot_be_a_usable_pass_even_if_oracle_approves(tmp_path, suite, monkeypatch):
    fake_runtime(monkeypatch, suite, change=True)
    initial = source_manifest(suite / "repository")
    async def grade(*args):
        return {"passed": True, "failures": []}
    monkeypatch.setattr(benchmark, "grade", grade)
    async def run():
        record = await benchmark.run_case(Settings(root=tmp_path / "state"), suite, benchmark.load_cases(suite)[1],
            "single", Monitor(), safe(), timeout=1, max_calls=24, max_tokens=768, context_chars=14000)
        assert record["read_only_integrity"] is False
        assert record["usable_passed"] is False
        assert source_manifest(suite / "repository") == initial
    asyncio.run(run())


def grade_tools(monkeypatch, response, *, inspect_copy=None):
    class Tools:
        def __init__(self, root, state_root, command_timeout):
            self.root = root
            assert command_timeout == 20
        async def run_command(self, argv, timeout):
            assert argv[:2] == ["python", "_benchmark_oracle/oracle.py"] and timeout == 20
            if inspect_copy:
                inspect_copy(self.root)
            return response
    monkeypatch.setattr(benchmark, "WorkspaceTools", Tools)


def oracle_task(tmp_path):
    workspace = tmp_path / "generated-workspace"
    workspace.mkdir()
    (workspace / "module.py").write_text("value = 2\n")
    return {"id": "synthetic-task", "workspace": str(workspace), "answer": "Synthetic answer",
            "benchmark_case": "edit_alpha", "events": [
                {"type": "tool_result", "data": {"name": "read_file", "result": {"path": "module.py"}}},
                {"type": "tool_result", "data": {"name": "read_file", "result": {"error": "missing", "path": "missing.py"}}}]}


def test_oracle_is_added_only_to_grading_copy_and_cannot_mutate_generated_source(tmp_path, suite, monkeypatch):
    task = oracle_task(tmp_path)
    workspace = tmp_path / "generated-workspace"
    original = source_manifest(workspace)
    original_fixture = source_manifest(suite)
    def inspect(root):
        assert (root / "_benchmark_oracle" / "oracle.py").is_file()
        assert json.loads((root / "_benchmark_reads.json").read_text()) == ["module.py"]
        (root / "module.py").write_text("value = 99\n")
    grade_tools(monkeypatch, {"exit_code": 0, "timed_out": False, "output": json.dumps({"passed": True, "failures": []})}, inspect_copy=inspect)
    report = asyncio.run(benchmark.grade(suite, task, tmp_path / "state"))
    assert report["passed"]
    assert source_manifest(workspace) == original and source_manifest(suite) == original_fixture
    assert not (workspace / "_benchmark_oracle").exists()
    assert "Synthetic answer" not in json.dumps(report)


@pytest.mark.parametrize("exit_code,timed_out,body", [(1, False, {"passed": True, "failures": []}),
                                                       (None, False, {"passed": True, "failures": []}),
                                                       (0, True, {"passed": True, "failures": []}),
                                                       (0, False, {"passed": True, "failures": ["behavior_failed"]}),
                                                       (0, False, [])])
def test_oracle_never_passes_on_nonzero_exit_timeout_invalid_body_or_failed_assertions(tmp_path, suite, monkeypatch, exit_code, timed_out, body):
    grade_tools(monkeypatch, {"exit_code": exit_code, "timed_out": timed_out, "output": json.dumps(body)})
    report = asyncio.run(benchmark.grade(suite, oracle_task(tmp_path), tmp_path / "state"))
    assert report["passed"] is False


def test_grade_does_not_follow_generated_symlinks_to_external_files(tmp_path, suite, monkeypatch):
    task = oracle_task(tmp_path)
    external = tmp_path / "external.txt"
    external.write_text("synthetic-outside-workspace")
    (tmp_path / "generated-workspace" / "external-link.txt").symlink_to(external)
    def inspect(root):
        assert not (root / "external-link.txt").exists(), "Host grading read a model-created external symlink"
    grade_tools(monkeypatch, {"exit_code": 0, "timed_out": False, "output": json.dumps({"passed": True, "failures": []})}, inspect_copy=inspect)
    report = asyncio.run(benchmark.grade(suite, task, tmp_path / "state"))
    assert report["passed"] is False


@pytest.mark.parametrize("rest", [float("nan"), float("inf")])
def test_invalid_rest_budget_is_rejected_before_model_execution(tmp_path, suite, monkeypatch, rest):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid rest budget reached model execution")
    monkeypatch.setattr(benchmark, "run_case", forbidden)
    with pytest.raises(ValueError):
        asyncio.run(benchmark.compare(Settings(root=tmp_path / "state"), suite, tmp_path / "report.json", monitor=Monitor(), rest_seconds=rest))


@pytest.mark.parametrize("pause", [-1, float("nan"), float("inf"), 61])
def test_invalid_request_pause_never_starts_model_execution(tmp_path, suite, monkeypatch, pause):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid request pause reached execution")
    monkeypatch.setattr(benchmark, "run_case", forbidden)
    with pytest.raises(ValueError, match="Invalid benchmark budgets"):
        asyncio.run(benchmark.compare(Settings(root=tmp_path / "state"), suite, tmp_path / "report.json",
                                      monitor=Monitor(), request_pause_seconds=pause))
    assert not (tmp_path / "report.json").exists()
