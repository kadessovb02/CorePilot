#!/usr/bin/env python3
"""Standalone external oracle for the independent scheduling suite."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, fields
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib
from importlib.machinery import SourceFileLoader
import inspect
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tokenize
import unittest
from unittest.mock import patch


SUITE = Path(__file__).resolve().parent
CASE_KEYS = {"id", "prompt", "read_only", "required_reads", "oracle", "category", "max_calls", "timeout_seconds"}
TARGETS = {
    "fix_duration_ceiling": "scheduling/time_math.py", "fix_half_open_overlap": "scheduling/time_math.py",
    "fix_empty_intersection": "scheduling/time_math.py", "fix_recurrence_end": "scheduling/recurrence.py",
    "fix_month_rollover": "scheduling/recurrence.py", "fix_adjacent_merge": "scheduling/availability.py",
    "fix_exact_fit_slot": "scheduling/availability.py", "fix_resource_capacity": "scheduling/availability.py",
    "fix_utc_conversion": "scheduling/reminders.py", "fix_reminder_direction": "scheduling/reminders.py",
    "fix_single_cancellation": "scheduling/service.py", "fix_chronological_agenda": "scheduling/service.py",
}
SIGNATURES = {
    "time_math": {"duration_minutes": ("start", "end"), "overlaps": ("left", "right"), "clip_window": ("window", "horizon")},
    "recurrence": {"expand_dates": ("start", "end", "weekdays"), "monthly_occurrences": ("start", "months")},
    "availability": {"merge_windows": ("windows",), "find_slot": ("windows", "duration", "not_before"),
                     "remaining_capacity": ("resource", "bookings", "capacity")},
    "reminders": {"utc_timestamp": ("when",), "reminder_time": ("start", "lead")},
    "service": {"drop_booking": ("bookings", "identifier"), "sorted_agenda": ("bookings",)},
}
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".corepilot-tmp", "_benchmark_oracle"}
RUNNER_FILES = {"_benchmark_answer.json", "_benchmark_answer.txt", "_benchmark_answer.text", "_benchmark_reads.json", ".git"}


def load_cases(path=None):
    rows = json.loads(Path(path or SUITE / "cases.json").read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("cases must be a nonempty list")
    seen = set()
    for case in rows:
        if not isinstance(case, dict) or set(case) != CASE_KEYS:
            raise ValueError("unsupported case schema")
        for key in ("id", "prompt", "category", "oracle"):
            if not isinstance(case[key], str) or not case[key].strip():
                raise ValueError("nonempty case strings required")
        if case["id"] in seen or not re.fullmatch(r"[a-z][a-z0-9_]{1,79}", case["id"]) or case["oracle"] != case["id"]:
            raise ValueError("invalid or duplicate case oracle")
        seen.add(case["id"])
        if not isinstance(case["read_only"], bool) or case["id"] not in set(TARGETS) | set(SEMANTIC_RULES):
            raise ValueError("unsupported case or mode")
        if (case["id"] in TARGETS) == case["read_only"]:
            raise ValueError("case mode disagrees with oracle")
        reads = case["required_reads"]
        if not isinstance(reads, list) or not reads or any(not isinstance(name, str) for name in reads) or len(set(reads)) != len(reads):
            raise ValueError("required_reads must be distinct relative paths")
        for name in reads:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or not (SUITE / "repository" / name).is_file():
                raise ValueError("required read is outside fixture")
        for key, low, high in (("max_calls", 12, 24), ("timeout_seconds", 90, 150)):
            if isinstance(case[key], bool) or not isinstance(case[key], int) or not low <= case[key] <= high:
                raise ValueError("invalid case budget")
    return rows


def source_files(root):
    result, problems = {}, []
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in list(dirs):
            path = Path(directory) / name
            if name in IGNORED_DIRS:
                dirs.remove(name)
            elif path.is_symlink():
                problems.append(path.relative_to(root).as_posix())
                dirs.remove(name)
        for name in names:
            path = Path(directory) / name
            if directory == str(root) and name in RUNNER_FILES:
                continue
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or path.stat().st_size > 2_000_000:
                problems.append(relative)
            else:
                result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result, sorted(problems)


def style_errors(workspace, names):
    problems = []
    for name in sorted(names):
        path = workspace / name
        if not name.endswith(".py") or path.is_symlink() or not path.is_file():
            continue
        try:
            content = path.read_bytes()
            tree = ast.parse(content)
            for token in tokenize.tokenize(io.BytesIO(content).readline):
                if token.type == tokenize.INDENT and ("\t" in token.string or len(token.string) % 4):
                    problems.append("Indentation contract: " + name)
                    break
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not re.fullmatch(r"[a-z_][a-z0-9_]*", node.name):
                    problems.append("Function naming contract: " + name)
                roots = []
                if isinstance(node, ast.Import):
                    roots = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    roots = [(node.module or "").split(".")[0]]
                if any(root != "scheduling" and root not in sys.stdlib_module_names for root in roots):
                    problems.append("Dependency contract: " + name)
        except (SyntaxError, UnicodeError, tokenize.TokenError):
            problems.append("Invalid Python: " + name)
    return sorted(set(problems))


@contextmanager
def workspace_modules(workspace):
    saved = {key: value for key, value in sys.modules.items() if key == "scheduling" or key.startswith("scheduling.")}
    old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, str(workspace))
    sys.dont_write_bytecode = True
    original = SourceFileLoader.get_code

    def fresh(loader, fullname):
        if fullname == "scheduling" or fullname.startswith("scheduling."):
            return compile(loader.get_data(loader.path), loader.path, "exec", dont_inherit=True)
        return original(loader, fullname)

    try:
        with patch.object(SourceFileLoader, "get_code", fresh):
            importlib.invalidate_caches()
            modules = {"facade": importlib.import_module("scheduling")}
            modules.update({key: importlib.import_module("scheduling." + key) for key in SIGNATURES})
            if not Path(modules["facade"].__file__).resolve().is_relative_to(workspace):
                raise ValueError("wrong workspace import")
            yield modules
    finally:
        for key in list(sys.modules):
            if key == "scheduling" or key.startswith("scheduling."):
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode


class BehaviorChecks(unittest.TestCase):
    modules = None

    def moment(self, hours=0, minutes=0, seconds=0, microseconds=0):
        return datetime(2032, 5, 3) + timedelta(hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)

    def window(self, start, end):
        return self.modules["time_math"].Window(self.moment(start), self.moment(end))

    def booking(self, identifier="b1", resource="room-a", start=9, end=10, cancelled=False):
        return self.modules["time_math"].Booking(identifier, resource, self.moment(start), self.moment(end), cancelled)

    def test_api_compatibility(self):
        facade, time_math = self.modules["facade"], self.modules["time_math"]
        self.assertEqual(facade.__all__, ["Scheduler", "Window", "Booking", "expand_dates"])
        self.assertIs(facade.Scheduler, self.modules["service"].Scheduler)
        self.assertIs(facade.Window, time_math.Window)
        self.assertIs(facade.Booking, time_math.Booking)
        self.assertIs(facade.expand_dates, self.modules["recurrence"].expand_dates)
        for module, functions in SIGNATURES.items():
            for name, parameters in functions.items():
                signature = inspect.signature(getattr(self.modules[module], name))
                self.assertEqual(tuple(signature.parameters), parameters)
                self.assertTrue(all(value.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for value in signature.parameters.values()))
        self.assertEqual(inspect.signature(self.modules["recurrence"].expand_dates).parameters["weekdays"].default, tuple(range(7)))
        self.assertIsNone(inspect.signature(self.modules["availability"].find_slot).parameters["not_before"].default)
        self.assertEqual(inspect.signature(self.modules["reminders"].reminder_time).parameters["lead"].default, timedelta(minutes=15))
        self.assertEqual(inspect.signature(facade.Scheduler.reminders).parameters["lead"].default, timedelta(minutes=15))
        self.assertEqual(tuple(inspect.signature(facade.Scheduler.book).parameters), ("self", "identifier", "resource", "start", "end"))
        for name, parameters in (("__init__", ("self",)), ("cancel", ("self", "identifier")), ("agenda", ("self",)), ("reminders", ("self", "lead"))):
            self.assertEqual(tuple(inspect.signature(getattr(facade.Scheduler, name)).parameters), parameters)
        self.assertEqual([field.name for field in fields(time_math.Booking)], ["identifier", "resource", "start", "end", "cancelled"])
        self.assertEqual([field.name for field in fields(time_math.Window)], ["start", "end"])
        self.assertEqual(inspect.signature(time_math.Booking).parameters["cancelled"].default, False)
        self.assertEqual(tuple(inspect.signature(time_math.Booking).parameters), ("identifier", "resource", "start", "end", "cancelled"))
        self.assertEqual(tuple(inspect.signature(time_math.Window).parameters), ("start", "end"))
        for value in (self.window(1, 2), self.booking()):
            with self.assertRaises(FrozenInstanceError):
                value.start = self.moment(3)

    def test_preserved_behavior(self):
        time_math, recurrence, availability, reminders, service = (self.modules[key] for key in SIGNATURES)
        self.assertEqual(time_math.duration_minutes(self.moment(), self.moment(minutes=2)), 2)
        self.assertTrue(time_math.overlaps(self.window(1, 4), self.window(2, 3)))
        self.assertFalse(time_math.overlaps(self.window(1, 2), self.window(3, 4)))
        self.assertEqual(time_math.clip_window(self.window(1, 4), self.window(2, 5)), self.window(2, 4))
        with self.assertRaises(ValueError):
            time_math.Window(self.moment(2), self.moment(2))
        with self.assertRaises(TypeError):
            time_math.Window("start", "end")
        self.assertEqual(recurrence.expand_dates(date(2032, 5, 3), date(2032, 5, 4), (0,)), [date(2032, 5, 3)])
        self.assertEqual(recurrence.monthly_occurrences(date(2032, 4, 15), 2), [date(2032, 4, 15), date(2032, 5, 15)])
        self.assertEqual(availability.merge_windows([self.window(2, 4), self.window(1, 3)]), [self.window(1, 4)])
        self.assertEqual(availability.find_slot([self.window(9, 12)], timedelta(hours=1)), self.moment(9))
        self.assertEqual(availability.remaining_capacity("room-a", [self.booking()], 3), 2)
        utc = self.moment(9).replace(tzinfo=timezone.utc)
        self.assertEqual(reminders.utc_timestamp(utc), "2032-05-03T09:00:00+00:00")
        self.assertEqual(reminders.reminder_time(utc, timedelta(0)), utc)
        one, two = self.booking(), self.booking("b2", "room-b", 11, 12)
        self.assertEqual(service.drop_booking([one, two], "b1"), [two])
        self.assertEqual(service.sorted_agenda([one, two]), [one, two])
        scheduler = service.Scheduler()
        stored = scheduler.book("b1", "room-a", self.moment(9), self.moment(10))
        self.assertIs(scheduler.bookings["b1"], stored)
        scheduler.cancel("unknown")
        self.assertIs(scheduler.bookings["b1"], stored)
        self.assertEqual(service.Scheduler().bookings, {})

    def fix_duration_ceiling(self):
        function = self.modules["time_math"].duration_minutes
        for delta, expected in ((timedelta(0), 0), (timedelta(microseconds=1), 1), (timedelta(seconds=59), 1),
                                (timedelta(seconds=60), 1), (timedelta(seconds=61), 2), (timedelta(days=2, seconds=1), 2881)):
            with self.subTest(delta=delta):
                self.assertEqual(function(self.moment(), self.moment() + delta), expected)
                self.assertIs(type(function(self.moment(), self.moment() + delta)), int)
        start = self.moment(2).replace(tzinfo=timezone(timedelta(hours=2)))
        end = self.moment(1, 1, 1).replace(tzinfo=timezone(timedelta(hours=1)))
        self.assertEqual(function(start, end), 2)
        with self.assertRaises(ValueError):
            function(self.moment(2), self.moment(1))
        with self.assertRaises(TypeError):
            function("start", self.moment())

    def fix_half_open_overlap(self):
        function = self.modules["time_math"].overlaps
        for left, right, expected in ((self.window(1, 2), self.window(2, 3), False),
                                      (self.window(1, 3), self.window(2, 4), True),
                                      (self.window(1, 5), self.window(2, 3), True),
                                      (self.window(1, 2), self.window(1, 2), True),
                                      (self.window(1, 2), self.window(3, 4), False)):
            with self.subTest(left=left, right=right):
                self.assertEqual(function(left, right), expected)
                self.assertEqual(function(right, left), expected)
        window = self.modules["time_math"].Window
        aware = lambda hour: self.moment(hour).replace(tzinfo=timezone.utc)
        self.assertFalse(function(window(aware(1), aware(2)), window(aware(2), aware(3))))

    def fix_empty_intersection(self):
        function = self.modules["time_math"].clip_window
        for window, horizon in ((self.window(1, 2), self.window(2, 3)), (self.window(1, 2), self.window(3, 4)),
                                (self.window(3, 4), self.window(1, 2))):
            with self.subTest(window=window, horizon=horizon):
                self.assertIsNone(function(window, horizon))
        self.assertEqual(function(self.window(1, 5), self.window(2, 3)), self.window(2, 3))
        self.assertEqual(function(self.window(2, 3), self.window(1, 5)), self.window(2, 3))
        self.assertEqual(function(self.window(1, 4), self.window(2, 5)), self.window(2, 4))

    def fix_recurrence_end(self):
        function = self.modules["recurrence"].expand_dates
        monday, sunday = date(2032, 5, 3), date(2032, 5, 9)
        self.assertEqual(function(monday, monday), [monday])
        self.assertEqual(function(monday, sunday), [monday + timedelta(days=n) for n in range(7)])
        self.assertEqual(function(monday, sunday, iter((0, 6))), [monday, sunday])
        self.assertEqual(function(monday, sunday, ()), [])
        self.assertEqual(function(monday, monday, (6,)), [])
        for days in ((True,), (-1,), (7,), (1.0,)):
            with self.assertRaises(ValueError):
                function(monday, sunday, days)
        with self.assertRaises(ValueError):
            function(sunday, monday)
        with self.assertRaises(TypeError):
            function(datetime(2032, 5, 3), sunday)

    def fix_month_rollover(self):
        function = self.modules["recurrence"].monthly_occurrences
        for start, count, expected in ((date(2031, 12, 30), 3, [date(2031, 12, 30), date(2032, 1, 30), date(2032, 2, 29)]),
                                       (date(2032, 1, 31), 3, [date(2032, 1, 31), date(2032, 2, 29), date(2032, 3, 31)]),
                                       (date(2031, 1, 31), 3, [date(2031, 1, 31), date(2031, 2, 28), date(2031, 3, 31)]),
                                       (date(2032, 7, 4), 0, [])):
            with self.subTest(start=start):
                self.assertEqual(function(start, count), expected)
        start = date(2030, 6, 17)
        self.assertEqual(len(function(start, 25)), 25)
        self.assertEqual(function(start, 25)[-1], date(2032, 6, 17))
        for count in (True, 1.0, "2"):
            with self.assertRaises(TypeError):
                function(start, count)
        with self.assertRaises(ValueError):
            function(start, -1)
        with self.assertRaises(TypeError):
            function(datetime(2030, 6, 17), 1)

    def fix_adjacent_merge(self):
        function = self.modules["availability"].merge_windows
        rows = [self.window(3, 4), self.window(1, 2), self.window(2, 3), self.window(6, 7), self.window(1, 4)]
        before = rows.copy()
        self.assertEqual(function(iter(rows)), [self.window(1, 4), self.window(6, 7)])
        self.assertEqual(function([self.window(3, 4), self.window(1, 2), self.window(2, 3)]), [self.window(1, 4)])
        self.assertEqual(rows, before)
        self.assertEqual(function([]), [])
        self.assertIsNot(function(rows), rows)
        self.assertEqual(function([self.window(1, 5), self.window(2, 3), self.window(5, 6)]), [self.window(1, 6)])

    def fix_exact_fit_slot(self):
        function = self.modules["availability"].find_slot
        self.assertEqual(function([self.window(9, 10)], timedelta(hours=1)), self.moment(9))
        self.assertEqual(function([self.window(9, 12)], timedelta(hours=1), self.moment(11)), self.moment(11))
        self.assertEqual(function(iter([self.window(12, 13), self.window(9, 10)]), timedelta(hours=1)), self.moment(9))
        self.assertIsNone(function([self.window(9, 10)], timedelta(hours=2)))
        self.assertIsNone(function([self.window(9, 10)], timedelta(minutes=1), self.moment(10)))
        self.assertIsNone(function([], timedelta(minutes=1)))
        tz = timezone(timedelta(hours=-3))
        start = self.moment(9).replace(tzinfo=tz)
        end = self.moment(10).replace(tzinfo=tz)
        result = function([self.modules["time_math"].Window(start, end)], timedelta(hours=1))
        self.assertEqual(result, start)
        self.assertIs(result.tzinfo, tz)
        for duration in (timedelta(0), timedelta(seconds=-1)):
            with self.assertRaises(ValueError):
                function([], duration)
        with self.assertRaises(TypeError):
            function([], 60)

    def fix_resource_capacity(self):
        function = self.modules["availability"].remaining_capacity
        rows = [self.booking("one"), self.booking("two", "room-b"), self.booking("three", cancelled=True)]
        before = rows.copy()
        for resource, cap, expected in (("room-a", 3, 2), ("room-b", 3, 2), ("unknown", 3, 3), ("room-a", 0, 0)):
            self.assertEqual(function(resource, iter(rows), cap), expected)
        self.assertEqual(function("room-a", [self.booking("one"), self.booking("two")], 1), 0)
        self.assertEqual(function("room-a", [], 4), 4)
        self.assertEqual(rows, before)
        for capacity in (True, 1.0, "2"):
            with self.assertRaises(TypeError):
                function("room-a", [], capacity)
        with self.assertRaises(ValueError):
            function("room-a", [], -1)

    def fix_utc_conversion(self):
        function = self.modules["reminders"].utc_timestamp
        values = [(datetime(2032, 1, 1, 0, 30, tzinfo=timezone(timedelta(hours=2))), "2031-12-31T22:30:00+00:00"),
                  (datetime(2032, 12, 31, 23, 15, tzinfo=timezone(timedelta(hours=-4))), "2033-01-01T03:15:00+00:00"),
                  (datetime(2032, 5, 3, 9, 1, 2, 4567, tzinfo=timezone.utc), "2032-05-03T09:01:02+00:00")]
        for value, expected in values:
            original = (value, value.tzinfo)
            self.assertEqual(function(value), expected)
            self.assertEqual((value, value.tzinfo), original)
        with self.assertRaises(ValueError):
            function(self.moment(1))
        with self.assertRaises(TypeError):
            function("2032-05-03")

    def fix_reminder_direction(self):
        function = self.modules["reminders"].reminder_time
        start = self.moment(0, 5).replace(tzinfo=timezone(timedelta(hours=3)))
        self.assertEqual(function(start), start - timedelta(minutes=15))
        self.assertEqual(function(start, timedelta(days=2)), start - timedelta(days=2))
        self.assertEqual(function(start, timedelta(0)), start)
        self.assertIs(function(start).tzinfo, start.tzinfo)
        for lead in (-1, "15"):
            with self.assertRaises(TypeError):
                function(start, lead)
        with self.assertRaises(ValueError):
            function(start, timedelta(seconds=-1))
        scheduler = self.modules["service"].Scheduler()
        scheduler.book("visit", "room-c", start, start + timedelta(hours=1))
        self.assertEqual(scheduler.reminders(), {"visit": start - timedelta(minutes=15)})
        self.assertEqual(scheduler.reminders(timedelta(minutes=5)), {"visit": start - timedelta(minutes=5)})

    def fix_single_cancellation(self):
        function = self.modules["service"].drop_booking
        one, two, three = self.booking("one"), self.booking("two", start=11, end=12), self.booking("three", "room-b")
        rows = [one, two, three]
        self.assertEqual(function(iter(rows), "one"), [two, three])
        self.assertIs(function(rows, "one")[0], two)
        self.assertEqual(function(rows, "unknown"), rows)
        self.assertIsNot(function(rows, "unknown"), rows)
        self.assertEqual(rows, [one, two, three])
        self.assertEqual(function([], "one"), [])
        scheduler, independent = self.modules["service"].Scheduler(), self.modules["service"].Scheduler()
        for item in rows:
            scheduler.book(item.identifier, item.resource, item.start, item.end)
        scheduler.cancel("one")
        scheduler.cancel("one")
        self.assertEqual(list(scheduler.bookings), ["two", "three"])
        self.assertEqual(independent.bookings, {})

    def fix_chronological_agenda(self):
        function = self.modules["service"].sorted_agenda
        late, first_tie, second_tie = self.booking("a", start=12, end=13), self.booking("z", start=9, end=10), self.booking("b", "room-c")
        rows = [late, first_tie, second_tie]
        self.assertEqual(function(iter(rows)), [first_tie, second_tie, late])
        self.assertIs(function(rows)[0], first_tie)
        self.assertEqual(rows, [late, first_tie, second_tie])
        self.assertIsNot(function(rows), rows)
        self.assertEqual(function([]), [])
        scheduler = self.modules["service"].Scheduler()
        for value in rows:
            scheduler.book(value.identifier, value.resource, value.start, value.end)
        self.assertEqual([value.identifier for value in scheduler.agenda()], ["z", "b", "a"])
        tz = timezone.utc
        aware = [self.modules["time_math"].Booking(value.identifier, value.resource, value.start.replace(tzinfo=tz), value.end.replace(tzinfo=tz)) for value in rows]
        self.assertEqual(function(aware), [aware[1], aware[2], aware[0]])

    def inspection_source_facts(self):
        case = self.case_id
        service, time_math, recurrence, reminders = (self.modules[key] for key in ("service", "time_math", "recurrence", "reminders"))
        if case == "inspect_booking_storage":
            one, two = service.Scheduler(), service.Scheduler()
            stored = one.book("one", "room-a", self.moment(1), self.moment(2))
            self.assertIs(one.bookings["one"], stored)
            replacement = one.book("one", "room-b", self.moment(3), self.moment(4))
            self.assertEqual(one.bookings, {"one": replacement})
            self.assertEqual(two.bookings, {})
            self.assertTrue(time_math.Booking.__dataclass_params__.frozen)
        elif case == "inspect_policy_boundary":
            scheduler = service.Scheduler()
            scheduler.book("one", "room-a", self.moment(1), self.moment(3))
            scheduler.book("two", "room-a", self.moment(2), self.moment(4))
            self.assertEqual(len(scheduler.bookings), 2)
            source = ast.parse(inspect.getsource(service.Scheduler.book).strip())
            called = {ast.unparse(node.func) for node in ast.walk(source) if isinstance(node, ast.Call)}
            self.assertFalse(called & {"remaining_capacity", "find_slot", "overlaps"})
        elif case == "inspect_endpoint_behavior":
            self.assertTrue(time_math.overlaps(self.window(1, 2), self.window(2, 3)))
            with self.assertRaises(ValueError):
                time_math.clip_window(self.window(1, 2), self.window(2, 3))
        elif case == "inspect_recurrence_behavior":
            monday = date(2032, 5, 3)
            self.assertEqual(monday.weekday(), 0)
            self.assertEqual(recurrence.expand_dates(monday, monday), [])
            self.assertEqual(recurrence.expand_dates(monday, monday + timedelta(days=1), (0,)), [monday])
            for start in (date(2032, 12, 15), date(2032, 1, 31)):
                with self.assertRaises(ValueError):
                    recurrence.monthly_occurrences(start, 2)
        elif case == "inspect_timezone_behavior":
            value = datetime(2032, 5, 3, 9, tzinfo=timezone(timedelta(hours=2)))
            self.assertEqual(reminders.utc_timestamp(value), "2032-05-03T09:00:00+00:00")
            self.assertNotEqual(reminders.utc_timestamp(value), value.astimezone(timezone.utc).isoformat(timespec="seconds"))
            with self.assertRaises(ValueError):
                reminders.utc_timestamp(self.moment(9))
        elif case == "inspect_notification_boundary":
            scheduler = service.Scheduler()
            scheduler.book("one", "room-a", self.moment(9), self.moment(10))
            self.assertEqual(scheduler.reminders(), {"one": self.moment(9, 15)})
            self.assertNotIn("now(", inspect.getsource(service) + inspect.getsource(reminders))
        elif case == "inspect_cancellation_behavior":
            scheduler = service.Scheduler()
            for identifier, resource in (("one", "room-a"), ("two", "room-a"), ("three", "room-b")):
                scheduler.book(identifier, resource, self.moment(9), self.moment(10))
            scheduler.cancel("unknown")
            self.assertEqual(len(scheduler.bookings), 3)
            scheduler.cancel("one")
            self.assertEqual(list(scheduler.bookings), ["three"])
            self.assertFalse(scheduler.bookings["three"].cancelled)
        elif case == "inspect_external_dependencies":
            self.assertEqual(service.Scheduler().bookings, {})
            roots = []
            for key in SIGNATURES:
                for node in ast.walk(ast.parse(inspect.getsource(self.modules[key]))):
                    if isinstance(node, ast.Import):
                        roots.extend(alias.name.split(".")[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and not node.level:
                        roots.append((node.module or "").split(".")[0])
            self.assertTrue(all(root in sys.stdlib_module_names for root in roots))
        else:
            self.fail("unknown inspection oracle")


SEMANTIC_RULES = {
    "inspect_booking_storage": [("constructs and stores record", r"book.{0,150}Booking|Booking.{0,150}book"),
        ("frozen validation", r"frozen"), ("identifier dictionary", r"(?:dict|dictionary).{0,120}identifier|identifier.{0,120}(?:dict|dictionary)"),
        ("same-id replacement", r"replac|overwrit"), ("instance isolation", r"(?:separate|independent|isolat).{0,100}instance|per.instance"),
        ("four public exports", r"(?:(?:public\s+)?(?:exports?|facade)|__all__|re.exports?)(?=[^.!?\n]{0,240}\bScheduler\b)(?=[^.!?\n]{0,240}\bWindow\b)(?=[^.!?\n]{0,240}\bBooking\b)(?=[^.!?\n]{0,240}\bexpand_dates\b)[^.!?\n]+")],
    "inspect_policy_boundary": [("policy not enforced", r"(?:not|no|without|doesn't).{0,80}(?:enforc|reject|conflict|capacity)|(?:policy|capacity|conflict).{0,80}(?:not|no|independent)"),
        ("availability separate", r"(?:availability|remaining_capacity).{0,100}(?:helper|separate|independent)"),
        ("overlapping records accepted", r"overlap.{0,100}(?:accept|stor|both)|(?:accept|stor).{0,100}overlap")],
    "inspect_endpoint_behavior": [("half-open contract", r"half.open"), ("touching currently true", r"(?:touch|endpoint|adjacen).{0,100}(?:true|overlap)|true.{0,100}(?:touch|endpoint)"),
        ("nonpositive intersection rejected", r"clip_window.{0,160}(?:raises|ValueError)|(?:raises|ValueError).{0,100}clip_window"),
        ("positive Window duration", r"positive.{0,70}(?:duration|Window)|Window.{0,100}(?:positive|reject)")],
    "inspect_recurrence_behavior": [("daily end excluded", r"end.{0,80}(?:exclud|exclusive|not included)|exclud.{0,80}end"),
        ("weekday numbering", r"Monday.{0,30}0.{0,100}Sunday.{0,30}6"),
        ("current month computation", r"replace.{0,100}month|month.{0,100}replace|month.{0,70}(?:plus|\+|adding|increment(?:ed)?).{0,35}(?:offset|index)|(?:adds|increments?).{0,60}(?:offset|index).{0,60}(?:original|start(?:ing)?).{0,35}month"),
        ("rollover and clamp absent", r"(?:rollover|December|month.end|clamp).{0,160}(?:not|fail|ValueError|invalid)|(?:not|fail).{0,100}(?:rollover|clamp)")],
    "inspect_timezone_behavior": [("timezone label changed without clock conversion", r"replace.{0,100}tzinfo|(?:relabel|re.tag).{0,100}(?:UTC|timezone)|(?:assigns?|attaches?|sets?|labels?).{0,60}UTC.{0,100}(?:keeping|retaining|preserving|unchanged).{0,90}(?:clock|fields|wall.time)|(?:keeps?|retains?|preserves?).{0,70}(?:clock|fields|wall.time).{0,90}(?:assigns?|attaches?|sets?|labels?).{0,60}UTC"),
        ("instant altered", r"(?:instant|offset).{0,120}(?:wrong|chang|not|fail)|(?:wrong|chang|not|fail).{0,120}instant"),
        ("awareness required", r"(?:aware|naive).{0,100}(?:require|reject|ValueError)|require.{0,100}aware"),
        ("seconds and UTC suffix", r"seconds.{0,120}\+00:00|\+00:00.{0,120}seconds")],
    "inspect_notification_boundary": [("returned mapping", r"(?:dict|mapping).{0,100}(?:identifier|booking|datetime)|(?:identifier|booking).{0,100}(?:dict|mapping)"),
        ("helper flow", r"reminders.{0,120}reminder_time|reminder_time.{0,120}reminders"),
        ("default fifteen", r"15.{0,40}minutes|fifteen.{0,40}minutes"),
        ("current after-start", r"(?:after|later).{0,80}start|start\s*\+\s*lead"),
        ("no dispatch or worker", r"(?:no|not|without|doesn't).{0,150}(?:email|send|worker|dispatch)"),
        ("no live clock", r"(?:no|not|without|doesn't).{0,100}(?:clock|now)")],
    "inspect_cancellation_behavior": [("matched-resource filter", r"(?:filter|remov|drop).{0,120}(?:matched|same).{0,60}resource|resource.{0,120}filter"),
        ("other same-resource record removed", r"(?:other|both|all).{0,100}(?:resource|bookings).{0,100}(?:remov|drop)|remov.{0,100}(?:both|other)"),
        ("state replacement", r"(?:rebuild|replac).{0,100}(?:dict|bookings)|bookings.{0,100}(?:rebuild|replac)"),
        ("unknown-id no-op", r"unknown.{0,100}(?:unchanged|no.op|nothing|retain)"),
        ("cancelled flag not mutated", r"cancelled.{0,80}(?:not|false|unchanged)|(?:not|doesn't).{0,80}cancelled")],
    "inspect_external_dependencies": [("instance memory", r"(?:in.memory|instance).{0,100}(?:dict|bookings)|(?:dict|bookings).{0,100}in.memory"),
        ("no durable database", r"(?:no|not|without).{0,100}(?:database|durable|persist)"),
        ("no transport dispatch", r"(?:no|not|without).{0,100}(?:http|transport|dispatch)"),
        ("standard library", r"standard.library"),
        ("scope limitation", r"external.{0,120}(?:unverified|unknown|not|unsupported)|(?:only|inspected).{0,100}fixture")],
}
FALSE_ASSERTIONS = {
    "inspect_booking_storage": [r"(?:Scheduler|Window|Booking|expand_dates)\s+is\s+not\s+(?:exported|public)",
                                r"(?:public\s+exports?|__all__)\s+(?:omit|exclude|lack)(?:s)?\s+(?:Scheduler|Window|Booking|expand_dates)"],
    "inspect_policy_boundary": [r"(?:automatically|always)\s+(?:rejects|prevents)\s+overlap", r"book\s+(?:enforces|checks)\s+capacity"],
    "inspect_recurrence_behavior": [r"daily\s+end\s+(?:date\s+)?is\s+included", r"monthly_occurrences\s+handles\s+December\s+rollover"],
    "inspect_timezone_behavior": [r"utc_timestamp\s+preserves\s+(?:the\s+)?instant",
                                  r"utc_timestamp\s+(?:converts|translates)\s+(?:the\s+)?(?:original\s+)?(?:instant|clock\s+fields|datetime)\s+to\s+UTC"],
    "inspect_notification_boundary": [r"(?:sends|dispatches)\s+(?:an?\s+)?email", r"(?:starts|runs)\s+(?:an?\s+)?background\s+worker"],
    "inspect_external_dependencies": [r"(?:uses|stores\s+bookings\s+in)\s+(?:SQLite|PostgreSQL|a\s+database)", r"dispatches\s+reminders\s+over\s+HTTP"],
}


def semantic_grade(case_id, answer):
    missing = [label for label, pattern in SEMANTIC_RULES[case_id] if not re.search(pattern, answer, re.I | re.S)]
    contradictions = []
    for pattern in FALSE_ASSERTIONS.get(case_id, ()):
        for match in re.finditer(pattern, answer, re.I):
            prefix = re.split(r"[.!?\n]", answer[:match.start()])[-1][-100:]
            tail = re.split(r"[.!?\n]", answer[match.end():], maxsplit=1)[0]
            negated = re.search(r"\b(?:not|never|doesn't|does\s+not)\s+$", prefix, re.I)
            quoted_refutation = (re.search(r"\b(?:claim|assertion)\b.{0,40}$", prefix, re.I)
                                and re.search(r"\bis\s+(?:false|incorrect|unsupported)\b", tail, re.I))
            if not negated and not quoted_refutation:
                contradictions.append("Explicit unsupported affirmative statement")
    return {"passed": not missing and not contradictions, "missing_criteria": missing, "contradictions": contradictions,
            "limitation": "Finite English answer criteria and explicit contradiction patterns; manual factual review remains required."}


def evaluate(case_id, workspace, answer=None, observed_reads=(), *, facts_only=False):
    cases = {case["id"]: case for case in load_cases()}
    if case_id not in cases:
        raise ValueError("unknown case")
    case = cases[case_id]
    if facts_only and not case["read_only"]:
        raise ValueError("facts_only is not an edit grade")
    workspace = Path(workspace).resolve()
    baseline, baseline_problems = source_files(SUITE / "repository")
    actual, source_problems = source_files(workspace)
    if baseline_problems:
        raise ValueError("baseline integrity failed")
    changed = sorted(name for name, digest in baseline.items() if actual.get(name) != digest)
    unexpected = sorted(set(actual) - set(baseline))
    allowed = set() if case["read_only"] else {TARGETS[case_id]}
    allowed_test = lambda name: not case["read_only"] and bool(re.fullmatch(r"tests/(?:__init__|test_[a-zA-Z0-9_]+)\.py", name))
    integrity = list(source_problems) + ["Unexpected change: " + name for name in changed if name not in allowed]
    integrity += ["Unexpected file: " + name for name in unexpected if not allowed_test(name)]
    guidance = style_errors(workspace, set(changed) | set(unexpected))
    reads = {PurePosixPath(name).as_posix() for name in observed_reads if isinstance(name, str)
             and not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts}
    missing = sorted(set(case["required_reads"]) - reads)
    report = {"schema_version": 1, "case_id": case_id, "oracle": case_id,
              "oracle_kind": "inspection_source_facts" if facts_only else "inspection_semantic_proxy" if case["read_only"] else "unittest",
              "passed": False, "integrity_errors": integrity, "style_errors": guidance, "changed_files": changed,
              "required_reads_missing": missing, "tests_run": 0, "failure_count": 0, "error_count": 0, "failures": [], "errors": []}
    captured = io.StringIO()
    with redirect_stdout(captured), redirect_stderr(captured):
        try:
            if source_problems:
                raise ValueError("workspace includes unsafe file entries")
            with workspace_modules(workspace) as modules:
                scoped = type("ScopedChecks", (BehaviorChecks,), {"modules": modules, "case_id": case_id})
                names = ("test_api_compatibility", "test_preserved_behavior", "inspection_source_facts" if case["read_only"] else case_id)
                result = unittest.TestResult()
                unittest.TestSuite(scoped(name) for name in names).run(result)
                report["tests_run"] = result.testsRun
                for key in ("failures", "errors"):
                    entries = getattr(result, key)
                    report["failure_count" if key == "failures" else "error_count"] = len(entries)
                    report[key] = [{"test": getattr(item[0], "test_case", item[0])._testMethodName,
                                    "detail": item[1].replace(str(workspace), "<workspace>").replace(str(SUITE), "<oracle>")[-600:]}
                                   for item in entries[:3]]
                report["passed"] = result.wasSuccessful() and not integrity and not guidance
        except Exception as exc:
            report["error_count"] += 1
            report["errors"].append({"test": "import_workspace", "detail": type(exc).__name__ + ": " + str(exc).replace(str(workspace), "<workspace>")[:400]})
    after, after_problems = source_files(workspace)
    if after != actual or after_problems != source_problems:
        report["integrity_errors"].append("Source changed while grading")
        report["passed"] = False
    if case["read_only"] and not facts_only:
        report["semantic"] = semantic_grade(case_id, answer or "")
        report["passed"] = report["passed"] and not missing and report["semantic"]["passed"]
    report["read_gate_enforced"] = case["read_only"] and not facts_only
    report["captured_output_chars"] = len(captured.getvalue())
    report["fixture_fingerprint"] = hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--answer-file")
    parser.add_argument("--reads-file")
    parser.add_argument("--facts-only", action="store_true")
    args = parser.parse_args()
    try:
        answer = None
        if args.answer_file:
            path = Path(args.answer_file)
            if path.stat().st_size > 80_000:
                raise ValueError("answer too large")
            answer = path.read_text()
            if path.suffix == ".json":
                parsed = json.loads(answer)
                answer = parsed.get("answer") if isinstance(parsed, dict) else parsed
            if not isinstance(answer, str) or len(answer) > 20_000:
                raise ValueError("invalid answer")
        reads = json.loads(Path(args.reads_file).read_text()) if args.reads_file else []
        if not isinstance(reads, list) or len(reads) > 200:
            raise ValueError("invalid read trace")
        report = evaluate(args.case, args.workspace, answer, reads, facts_only=args.facts_only)
    except Exception as exc:
        report = {"schema_version": 1, "case_id": args.case, "passed": False, "oracle_kind": "configuration_error",
                  "errors": [{"detail": type(exc).__name__ + ": " + str(exc)[:400]}]}
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
