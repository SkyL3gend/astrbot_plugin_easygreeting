"""Behavioral tests without a running AstrBot or external model."""

import asyncio
import copy
import importlib.util
import json
import logging
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scheduler  # noqa: E402


def defaults():
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))

    def values(items):
        return {
            key: values(field["items"])
            if field["type"] == "object"
            else field["default"]
            for key, field in items.items()
        }

    return values(schema)


class Config(dict):
    def save_config(self):
        pass


class FakeStar:
    def __init__(self, context):
        self.context = context

    async def get_kv_data(self, key, default=None):
        return copy.deepcopy(self.context.storage.get(key, default))

    async def put_kv_data(self, key, value):
        self.context.storage[key] = copy.deepcopy(value)


class FakeChain:
    def message(self, text):
        self.text = text
        return self


def command(name):
    def decorate(function):
        function.command_name = name
        return function

    return decorate


def permission(value):
    def decorate(function):
        function.permission = value
        return function

    return decorate


def load_plugin():
    # Only the adapter boundary is replaced. Tests execute the real scheduler,
    # generation routing, persistence, send gating, and cancellation logic.
    modules = {}
    for name in ("astrbot", "astrbot.api", "astrbot.api.event", "astrbot.api.star"):
        modules[name] = types.ModuleType(name)
    modules["astrbot.api"].AstrBotConfig = Config
    log = logging.getLogger("easygreeting-test")
    log.addHandler(logging.NullHandler())
    log.propagate = False
    modules["astrbot.api"].logger = log
    modules["astrbot.api.event"].AstrMessageEvent = object
    modules["astrbot.api.event"].MessageChain = FakeChain
    modules["astrbot.api.event"].filter = types.SimpleNamespace(
        command=command,
        permission_type=permission,
        PermissionType=types.SimpleNamespace(ADMIN="admin"),
    )
    modules["astrbot.api.star"].Star = FakeStar
    modules["astrbot.api.star"].Context = object
    package = types.ModuleType("easygreeting_under_test")
    package.__path__ = [str(ROOT)]
    modules[package.__name__] = package
    modules[f"{package.__name__}.scheduler"] = scheduler
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.main", ROOT / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


plugin_module = load_plugin()


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.config = defaults()
        self.config["timezone"] = "UTC"
        self.config["private_whitelist"] = ["bot:FriendMessage:alice"]
        self.now = datetime(2026, 10, 24, 7, 0, tzinfo=timezone.utc)

    def test_defaults_are_valid_and_prompt_defaults_match(self):
        settings = scheduler.Settings.from_config(defaults())
        self.assertEqual(settings.tz.utcoffset(None), timedelta(hours=8))
        self.assertEqual(settings.targets, frozenset())
        self.assertEqual(settings.greetings[0].prompt, scheduler.DEFAULT_MORNING_PROMPT)
        self.assertEqual(settings.greetings[1].prompt, scheduler.DEFAULT_NIGHT_PROMPT)

    def test_blank_values_use_defaults(self):
        self.config.update(
            {
                "timezone": "",
                "provider_id": None,
                "group_whitelist": ["", "  "],
                "private_whitelist": [None, ""],
                "generation_timeout_seconds": "",
                "enabled": "",
                "morning": {
                    "time": "",
                    "random_before_minutes": "",
                    "random_after_minutes": None,
                    "prompt": None,
                    "enabled": "",
                },
                "night": None,
            }
        )
        settings = scheduler.Settings.from_config(self.config)
        self.assertEqual(settings.tz.utcoffset(None), timedelta(hours=8))
        self.assertEqual(settings.provider_id, "")
        self.assertEqual(settings.timeout, 60)
        self.assertEqual(settings.targets, frozenset())
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.greetings[0].at.strftime("%H:%M"), "08:00")
        self.assertEqual(settings.greetings[0].before, 30)
        self.assertEqual(settings.greetings[0].after, 30)
        self.assertEqual(settings.greetings[0].prompt, scheduler.DEFAULT_MORNING_PROMPT)
        self.assertEqual(settings.greetings[1].at.strftime("%H:%M"), "23:00")
        self.assertEqual(settings.greetings[1].prompt, scheduler.DEFAULT_NIGHT_PROMPT)

    def test_random_draws_stay_within_each_window(self):
        settings = scheduler.Settings.from_config(self.config)
        jobs = {}
        scheduler.plan_jobs(settings, jobs, self.now)
        self.assertEqual(len(jobs), 4)
        for job in jobs.values():
            greeting = next(g for g in settings.greetings if g.kind == job["kind"])
            day = datetime.fromisoformat(job["day"]).date()
            start, end = greeting.window(day, settings.tz)
            self.assertGreaterEqual(job["scheduled_at"], start)
            self.assertLess(job["scheduled_at"], end)

    def test_zero_range_covers_one_minute(self):
        self.config["morning"].update(random_before_minutes=0, random_after_minutes=0)
        greeting = scheduler.Settings.from_config(self.config).greetings[0]
        start, end = greeting.window(self.now.date(), timezone.utc)
        self.assertEqual(end - start, 60)

    def test_reload_retains_existing_random_draws(self):
        settings = scheduler.Settings.from_config(self.config)
        jobs = {}
        scheduler.plan_jobs(settings, jobs, self.now)
        original = copy.deepcopy(jobs)
        with patch.object(
            scheduler.RNG, "uniform", side_effect=AssertionError("redraw")
        ):
            scheduler.plan_jobs(settings, jobs, self.now + timedelta(minutes=5))
        self.assertEqual(jobs, original)

    def test_each_chat_gets_an_independent_draw(self):
        self.config["private_whitelist"].append("bot:FriendMessage:bob")
        settings = scheduler.Settings.from_config(self.config)
        jobs = {}
        with patch.object(
            scheduler.RNG, "uniform", side_effect=lambda a, b: a + len(jobs)
        ):
            scheduler.plan_jobs(settings, jobs, self.now)
        today_morning = [
            job["scheduled_at"]
            for job in jobs.values()
            if job["day"] == self.now.date().isoformat() and job["kind"] == "morning"
        ]
        self.assertEqual(len(set(today_morning)), 2)

    def test_midnight_window_includes_yesterdays_anchor(self):
        self.config["morning"]["enabled"] = False
        self.config["night"].update(
            time="23:50", random_before_minutes=0, random_after_minutes=30
        )
        settings = scheduler.Settings.from_config(self.config)
        now = self.now.replace(hour=0, minute=5)
        jobs = {}
        scheduler.plan_jobs(settings, jobs, now)
        yesterday = (now.date() - timedelta(days=1)).isoformat()
        self.assertTrue(any(job["day"] == yesterday for job in jobs.values()))

    def test_early_anchor_can_start_on_previous_day(self):
        self.config["night"].update(
            time="00:10", random_before_minutes=30, random_after_minutes=0
        )
        greeting = scheduler.Settings.from_config(self.config).greetings[1]
        start, _ = greeting.window(self.now.date(), timezone.utc)
        self.assertEqual(datetime.fromtimestamp(start, timezone.utc).day, 23)

    def test_first_enable_midwindow_only_uses_remaining_time(self):
        settings = scheduler.Settings.from_config(self.config)
        now = self.now.replace(hour=8, minute=20)
        jobs = {}
        scheduler.plan_jobs(settings, jobs, now)
        self.assertTrue(
            all(job["scheduled_at"] >= now.timestamp() for job in jobs.values())
        )

    def test_expired_windows_are_not_backfilled(self):
        settings = scheduler.Settings.from_config(self.config)
        jobs = {}
        scheduler.plan_jobs(settings, jobs, self.now.replace(hour=12))
        self.assertFalse(
            any(
                job["day"] == self.now.date().isoformat() and job["kind"] == "morning"
                for job in jobs.values()
            )
        )

    def test_group_switch_does_not_disable_private(self):
        self.config["group_whitelist"] = ["bot:GroupMessage:group"]
        self.config["group_enabled"] = False
        settings = scheduler.Settings.from_config(self.config)
        self.assertEqual(settings.targets, frozenset({"bot:FriendMessage:alice"}))

    def test_invalid_entry_in_disabled_whitelist_is_ignored(self):
        self.config["group_enabled"] = False
        self.config["group_whitelist"] = ["123456"]
        settings = scheduler.Settings.from_config(self.config)
        self.assertEqual(settings.targets, frozenset({"bot:FriendMessage:alice"}))

    def test_invalid_config_is_rejected(self):
        invalid = [
            {"timezone": "UTC+25:00"},
            {"private_whitelist": ["123456"]},
            {"private_whitelist": ["bot:GroupMessage:group"]},
            {"private_whitelist": "not-a-list"},
            {"generation_timeout_seconds": 0},
            {"morning": {"time": "24:00"}},
            {"morning": {"time": "8:00"}},
            {"morning": {"random_before_minutes": -1}},
            {"morning": {"random_before_minutes": 1.5}},
            {"morning": {"random_before_minutes": 720, "random_after_minutes": 720}},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                scheduler.Settings.from_config({**self.config, **changes})

    def test_only_pending_jobs_move_after_time_change(self):
        settings = scheduler.Settings.from_config(self.config)
        jobs = {}
        scheduler.plan_jobs(settings, jobs, self.now)
        key = next(iter(jobs))
        jobs[key]["status"] = "sent"
        sent = copy.deepcopy(jobs[key])
        self.config["morning"]["time"] = "09:00"
        scheduler.plan_jobs(scheduler.Settings.from_config(self.config), jobs, self.now)
        self.assertEqual(jobs[key], sent)


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = Config(defaults())
        self.config["timezone"] = "UTC"
        self.target = "bot:FriendMessage:alice"
        self.config["private_whitelist"] = [self.target]
        self.config["morning"].update(random_before_minutes=0, random_after_minutes=0)
        self.config["night"]["enabled"] = False
        self.context = types.SimpleNamespace(
            storage={},
            get_current_chat_provider_id=AsyncMock(return_value="session-model"),
            llm_generate=AsyncMock(
                return_value=types.SimpleNamespace(completion_text="早安！")
            ),
            send_message=AsyncMock(return_value=True),
        )
        self.plugin = plugin_module.EasyGreeting(self.context, self.config)
        self.now = datetime(2026, 10, 24, 8, 0, 10, tzinfo=timezone.utc)
        self.plugin._now = lambda: self.now
        self.random_patch = patch.object(
            scheduler.RNG, "uniform", side_effect=lambda a, b: a
        )
        self.random_patch.start()

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.random_patch.stop()

    async def drain(self, plugin=None):
        plugin = plugin or self.plugin
        tasks = list(plugin._workers.values())
        if tasks:
            await asyncio.gather(*tasks)
        await asyncio.sleep(0)

    async def trigger(self):
        await self.plugin._tick()
        await self.drain()

    async def test_sends_once_and_routes_to_current_session_model(self):
        await self.trigger()
        await self.trigger()
        self.context.send_message.assert_awaited_once()
        self.context.get_current_chat_provider_id.assert_awaited_once_with(
            umo=self.target
        )
        self.assertEqual(
            self.context.llm_generate.await_args.kwargs["chat_provider_id"],
            "session-model",
        )
        target, chain = self.context.send_message.await_args.args
        self.assertEqual((target, chain.text), (self.target, "早安！"))

    async def test_explicit_provider_and_custom_prompt(self):
        self.config["provider_id"] = "greeting-model"
        self.config["morning"]["prompt"] = "请用猫娘语气问候。"
        await self.trigger()
        self.context.get_current_chat_provider_id.assert_not_awaited()
        kwargs = self.context.llm_generate.await_args.kwargs
        self.assertEqual(kwargs["chat_provider_id"], "greeting-model")
        self.assertIn("请用猫娘语气问候。", kwargs["prompt"])
        self.assertIn("私聊", kwargs["prompt"])

    async def test_empty_whitelist_makes_no_model_or_send_calls(self):
        self.config["private_whitelist"] = []
        await self.trigger()
        self.context.llm_generate.assert_not_awaited()
        self.context.send_message.assert_not_awaited()

    async def test_blank_config_does_not_raise_or_send(self):
        self.config.update(
            {
                "timezone": "",
                "provider_id": None,
                "group_whitelist": [""],
                "private_whitelist": [""],
                "generation_timeout_seconds": "",
                "morning": {
                    "time": "",
                    "random_before_minutes": "",
                    "random_after_minutes": None,
                    "prompt": None,
                },
                "night": {},
            }
        )
        await self.trigger()
        self.context.llm_generate.assert_not_awaited()
        self.context.send_message.assert_not_awaited()

    async def test_all_disabled_switches_block_automatic_send(self):
        for key in ("enabled", "private_enabled", "morning"):
            with self.subTest(key=key):
                if key == "morning":
                    self.config["morning"]["enabled"] = False
                else:
                    self.config[key] = False
                await self.trigger()
                self.context.llm_generate.assert_not_awaited()
                if key == "morning":
                    self.config["morning"]["enabled"] = True
                else:
                    self.config[key] = True

    async def test_reload_keeps_sent_jobs_and_cancels_scheduler(self):
        await self.trigger()
        restored = plugin_module.EasyGreeting(self.context, self.config)
        restored._now = lambda: self.now
        try:
            await restored.initialize()
            await restored._tick()
            await self.drain(restored)
            self.context.send_message.assert_awaited_once()
        finally:
            await restored.terminate()
        self.assertIsNone(restored._scheduler)
        self.assertEqual(restored._workers, {})

    async def test_ambiguous_send_reservation_is_not_replayed(self):
        await self.trigger()
        for job in self.plugin._jobs.values():
            if job["status"] == "sent":
                job["status"] = "sending"
        await self.plugin._persist()
        restored = plugin_module.EasyGreeting(self.context, self.config)
        restored._now = lambda: self.now
        try:
            await restored.initialize()
            await restored._tick()
            await self.drain(restored)
            self.context.send_message.assert_awaited_once()
        finally:
            await restored.terminate()

    async def test_model_failure_retries_at_most_three_times_inside_window(self):
        self.config["morning"]["random_after_minutes"] = 10
        self.context.llm_generate.side_effect = RuntimeError("model offline")
        await self.trigger()
        await self.trigger()
        self.assertEqual(self.context.llm_generate.await_count, 1)
        for _ in range(4):
            self.now += timedelta(seconds=61)
            await self.trigger()
        self.assertEqual(self.context.llm_generate.await_count, 3)
        self.context.send_message.assert_not_awaited()

    async def test_blank_model_response_is_not_sent(self):
        self.context.llm_generate.return_value.completion_text = "  "
        await self.trigger()
        self.context.send_message.assert_not_awaited()

    async def test_generation_finishing_outside_window_is_dropped(self):
        async def delayed(**kwargs):
            self.now += timedelta(minutes=2)
            return types.SimpleNamespace(completion_text="太迟了")

        self.context.llm_generate.side_effect = delayed
        await self.trigger()
        self.context.send_message.assert_not_awaited()

    async def test_removing_whitelist_while_generating_blocks_send(self):
        async def remove(**kwargs):
            self.config["private_whitelist"] = []
            return types.SimpleNamespace(completion_text="不应发送")

        self.context.llm_generate.side_effect = remove
        await self.trigger()
        self.context.send_message.assert_not_awaited()

    async def test_changed_time_during_generation_blocks_old_job(self):
        async def change(**kwargs):
            self.config["morning"]["time"] = "09:00"
            return types.SimpleNamespace(completion_text="不应发送")

        self.context.llm_generate.side_effect = change
        await self.trigger()
        self.context.send_message.assert_not_awaited()

    async def test_send_failure_does_not_retry_and_spam(self):
        self.context.send_message.return_value = False
        await self.trigger()
        await self.trigger()
        self.context.send_message.assert_awaited_once()
        self.assertTrue(
            any(job["status"] == "failed" for job in self.plugin._jobs.values())
        )

    async def test_failed_plan_write_prevents_side_effects_then_recovers(self):
        original = self.plugin.put_kv_data
        self.plugin.put_kv_data = AsyncMock(side_effect=RuntimeError("disk full"))
        with self.assertRaises(RuntimeError):
            await self.plugin._tick()
        self.context.llm_generate.assert_not_awaited()
        self.plugin.put_kv_data = original
        await self.trigger()
        self.context.send_message.assert_awaited_once()

    async def test_reservation_write_failure_prevents_send(self):
        original = self.plugin.put_kv_data

        async def fail_reservation(key, value):
            if any(job["status"] == "sending" for job in value.values()):
                raise RuntimeError("disk full")
            await original(key, value)

        self.plugin.put_kv_data = fail_reservation
        await self.plugin._tick()
        tasks = list(self.plugin._workers.values())
        await asyncio.gather(*tasks, return_exceptions=True)
        self.context.send_message.assert_not_awaited()

    async def test_reservation_write_failure_recovers_on_next_tick(self):
        original = self.plugin.put_kv_data
        failed = False

        async def fail_once(key, value):
            nonlocal failed
            if not failed and any(job["status"] == "sending" for job in value.values()):
                failed = True
                raise RuntimeError("disk full")
            await original(key, value)

        self.plugin.put_kv_data = fail_once
        await self.plugin._tick()
        tasks = list(self.plugin._workers.values())
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0)

        today = self.now.date().isoformat()
        job = next(
            job
            for job in self.plugin._jobs.values()
            if job["day"] == today and job["kind"] == "morning"
        )
        self.assertEqual(job["status"], "pending")
        self.context.send_message.assert_not_awaited()

        self.plugin.put_kv_data = original
        await self.trigger()
        self.context.send_message.assert_awaited_once()

    async def test_redraw_during_generation_does_not_send_stale_job(self):
        calls = 0

        async def generate(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                self.config["morning"]["time"] = "09:00"
                await self.plugin._tick()
                self.config["morning"]["time"] = "08:00"
                await self.plugin._tick()
            return types.SimpleNamespace(completion_text=f"问候{calls}")

        self.context.llm_generate.side_effect = generate
        await self.trigger()
        self.context.send_message.assert_not_awaited()
        await self.trigger()
        self.context.send_message.assert_awaited_once()

    async def test_concurrent_generation_is_limited_and_other_chats_still_send(self):
        self.config["private_whitelist"] = [f"bot:FriendMessage:{i}" for i in range(5)]
        release = asyncio.Event()
        three_started = asyncio.Event()
        started = 0

        async def blocking(**kwargs):
            nonlocal started
            started += 1
            if started == 3:
                three_started.set()
            await release.wait()
            return types.SimpleNamespace(completion_text="早安")

        self.context.llm_generate.side_effect = blocking
        await self.plugin._tick()
        await asyncio.wait_for(three_started.wait(), timeout=2)
        self.assertEqual(started, 3)
        release.set()
        await self.drain()
        self.assertEqual(self.context.send_message.await_count, 5)

    async def test_termination_cancels_inflight_generation(self):
        started = asyncio.Event()

        async def blocking(**kwargs):
            started.set()
            await asyncio.Event().wait()

        self.context.llm_generate.side_effect = blocking
        await self.plugin._tick()
        await asyncio.wait_for(started.wait(), timeout=2)
        await self.plugin.terminate()
        self.context.send_message.assert_not_awaited()
        self.assertEqual(self.plugin._workers, {})

    async def test_admin_commands_manage_both_whitelists(self):
        for target, key in (
            ("bot:GroupMessage:class", "group_whitelist"),
            ("bot:FriendMessage:bob", "private_whitelist"),
        ):
            event = types.SimpleNamespace(
                unified_msg_origin=target, plain_result=lambda text: text
            )
            replies = [reply async for reply in self.plugin.enable_session(event)]
            self.assertIn("已加入", replies[0])
            self.assertIn(target, self.config[key])
            replies = [reply async for reply in self.plugin.disable_session(event)]
            self.assertIn("已移出", replies[0])
            self.assertNotIn(target, self.config[key])

    async def test_all_commands_require_astrbot_admin(self):
        for name in (
            "show_session",
            "enable_session",
            "disable_session",
            "show_status",
            "test_greeting",
        ):
            self.assertEqual(getattr(self.plugin, name).permission, "admin")

    async def test_manual_test_uses_proactive_send_without_consuming_daily_job(self):
        self.config["private_whitelist"] = []
        event = types.SimpleNamespace(
            unified_msg_origin=self.target, plain_result=lambda text: text
        )
        replies = [reply async for reply in self.plugin.test_greeting(event, "早安")]
        self.assertEqual(replies, [])
        self.context.send_message.assert_awaited_once()
        self.assertEqual(self.plugin._jobs, {})

    async def test_failed_whitelist_save_rolls_back_membership(self):
        event = types.SimpleNamespace(
            unified_msg_origin="bot:FriendMessage:bob", plain_result=lambda text: text
        )
        with patch.object(self.config, "save_config", side_effect=OSError("disk full")):
            replies = [reply async for reply in self.plugin.enable_session(event)]
        self.assertIn("开启失败", replies[0])
        self.assertNotIn(event.unified_msg_origin, self.config["private_whitelist"])

    async def test_repeated_initialize_keeps_only_one_scheduler(self):
        await self.plugin.initialize()
        task = self.plugin._scheduler
        await self.plugin.initialize()
        self.assertIs(self.plugin._scheduler, task)

    async def test_invalid_config_does_not_prevent_initialize(self):
        self.config["morning"]["time"] = "8:00"
        await self.plugin.initialize()
        self.assertIsNotNone(self.plugin._scheduler)
        self.context.send_message.assert_not_awaited()

    async def test_invalid_state_is_reset_instead_of_failing_initialize(self):
        self.context.storage[plugin_module.STATE_KEY] = ["broken"]
        self.config["private_whitelist"] = []
        await self.plugin.initialize()
        self.assertEqual(self.plugin._jobs, {})
        self.assertIsNotNone(self.plugin._scheduler)


if __name__ == "__main__":
    unittest.main()
