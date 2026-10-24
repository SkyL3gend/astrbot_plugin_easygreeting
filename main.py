"""简单早晚安：在每日随机时间生成并主动发送问候。"""

from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star

from .scheduler import Greeting, Settings, plan_jobs, session_kind

STATE_KEY = "daily_greeting_jobs_v1"
POLL_SECONDS = 5
MAX_GENERATION_ATTEMPTS = 3
RETRY_SECONDS = 60


class EasyGreeting(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._jobs: dict[str, dict] = {}
        self._saved_jobs: dict[str, dict] = {}
        self._scheduler: asyncio.Task | None = None
        self._workers: dict[str, asyncio.Task] = {}
        self._state_lock = asyncio.Lock()
        self._generation_slots = asyncio.Semaphore(3)
        self._last_config_error = ""

    async def initialize(self):
        """Restore draws and start on initial load as well as plugin reload."""
        if self._scheduler and not self._scheduler.done():
            return
        state = await self.get_kv_data(STATE_KEY, {})
        if not isinstance(state, dict):
            logger.error("[简单早晚安] 持久化计划格式错误，已重置")
            state = {}
            try:
                await self.put_kv_data(STATE_KEY, state)
            except Exception as exc:
                logger.error(f"[简单早晚安] 重置持久化计划失败：{exc}")
        self._jobs = state
        self._saved_jobs = copy.deepcopy(state)
        self._scheduler = asyncio.create_task(
            self._run(), name="easygreeting-scheduler"
        )

    async def terminate(self):
        """Stop all work so a hot reload does not leave a second scheduler."""
        tasks = list(self._workers.values())
        if self._scheduler:
            tasks.append(self._scheduler)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._workers.clear()
        self._scheduler = None

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    async def _persist(self):
        # A snapshot prevents concurrent workers from changing a value mid-write.
        async with self._state_lock:
            snapshot = copy.deepcopy(self._jobs)
            await self.put_kv_data(STATE_KEY, snapshot)
            self._saved_jobs = snapshot

    async def _run(self):
        while True:
            try:
                await self._tick()
                self._last_config_error = ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                message = str(exc)
                if message != self._last_config_error:
                    logger.error(f"[简单早晚安] 调度失败：{message}")
                    self._last_config_error = message
            await asyncio.sleep(POLL_SECONDS)

    async def _tick(self):
        settings = Settings.from_config(self.config)
        now = self._now()
        # A preceding failed write must be retried before any side effects.
        plan_jobs(settings, self._jobs, now)
        if self._jobs != self._saved_jobs:
            await self._persist()
        if not settings.enabled:
            return
        enabled_kinds = {g.kind for g in settings.greetings if g.enabled}
        timestamp = now.timestamp()
        for key, job in self._jobs.items():
            if (
                key in self._workers
                or job["status"] != "pending"
                or job["target"] not in settings.targets
                or job["kind"] not in enabled_kinds
                or timestamp < max(job["scheduled_at"], job["retry_at"])
                or timestamp >= job["expires_at"]
            ):
                continue
            task = asyncio.create_task(self._deliver(key), name="easygreeting-delivery")
            self._workers[key] = task
            task.add_done_callback(
                lambda finished, job_key=key: self._worker_done(job_key, finished)
            )

    def _worker_done(self, key: str, task: asyncio.Task):
        self._workers.pop(key, None)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error(f"[简单早晚安] 任务执行失败：{error}")

    def _allowed_greeting(
        self, key: str, job: dict
    ) -> tuple[Settings, Greeting] | None:
        # A redraw replaces the dict; a stale worker must not send it.
        if self._jobs.get(key) is not job or job["status"] != "pending":
            return None
        settings = Settings.from_config(self.config)
        timestamp = self._now().timestamp()
        if (
            not settings.enabled
            or job["target"] not in settings.targets
            or timestamp < job["scheduled_at"]
            or timestamp >= job["expires_at"]
        ):
            return None
        for greeting in settings.greetings:
            if greeting.kind == job["kind"] and greeting.enabled:
                # A configuration edit during generation invalidates the old
                # window; the scheduler will draw a replacement next tick.
                if greeting.signature(settings.timezone_name) == job["signature"]:
                    return settings, greeting
        return None

    async def _generate(
        self, target: str, settings: Settings, greeting: Greeting
    ) -> str:
        async with asyncio.timeout(settings.timeout):
            provider_id = settings.provider_id
            if not provider_id:
                provider_id = await self.context.get_current_chat_provider_id(umo=target)
            if not provider_id:
                raise RuntimeError("没有可用的聊天模型，请在插件配置中选择模型")
            now = self._now().astimezone(settings.tz)
            scene = "群聊，请面向群里的大家问候" if session_kind(target) == "group" else "私聊"
            prompt = (
                f"{greeting.prompt}\n\n"
                f"本次问候类型：{greeting.label}\n"
                f"当前时间：{now:%Y-%m-%d %H:%M}（{settings.timezone_name}）\n"
                f"发送场景：{scene}"
            )
            response = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
            )
            text = (response.completion_text or "").strip()
            if not text:
                raise RuntimeError("模型返回了空内容")
            return text

    async def _send_text(self, target: str, text: str):
        async with asyncio.timeout(30):
            sent = await self.context.send_message(target, MessageChain().message(text))
        if sent is False:
            raise RuntimeError("找不到目标消息平台，或平台不支持主动发送")

    async def _deliver(self, key: str):
        async with self._generation_slots:
            job = self._jobs.get(key)
            if job is None:
                return
            allowed = self._allowed_greeting(key, job)
            if job["status"] != "pending" or not allowed:
                return
            settings, greeting = allowed
            if job["attempts"] >= MAX_GENERATION_ATTEMPTS:
                job["status"] = "failed"
                await self._persist()
                return
            job["attempts"] += 1
            try:
                # No message is sent if saving the attempt fails.
                await self._persist()
                text = await self._generate(job["target"], settings, greeting)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                job["retry_at"] = self._now().timestamp() + RETRY_SECONDS
                if job["attempts"] >= MAX_GENERATION_ATTEMPTS:
                    job["status"] = "failed"
                await self._persist()
                logger.warning(
                    f"[简单早晚安] {greeting.label}生成失败"
                    f"（第 {job['attempts']} 次）：{exc}"
                )
                return
            # Disabling a chat while the model is running takes effect before send.
            if not self._allowed_greeting(key, job):
                return
            # Persist the send reservation first. If interrupted here, a reload
            # skips it rather than risking a second message from the same job.
            job["status"] = "sending"
            try:
                await self._persist()
            except Exception:
                # Nothing was sent; let the next tick retry after storage recovers.
                job["status"] = "pending"
                raise
            try:
                await self._send_text(job["target"], text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                job["status"] = "failed"
                logger.error(f"[简单早晚安] {greeting.label}发送失败：{exc}")
            else:
                job["status"] = "sent"
                logger.info(f"[简单早晚安] {greeting.label}已发送")
            await self._persist()

    @filter.command("早晚安会话")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def show_session(self, event: AstrMessageEvent):
        """显示当前会话标识，供填写白名单。"""
        yield event.plain_result(f"当前会话标识：\n{event.unified_msg_origin}")

    @filter.command("早晚安开启")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def enable_session(self, event: AstrMessageEvent):
        """将当前群聊或私聊加入白名单。"""
        target = event.unified_msg_origin
        try:
            kind = session_kind(target)
            key = f"{kind}_whitelist"
            original = list(self.config.get(key, []))
            self.config[key] = list(dict.fromkeys([*original, target]))
            try:
                settings = Settings.from_config(self.config)
                self.config.save_config()
            except Exception:
                self.config[key] = original
                raise
            label = "群聊" if kind == "group" else "私聊"
            extra = ""
            if not settings.enabled or not self.config.get(f"{kind}_enabled", True):
                extra = "\n对应的总开关当前已关闭，请在插件配置中开启。"
            yield event.plain_result(f"当前{label}已加入早晚安白名单。{extra}")
        except Exception as exc:
            yield event.plain_result(f"开启失败：{exc}")

    @filter.command("早晚安关闭")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def disable_session(self, event: AstrMessageEvent):
        """将当前群聊或私聊移出白名单。"""
        try:
            target = event.unified_msg_origin
            key = f"{session_kind(target)}_whitelist"
            original = list(self.config.get(key, []))
            self.config[key] = [value for value in original if value.strip() != target]
            try:
                self.config.save_config()
            except Exception:
                self.config[key] = original
                raise
            yield event.plain_result("当前会话已移出早晚安白名单。")
        except Exception as exc:
            yield event.plain_result(f"关闭失败：{exc}")

    @filter.command("早晚安状态")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def show_status(self, event: AstrMessageEvent):
        """查看当前会话是否启用及待执行计划。"""
        try:
            settings = Settings.from_config(self.config)
            target = event.unified_msg_origin
            enabled = settings.enabled and target in settings.targets
            lines = [
                "简单早晚安",
                f"时区：{settings.timezone_name}",
                f"当前会话：{'已启用' if enabled else '未启用'}",
            ]
            for greeting in settings.greetings:
                detail = "已关闭"
                if greeting.enabled:
                    detail = (
                        f"{greeting.at:%H:%M}，提前 {greeting.before} / "
                        f"延后 {greeting.after} 分钟"
                    )
                lines.append(f"{greeting.label}：{detail}")
                candidates = [
                    job for job in self._jobs.values()
                    if enabled
                    and job["target"] == target
                    and job["kind"] == greeting.kind
                    and job["status"] == "pending"
                    and job["expires_at"] > self._now().timestamp()
                ]
                if greeting.enabled and candidates:
                    next_job = min(candidates, key=lambda job: job["scheduled_at"])
                    at = datetime.fromtimestamp(next_job["scheduled_at"], settings.tz)
                    lines.append(f"下次计划：{at:%m-%d %H:%M:%S}")
            yield event.plain_result("\n".join(lines))
        except Exception as exc:
            yield event.plain_result(f"配置有误：{exc}")

    @filter.command("早晚安测试")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def test_greeting(self, event: AstrMessageEvent, kind: str = "早安"):
        """立即生成早安或晚安测试消息，不影响当天定时计划。"""
        if kind not in {"早安", "晚安"}:
            yield event.plain_result("用法：/早晚安测试 早安 或 /早晚安测试 晚安")
            return
        try:
            settings = Settings.from_config(self.config)
            greeting = next(g for g in settings.greetings if g.label == kind)
            async with self._generation_slots:
                text = await self._generate(
                    event.unified_msg_origin, settings, greeting
                )
            await self._send_text(event.unified_msg_origin, text)
        except Exception as exc:
            logger.warning(f"[简单早晚安] 测试失败：{exc}")
            yield event.plain_result("生成或主动发送失败，请检查聊天模型、平台支持情况和日志。")
