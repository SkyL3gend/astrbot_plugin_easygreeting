"""Configuration and daily random scheduling, independent of AstrBot."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_MORNING_PROMPT = (
    "请主动发送一条自然、温暖的早安问候，祝对方今天心情愉快。"
    "用中文，控制在1至3句话，可适度使用表情。"
    "只输出要发送的问候正文，不要解释任务，不要虚构天气或对方的个人情况。"
)
DEFAULT_NIGHT_PROMPT = (
    "请主动发送一条自然、温柔的晚安问候，提醒对方放松、好好休息。"
    "用中文，控制在1至3句话，可适度使用表情。"
    "只输出要发送的问候正文，不要解释任务，不要虚构天气或对方的个人情况。"
)
RNG = random.SystemRandom()
DEFAULT_TIMEZONE = "UTC+08:00"
DEFAULT_MORNING_TIME = "08:00"
DEFAULT_NIGHT_TIME = "23:00"
DEFAULT_BEFORE_MINUTES = 30
DEFAULT_AFTER_MINUTES = 30
DEFAULT_GENERATION_TIMEOUT = 60


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def read_text(value: Any, default: str) -> str:
    return default if is_blank(value) else str(value).strip()


def read_bool(value: Any, default: bool = True) -> bool:
    if is_blank(value):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def get_timezone(value: str) -> tzinfo:
    """Accept portable fixed offsets as well as IANA timezone names."""
    if value == "UTC":
        return timezone.utc
    match = re.fullmatch(r"UTC([+-])(\d{2}):(\d{2})", value)
    if match:
        hours, minutes = int(match[2]), int(match[3])
        if hours > 23 or minutes > 59:
            raise ValueError("UTC 时区偏移超出范围")
        offset = timedelta(hours=hours, minutes=minutes)
        return timezone(offset if match[1] == "+" else -offset)
    try:
        return ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"无法识别时区 {value!r}，可使用 UTC+08:00；IANA 时区需要 tzdata"
        ) from exc


def session_kind(umo: str) -> str:
    """Validate a full session ID so different bots cannot share a bare ID."""
    parts = umo.split(":", 2)
    if len(parts) != 3 or not parts[0].strip() or not parts[2].strip():
        raise ValueError("白名单需填写完整会话标识，可用 /早晚安会话 获取")
    kinds = {"GroupMessage": "group", "FriendMessage": "private"}
    if parts[1] not in kinds:
        raise ValueError("只支持群聊 GroupMessage 和私聊 FriendMessage")
    return kinds[parts[1]]


def read_minutes(value: Any, label: str, default: int = 30) -> int:
    if is_blank(value):
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 720:
        raise ValueError(f"{label} 必须是 0 至 720 的整数")
    return value


@dataclass(frozen=True)
class Greeting:
    kind: str
    label: str
    enabled: bool
    at: time
    before: int
    after: int
    prompt: str

    @classmethod
    def from_config(cls, kind: str, config: dict) -> Greeting:
        if not isinstance(config, dict):
            config = {}
        morning = kind == "morning"
        label = "早安" if morning else "晚安"
        default_time = DEFAULT_MORNING_TIME if morning else DEFAULT_NIGHT_TIME
        value = read_text(config.get("time"), default_time)
        if not re.fullmatch(r"\d{2}:\d{2}", value):
            raise ValueError(f"{label}时间需使用 HH:MM 格式，如 {default_time}")
        try:
            at = time.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{label}时间无效：{value}") from exc
        before = read_minutes(
            config.get("random_before_minutes"), label, DEFAULT_BEFORE_MINUTES
        )
        after = read_minutes(
            config.get("random_after_minutes"), label, DEFAULT_AFTER_MINUTES
        )
        if before + after >= 1440:
            raise ValueError(f"{label}随机窗口必须小于 24 小时")
        prompt_value = config.get("prompt")
        prompt = prompt_value.strip() if isinstance(prompt_value, str) else ""
        prompt = prompt or (DEFAULT_MORNING_PROMPT if morning else DEFAULT_NIGHT_PROMPT)
        return cls(
            kind,
            label,
            read_bool(config.get("enabled"), True),
            at,
            before,
            after,
            prompt,
        )

    def window(self, day: date, tz: tzinfo) -> tuple[float, float]:
        anchor = datetime.combine(day, self.at, tzinfo=tz)
        start = anchor - timedelta(minutes=self.before)
        # HH:MM is a whole minute. Zero offsets therefore mean that minute,
        # and polling cannot miss an exact second at HH:MM:00.
        end = anchor + timedelta(minutes=self.after + 1)
        return start.timestamp(), end.timestamp()

    def signature(self, timezone_name: str) -> str:
        value = f"{timezone_name}|{self.at}|{self.before}|{self.after}"
        return hashlib.sha256(value.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Settings:
    enabled: bool
    timezone_name: str
    tz: tzinfo
    provider_id: str
    timeout: int
    targets: frozenset[str]
    greetings: tuple[Greeting, ...]

    @classmethod
    def from_config(cls, config: dict) -> Settings:
        timezone_name = read_text(config.get("timezone"), DEFAULT_TIMEZONE)
        tz = get_timezone(timezone_name)
        targets = set()
        for kind in ("group", "private"):
            if not read_bool(config.get(f"{kind}_enabled"), True):
                # A disabled list is not used; stale entries should not block
                # greetings for the other chat types.
                continue
            values = config.get(f"{kind}_whitelist", [])
            if is_blank(values):
                values = []
            if not isinstance(values, list):
                raise ValueError(f"{kind}_whitelist 必须是列表")
            for value in values:
                if value is None:
                    continue
                if not isinstance(value, str):
                    raise ValueError(f"{kind}_whitelist 中包含非字符串条目")
                if not value.strip():
                    continue
                if session_kind(value.strip()) != kind:
                    raise ValueError(f"{kind}_whitelist 中包含类型不匹配的会话")
                targets.add(value.strip())
        timeout = config.get("generation_timeout_seconds")
        if is_blank(timeout):
            timeout = DEFAULT_GENERATION_TIMEOUT
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int)
            or not 5 <= timeout <= 300
        ):
            raise ValueError("模型超时时间必须是 5 至 300 的整数秒")
        provider_value = config.get("provider_id")
        provider_id = provider_value.strip() if isinstance(provider_value, str) else ""
        return cls(
            read_bool(config.get("enabled"), True),
            timezone_name,
            tz,
            provider_id,
            timeout,
            frozenset(targets),
            tuple(
                Greeting.from_config(kind, config.get(kind) or {})
                for kind in ("morning", "night")
            ),
        )


def plan_jobs(settings: Settings, jobs: dict[str, dict], now: datetime) -> bool:
    """Keep persisted draws; only unfinished jobs move when the window changes."""
    changed = False
    today = now.astimezone(settings.tz).date()
    cutoff = (today - timedelta(days=3)).isoformat()
    for key, job in list(jobs.items()):
        if job["day"] < cutoff:
            del jobs[key]
            changed = True
    if not settings.enabled:
        return changed
    timestamp = now.timestamp()
    for delta in (-1, 0, 1):
        day = today + timedelta(days=delta)
        for greeting in settings.greetings:
            if not greeting.enabled:
                continue
            start, end = greeting.window(day, settings.tz)
            if end <= timestamp:
                continue
            signature = greeting.signature(settings.timezone_name)
            for target in sorted(settings.targets):
                key = json.dumps(
                    [day.isoformat(), greeting.kind, target], ensure_ascii=False
                )
                old = jobs.get(key)
                if old and (
                    old["status"] != "pending" or old["signature"] == signature
                ):
                    continue
                earliest = max(start, timestamp)
                jobs[key] = {
                    "day": day.isoformat(),
                    "kind": greeting.kind,
                    "target": target,
                    "signature": signature,
                    "scheduled_at": RNG.uniform(earliest, max(earliest, end - 5)),
                    "expires_at": end,
                    "status": "pending",
                    "attempts": 0,
                    "retry_at": 0,
                }
                changed = True
    return changed
