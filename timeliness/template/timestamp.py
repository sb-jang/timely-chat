import random
from datetime import datetime

from .base import BaseTemplate

WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


class TimestampTemplate(BaseTemplate):
    @staticmethod
    def template_default(dt: datetime) -> str:
        ampm = "오전" if dt.hour < 12 else "오후"
        return random.choice(
            [
                # 2023년 1월 2일 월요일 3시 4분
                f"{dt.strftime('%Y년 %-m월 %-d일')} {WEEKDAYS[dt.weekday()]}요일 {dt.strftime('%-H시 %-M분')}",
                # 2023년 1월 2일 월요일 3시 4분 AM
                f"{dt.strftime('%Y년 %-m월 %-d일')} {WEEKDAYS[dt.weekday()]}요일 {dt.strftime('%-I시 %-M분 %p')}",
                # 2023년 1월 2일 월요일 오전 3시 4분
                f"{dt.strftime('%Y년 %-m월 %-d일')} {WEEKDAYS[dt.weekday()]}요일 {ampm} {dt.strftime('%-I시 %-M분')}",
            ]
        )

    @staticmethod
    def template_1(dt: datetime) -> str:
        ampm = "오전" if dt.hour < 12 else "오후"
        return random.choice(
            [
                # 2023-01-02 03:04
                dt.strftime("%Y-%m-%d %H:%M"),
                # 2023-01-02 03:04 AM
                dt.strftime("%Y-%m-%d %I:%M %p"),
                # 2023-01-02 오전 03:04
                f"{dt.strftime('%Y-%m-%d')} {ampm} {dt.strftime('%I:%M')}",
            ]
        )

    @staticmethod
    def template_2(dt: datetime) -> str:
        ampm = "오전" if dt.hour < 12 else "오후"
        return random.choice(
            [
                # 2023-1-2 03:04
                dt.strftime("%Y-%-m-%-d %H:%M"),
                # 2023-1-2 03:04 AM
                dt.strftime("%Y-%-m-%-d %I:%M %p"),
                # 2023-1-2 오전 03:04
                f"{dt.strftime('%Y-%-m-%-d')} {ampm} {dt.strftime('%I:%M')}",
            ]
        )

    @staticmethod
    def template_3(dt: datetime) -> str:
        ampm = "오전" if dt.hour < 12 else "오후"
        return random.choice(
            [
                # 2023/01/02 03:04
                dt.strftime("%Y/%m/%d %H:%M"),
                # 2023/01/02 03:04 AM
                dt.strftime("%Y/%m/%d %I:%M %p"),
                # 2023/01/02 오전 03:04
                f"{dt.strftime('%Y/%m/%d')} {ampm} {dt.strftime('%I:%M')}",
            ]
        )

    @staticmethod
    def template_4(dt: datetime) -> str:
        ampm = "오전" if dt.hour < 12 else "오후"
        return random.choice(
            [
                # 2023/1/2 03:04
                dt.strftime("%Y/%-m/%-d %H:%M"),
                # 2023/1/2 03:04 AM
                dt.strftime("%Y/%-m/%-d %I:%M %p"),
                # 2023/1/2 오전 03:04
                f"{dt.strftime('%Y/%-m/%-d')} {ampm} {dt.strftime('%I:%M')}",
            ]
        )

    @staticmethod
    def template_time_only_1(dt: datetime) -> str:
        # 03:04
        return dt.strftime("%H:%M")

    @staticmethod
    def template_time_only_2(dt: datetime) -> str:
        # 03:04 AM
        return dt.strftime("%I:%M %p")

    @staticmethod
    def template_time_only_3(dt: datetime) -> str:
        # 오전 03:04
        return f"오전 {dt.strftime('%I:%M')}" if dt.hour < 12 else f"오후 {dt.strftime('%I:%M')}"
