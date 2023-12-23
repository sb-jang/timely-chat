import random
from datetime import datetime

from .base import BaseTemplate

WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


class DateTemplate(BaseTemplate):
    @staticmethod
    def template_default(dt: datetime) -> str:
        return random.choice(
            [
                # 2023년 1월 2일 월요일
                f"{dt.strftime('%Y년 %-m월 %-d일')} {WEEKDAYS[dt.weekday()]}요일",
                # 1월 2일 월요일
                f"{dt.strftime('%-m월 %-d일')} {WEEKDAYS[dt.weekday()]}요일",
            ]
        )

    @staticmethod
    def template_1(dt: datetime) -> str:
        return random.choice(
            [
                # 2023년 1월 2일 월요일에 나눈 대화입니다.
                f"{dt.strftime('%Y년 %-m월 %-d일')} {WEEKDAYS[dt.weekday()]}요일에 나눈 대화입니다.",
                # 1월 2일 월요일에 나눈 대화입니다.
                f"{dt.strftime('%-m월 %-d일')} {WEEKDAYS[dt.weekday()]}요일에 나눈 대화입니다.",
            ]
        )

    @staticmethod
    def template_2(dt: datetime) -> str:
        return random.choice(
            [
                # 2023년 1월 2일 월요일의 대화
                f"{dt.strftime('%Y년 %-m월 %-d일')} {WEEKDAYS[dt.weekday()]}요일의 대화",
                # 1월 2일 월요일의 대화
                f"{dt.strftime('%-m월 %-d일')} {WEEKDAYS[dt.weekday()]}요일의 대화",
            ]
        )

    @staticmethod
    def template_3(dt: datetime) -> str:
        return random.choice(
            [
                # 2023-01-02
                dt.strftime("%Y-%m-%d"),
                # 2023-1-2
                dt.strftime("%Y-%-m-%-d"),
                # 2023/01/02
                dt.strftime("%Y/%m/%d"),
                # 2023/1/2
                dt.strftime("%Y/%-m/%-d"),
                # 2023.01.02
                dt.strftime("%Y.%m.%d"),
                # 2023.1.2
                dt.strftime("%Y.%-m.%-d"),
                # 20230102
                dt.strftime("%Y%m%d"),
                # 230102
                dt.strftime("%y%m%d"),
                # Jan 02, 2023
                dt.strftime("%b %d, %Y"),
                # January 02, 2023
                dt.strftime("%B %d, %Y"),
                # 02/Jan/2023
                dt.strftime("%d/%b/%Y"),
            ]
        )
