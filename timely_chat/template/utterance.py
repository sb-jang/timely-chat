from datetime import datetime
from typing import Callable, Optional

from .base import BaseTemplate


class UtteranceTemplate(BaseTemplate):
    @staticmethod
    def template_default(
        utterance: str, speaker: str, timestamp: datetime, timestamp_template: Optional[Callable]
    ) -> str:
        if timestamp_template is not None:
            timestamp = timestamp_template(timestamp)
            return f"{timestamp} {speaker}: {utterance}"
        else:
            return f"{speaker}: {utterance}"

    @staticmethod
    def template_1(utterance: str, speaker: str, timestamp: datetime, timestamp_template: Optional[Callable]) -> str:
        if timestamp_template is not None:
            timestamp = timestamp_template(timestamp)
            return f"[{timestamp}] {speaker}: {utterance}"
        else:
            return f"{speaker}: {utterance}"

    @staticmethod
    def template_2(utterance: str, speaker: str, timestamp: datetime, timestamp_template: Optional[Callable]) -> str:
        if timestamp_template is not None:
            timestamp = timestamp_template(timestamp)
            return f"{speaker}\t{timestamp}\t{utterance}"
        else:
            return f"{speaker}\t{utterance}"

    @staticmethod
    def template_3(utterance: str, speaker: str, timestamp: datetime, timestamp_template: Optional[Callable]) -> str:
        if timestamp_template is not None:
            timestamp = timestamp_template(timestamp)
            return f"{speaker}\t{utterance}\t{timestamp}"
        else:
            return f"{speaker}\t{utterance}"

    @staticmethod
    def template_4(utterance: str, speaker: str, timestamp: datetime, timestamp_template: Optional[Callable]) -> str:
        if timestamp_template is not None:
            timestamp = timestamp_template(timestamp)
            return f'{speaker}: "{utterance}" ({timestamp})'
        else:
            return f'{speaker}: "{utterance}"'
