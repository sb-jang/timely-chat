import random
from typing import Callable, List


class BaseTemplate:
    @staticmethod
    def template_default(*args, **kwargs) -> Callable:
        """
        기본 템플릿 함수입니다.

        :param args: 템플릿 함수의 인자
        :param kwargs: 템플릿 함수의 인자
        :return: 템플릿 함수의 반환값
        """
        raise NotImplementedError

    @classmethod
    def all_templates(cls) -> List[Callable]:
        """
        모든 템플릿 함수를 반환합니다.

        :return: 템플릿 함수 리스트
        """
        return [
            func
            for name, func in cls.__dict__.items()
            if name.startswith("template") and isinstance(func, staticmethod)
        ]

    @classmethod
    def get_template(cls, determinism: bool = False, return_default: bool = False) -> Callable:
        """
        템플릿 함수 중 하나를 반환합니다.

        :param determinism: 템플릿 함수 내부에 랜덤 함수를 사용할 때, 동일한 결과를 반환하도록 합니다. 템플릿은 랜덤하게 선택됩니다.
        :param return_default: 템플릿 함수를 랜덤하게 선택하지 않고, 기본 템플릿 함수(`template_default`)를 반환합니다.
        :return: 템플릿 함수
        """
        template = random.choice(cls.all_templates())
        if return_default:
            template = cls.__dict__["template_default"]

        if determinism:
            seed = random.random()

            def fixed_template(*args, **kwargs):
                random.seed(seed)
                return template.__func__(*args, **kwargs)

            return fixed_template
        return template.__func__
