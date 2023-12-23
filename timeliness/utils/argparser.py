from argparse import ArgumentParser
from dataclasses import field as data_field
from dataclasses import fields
from typing import Any, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


def group(title: str, description: Optional[str] = None):
    """
    Dataclass를 ArgumentParser의 그룹으로 지정합니다.

    :param title: 그룹명
    :param description: 그룹 설명
    """

    def wrapper(cls):
        cls.__arg_group_info__ = {"title": title, "description": description}
        return cls

    return wrapper


def field(default: Optional[Any] = None, required: bool = False, help: Optional[str] = None):
    """
    Dataclass내 속성을 ArgumentParser의 각 옵션들로 사용합니다.

    :param default: 기본값
    :param required: required 여부
    :param help: 옵션 설명
    """
    return data_field(default=default, metadata={"help": help, "required": required})


def build_parser(parser: ArgumentParser, classes: Tuple):
    """
    정의된 parser와 Dataclass를 사용하여 parser를 빌드합니다.

    :param parser: ArgumentParser
    :param classes: 인자로 사용할 Dataclass 목록
    """
    for cls in classes:
        controller = parser

        if cls.__arg_group_info__:
            title = cls.__arg_group_info__["title"]
            description = cls.__arg_group_info__["description"]
            controller = parser.add_argument_group(title=title, description=description)

        args = fields(cls)

        for arg in args:
            arg_name = "--" + arg.name.replace("_", "-")
            arg_type = arg.type

            # Optional 타입 지원
            if hasattr(arg_type, "__args__"):
                arg_type = arg_type.__args__[0]

            if arg.type == bool:
                controller.add_argument(arg_name, action="store_true", help=arg.metadata["help"])
            else:
                if arg.metadata["required"]:
                    controller.add_argument(arg_name, type=arg_type, help=arg.metadata["help"], required=True)
                else:
                    controller.add_argument(arg_name, default=arg.default, type=arg_type, help=arg.metadata["help"])

    return parser


def parse_args(parser: ArgumentParser, cls: Type[T]) -> T:
    """
    프로그램 인자로부터 데이터클래스로 변환합니다.

    :param parser: ArgumentParser
    :param cls: 변환할 Dataclass
    """
    parsed = parser.parse_args()

    args = fields(cls)
    kwargs = {arg.name: getattr(parsed, arg.name) for arg in args}

    return cls(**kwargs)
