from setuptools import find_packages, setup

setup(
    name="timely-chat-research-repository",
    version="0.0.1",
    description="timely-chat research repository",
    install_requires=[],
    url="https://github.com/sb-jang/timely-chat-research.git",
    author="Seongbo Jang",
    author_email="jang.sb@postech.ac.kr",
    packages=find_packages(exclude=["tests"]),
)
