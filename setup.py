from setuptools import find_packages, setup

setup(
    name="timeliness-research-repository",
    version="0.0.1",
    description="timeliness research repository",
    install_requires=[],
    url="https://github.com/sb-jang/timeliness-research.git",
    author="Seongbo Jang",
    author_email="jang.sb@postech.ac.kr",
    packages=find_packages(exclude=["tests"]),
)
