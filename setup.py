# coding=utf-8

from setuptools import setup, find_packages

setup(
    name="info_bpr",
    python_requires='>3.5.0',
    version="0.0.1",
    author="Jun Hu",
    author_email="hujunxianligong@gmail.com",
    packages=find_packages(
        exclude=[
            'benchmarks',
            'data',
            'demo',
            'dist',
            'doc',
            'docs',
            'logs',
            'models',
            'test',
            'demo_tensorflow_info_bpr.py',
            'demo_torch_info_bpr.py'
        ]
    ),
    install_requires=[
        "numpy >= 1.17.4",
        "tqdm"
    ],
    extras_require={
    },
    description="InfoBPR: Simple Yet Powerful Ranking Loss",
    license="GNU General Public License v3.0 (See LICENSE)",
    # long_description=open("README.rst", "r", encoding="utf-8").read(),
    long_description="InfoBPR: Simple Yet Powerful Ranking Loss",
    url="https://github.com/CrawlScript/InfoBPR"
)
