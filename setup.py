import pytest
from setuptools import setup, find_packages

setup(
    name="retry-on",
    version="0.1.0",
    author="Juan Sugg",
    author_email="juanpedrosugg@gmail.com",
    description="Advanced Retry Mechanism Library for Python",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jsugg/retry-on-decorator",
    package_dir={"": "retry_on"},
    packages=find_packages(where="retry_on"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=open('requirements.txt').read().splitlines(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-asyncio'],
    test_suite='pytest_discover',
    cmdclass={'test': pytest},
)
