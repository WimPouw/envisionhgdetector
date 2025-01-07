# setup.py

from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt') as f:
    required = f.read().splitlines()

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="envisionhgdetector",
    version="0.0.4.5",
    author="Wim Pouw",
    author_email="wim.pouw@donders.ru.nl",
    description="Hand gesture detection using MediaPipe and CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wimpouw/envisionhgdetector",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.9",
    install_requires=required,
    include_package_data=True,
    package_data={
        'envisionhgdetector': ['model/*.h5'],
    },
)