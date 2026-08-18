# setup.py
from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    required = f.read().splitlines()

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="envisionhgdetector",
    version="3.0.8",
    author="Wim Pouw, Sharjeel Ahmed Shaikh, James Trujillo, Bosco Yung,  Antonio Rueda-Toicen, Gerard de Melo, Babajide Owoyele",
    author_email="w.pouw@tilburguniversity.edu",
    description="Hand gesture detection using MediaPipe and CNN, kinematic analysis, and visualization.",
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
    python_requires=">=3.10",
    install_requires=required,
    include_package_data=True,
    package_data={
        'envisionhgdetector': ['model/*.h5', 'model/*.pkl'],
    },
)