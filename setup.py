import os
import configparser
from setuptools import setup, find_packages


VERSION = '1.0.0'

setup(
    name='ASR project',
    version=VERSION,
    description='Automatic Speech Recognition System',
    author='ttaoREtw',
    author_email='ttaoREtw@gmail.com',
    packages=find_packages(),
    python_requires='>=3.5',
    install_requires=[
        'numpy',
        'scipy',
        'python-speech-features',
        'tensorflow'
    ]
)
