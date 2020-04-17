# -*- coding: utf-8 -*-py
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='cam_readability_assessment',
    version='1.0',
    packages=find_packages(),
    url='',
    license='',
    author='ines_blin',
    author_email='ines.blin@student.ecp',
    description='',
    #install_requires=requirements,
)