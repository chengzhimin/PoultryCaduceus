#!/usr/bin/env python
"""
PoultryCaduceus: A Bidirectional DNA Language Model for Chicken Genome
"""

from setuptools import setup, find_packages
import os

# Read version from package
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'poultry_caduceus', '__version__.py')
    with open(version_file) as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

# Read long description from README
def get_long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def get_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='poultry-caduceus',
    version=get_version(),
    author='Your Name',
    author_email='your.email@example.com',
    description='A Bidirectional DNA Language Model for Chicken Genome',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/YOUR_USERNAME/PoultryCaduceus',
    project_urls={
        'Bug Tracker': 'https://github.com/YOUR_USERNAME/PoultryCaduceus/issues',
        'Documentation': 'https://github.com/YOUR_USERNAME/PoultryCaduceus#documentation',
        'Source Code': 'https://github.com/YOUR_USERNAME/PoultryCaduceus',
    },
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'scripts']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    python_requires='>=3.9',
    install_requires=get_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=23.0',
            'isort>=5.12',
            'flake8>=6.0',
            'mypy>=1.0',
        ],
        'docs': [
            'sphinx>=6.0',
            'sphinx-rtd-theme>=1.2',
            'myst-parser>=1.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'poultry-caduceus=poultry_caduceus.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'poultry_caduceus': ['configs/*.yaml'],
    },
    keywords=[
        'deep learning',
        'genomics',
        'DNA language model',
        'chicken genome',
        'poultry',
        'bioinformatics',
        'MPRA',
        'regulatory genomics',
    ],
)
