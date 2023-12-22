from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path):
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = 'baseline_optimal',
    version = '0.0.4',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author = 'Shunji Wan',
    author_email = 'sw3843@columbia.edu',
    url='https://github.com/sjwan01/baseline_optimal',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)