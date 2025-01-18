from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='proteinttt',
    author='ProteinTTT developers',
    license="MIT",
    version='0.0.1',
    packages=find_packages(),
    install_requires=required,
    include_package_data=True
)
