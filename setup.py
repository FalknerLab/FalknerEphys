from setuptools import find_packages, setup

with open("requirements.txt") as f:
    reqs = f.readlines()

setup(
    name="falknerephys",
    version="0.0.1",
    description="Analyses code for Falkner Lab silicon probe recordings",
    url="https://github.com/FalknerLab/FalknerEphys.git",
    author="David Allen, Nancy Mack, Bartul Mimica",
    author_email="falknermice@gmail.com",
    packages=['falknerephys', 'falknerephys.io', 'falknerephys.demos'],
    python_requires=">=3.8",
    install_requires=reqs,
    license_files=("LICENCE",),
    license="BSD-3 Licence")
