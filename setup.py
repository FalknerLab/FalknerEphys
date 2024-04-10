from setuptools import find_namespace_packages, setup

with open("requirements.txt") as f:
    reqs = f.readlines()

setup(
    name="falknerephys",
    version="0.0.1",
    description="Analyses code for Falkner Lab silicon probe recordings",
    url="https://github.com/FalknerLab/FalknerEphys.git",
    author="David Allen, Nancy Mack, Bartul Mimica",
    author_email="falknermice@gmail.com",
    packages=find_namespace_packages(),
    package_dir={'wm': 'falknerephys/demos/wm'},
    package_data={'wm': ['phy/*.npy', 'wm_demo_DAQ.h5', 'wm_demo_SLEAP.h5']},
    python_requires=">=3.8",
    install_requires=reqs,
    license_files=("LICENCE",),
    license="BSD-3 Licence")
