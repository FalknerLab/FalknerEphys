from setuptools import find_namespace_packages, setup

with open("requirements.txt") as f:
    reqs = f.readlines()

def list_data():
    data_files = ['falknerephys/demos/wm/phy/*.npy',
                  'falknerephys/demos/wm/wm_demo_DAQ.h5',
                  'falknerephys/demos/wm/wm_demo_SLEAP.h5']
    return data_files


# setup(
#     name="falknerephys",
#     version="0.0.1",
#     description="Analyses code for Falkner Lab silicon probe recordings",
#     url="https://github.com/FalknerLab/FalknerEphys.git",
#     author="David Allen, Nancy Mack, Bartul Mimica",
#     author_email="falknermice@gmail.com",
#     packages=find_namespace_packages(),
#     package_data={'falknerephys': list_data()},
#     python_requires=">=3.8",
#     install_requires=reqs,
#     license_files=("LICENCE",),
#     license="BSD-3 Licence")

setup(
    name="falknerephys",
    version="0.0.1",
    description="Analyses code for Falkner Lab silicon probe recordings",
    url="https://github.com/FalknerLab/FalknerEphys.git",
    author="David Allen, Nancy Mack, Bartul Mimica",
    author_email="falknermice@gmail.com",
    packages=find_namespace_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=reqs,
    license_files=("LICENCE",),
    license="BSD-3 Licence")
