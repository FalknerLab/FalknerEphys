# FalknerEphys
Analysis code for chronic silicon probe recordings performed in Falkner Lab

## Installing as Developer
1. Open your IDE of choice and Git clone via version control or with terminal (below)

`git clone https://github.com/FalknerLab/FalknerEphys.git`

Note: requires login info for FalknerLab GitHub

2. In IDE terminal build environment with environment.yml file

`conda env create --file environment.yml`

3. In IDE, set project interpreter to FalknerEphys

## Installing as a Package
1. In Anaconda/Miniconda terminal activate and/or create an evironment where you want to install the package

`conda env create --name MyEphysAnalysis`<br>
`conda activate MyEphysAnalysis`

3. Get pip

`conda install pip`

4. Install FalknerEphys package via pip

`pip install git+https://{token}@github.com/FalknerLab/FalknerEphys.git`<br>
Where {token} is generated from FalknerLab GitHub periodically and moved to the Cup in Dave/FL_token.txt

5. Try importing in an IDE and running a demo

`from falknerephys.demos import wm_import_demo`<br>
`wm_import_demo()`


## Running Demos
[Demo scripts](falknerephys/demos/) show processing steps for the different Falkner Lab data streams

Examples include:<br>
-White Matter wireless recordings in the territory rig
