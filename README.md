# FalknerEphys
Analysis code for chronic silicon probe recordings performed in Falkner Lab

**Note**: requires login info for FalknerLab GitHub (Login to Github on your IDE)

# Installing as Developer
1. Open your IDE of choice and Git clone via version control or with terminal (below)

`git clone https://github.com/FalknerLab/FalknerEphys.git`

2. In IDE terminal build environment with environment.yml file

`conda env create --file environment.yml`

3. In IDE, set project interpreter to FalknerEphys

# Installing as a Package
1. In Anaconda/Miniconda terminal activate and/or create an evironment where you want to install the package

`conda env create --name MyEphysAnalysis pip -y`<br>
`conda activate MyEphysAnalysis`

2. Install FalknerEphys package via pip

`pip install git+https://github.com/FalknerLab/FalknerEphys.git`<br>

5. Try importing in an IDE

`import falknerephys as fe`

6. Running command line functions

## _Bombcell_

`falknerephys -bombcell "your_path_to_imec_ap.bin" "your_path_to_imec_ap.META" "your_path_to_kilosort_folder"`

This will run BC curration on the given folder and save output there

## _Kilosort_

**NOTE: If you want to use GPU, you must run the following in your environment**

`pip uninstall torch`

`pip3 install torch --index-url https://download.pytorch.org/whl/cu118`

Run kilosort4 on a given .bin and channel map using default settings

`falknerephys -kilosort "your_path_to_imec_ap.bin" "your_path_to_channel_map.json"`

-or-

`falknerephys -kilosort`

Will trigger a dialog window to select one or many ap.bin files to batch process. Second dialog to specify channelmap.json file

## _Brainreg_

`falknerephys -brainreg "your_path_to_histology.tiff" "your_path_to_channel_map.json"`

Will register a given tiff stack (from PNI light sheet, CM-DiI channel) to Allen CCFv3 and apply the given channel map to automatically reconstruct shank trajectories and map NPX channel #s to 3D coordinates in Allen space.


## Running Demos (WIP)
[Demo scripts](falknerephys/demos/) show processing steps for the different Falkner Lab data streams

Examples include:<br>

-White Matter wireless recordings in home cage
`from falknerephys.demos import wm_import_demo`<br>
`wm_import_demo()`

-Nothing else for now

