# UR10 Palletizing Environment

1. [Environment Installation](#environment-installation)
2. [VSCode Setting](#vscode-setting)
3. [Running Code](#running-code)

### Environment installation
Follow the orbit installation process [here](https://isaac-orbit.github.io/orbit/source/setup/installation.html)
The orbit conda environment should be activated in order to run this code.

### VSCode Setting
run the following command in the terminal
```
export workspaceFolder=${CURRENT_WORKSPACE_FOLDER}
ln -s ISAACSIM_PATH _isaac_sim
export CARB_APP_PATH=${workspaceFolder}/_isaac_sim/kit
export ISAAC_PATH=${workspaceFolder}/_isaac_sim
export EXP_PATH=${workspaceFolder}/_isaac_sim/apps
source ${workspaceFolder}/_isaac_sim/kit/setup_python_env.sh
printenv > .vscode/.python.env
python ${workspaceFolder}/.vscode/tools/setup_vscode.py
```

### running code
```
conda activate orbit
python test_palload.py --model_path [MODEL_PATH]
```

### TroubleShooting
Here is a list of common problems you might encounter.

* Assets from the Isaac Nucleus are not loaded
  *  open the web browser and type localhost:3080. Restart everything that was stopped or paused.
