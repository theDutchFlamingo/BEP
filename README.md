# BEP
The repository in which all my python files etc. will be

## Code structure
The structure of the python code is as follows: the code for the generation of the noiseless clusters image (the heatmap) is in _NoiselessSubsystems.py_. The code for the generation of the image which contains information about the sizes of the noiseless clusters is in _NoiselessSizes.py_. Both of these have common functions in _common.py_.

The code for the moment simulations are in _MomentSims.py_ and _DensitySims.py_, both of which have many common functions in _moments_common.py_ for all the definitions of matrices, constants and the integration routine and _moments_plotter.py_ for the plotting of moments (duh). The density simulations also use _operators.py_ for the definition of operators and initial conditions.
