# Setup for blechpy development

## Installing required pacakges

First create a virtual environment using whichever method you choose. I prefer using Miniconda and python 3.7. Later versions of python 3 should work fine. 

```bash
conda create -n dev python=3.7
conda activate dev
conda install ipython
pip install --user --upgrade setuptools wheel twine pdoc3
```

`setuptools` and `wheel` are for building the project. 
`pdoc3` is used to create the Documentation html from docstrings in the code.
`twine` is used to upload to PyPi.

## Create an environment for testing changes
This is a separate conda environment where you install blechpy in editable mode so you can test your changes before delpoying. 

First close the (blechpy)[https://github.com/nubs01/blechpy] repo from github.
```bash
conda create -n blech-dev python=3.7
conda activate blech-dev
conda install ipython
cd /path/to/blechpy # This is the parent directory that contains the setup.py file
pip install -e .
```

Now when this environment is activate, the version of blechpy being used is not the package from PyPi but instead the raw code on your machine. 

## Create a PyPi account

- Go to pypi.org
- Create an account
- Get a blechpy admin to add your username as a *Maintainer* on the blechpy package

## Making Changes

Now you can clone the blechpy github repository and begin editting and improving. 

Best practice is to: 
- create a new branch 
- edit that branch
- test
- merge back to master

When you make edits,
- Make new functions as small, modular and general as possible
- Use docstring style already used in blechpy package 
 - blechpy dosumentation is made using pdoc which automatically generates the documentation webpage from the docstrings
- Try not to break anything
- Avoid spaghetti code: don't import modules into other modules that depend on the imported module, i.e. Don't import `blechpy.datastructures.dataset` into `blechpy.dio.h5io`

## Building and deploying to pypi

- Update the CHANGES.txt file to include a descriptions of the changes you have made
- Update version number in setup.py
- Activate dev environment: `conda activate dev` 
- Build: `python setup.py sdist bdist_wheel`
- Deploy: `twine upload --verbose --skip-existing dist/*`
 - enter you username and password for pypi
