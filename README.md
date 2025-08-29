Software Engineering - Fall 2025 Project on Deep Vision Models for Pediatric Pneumonia Identification
- [Pytorch](#pytorch)
- [Setting up python](#setting-up-python)
  - [Python venv](#python-venv)
    - [Create a venv](#create-a-venv)
  - [Python venv activation](#python-venv-activation)
      - [Windows venv activation](#windows-venv-activation)
      - [Mac or linux venv activation](#mac-or-linux-venv-activation)
  - [Install dependencies](#install-dependencies)
  - [Downloading libraries](#downloading-libraries)
    - [Updating requirements.txt](#updating-requirementstxt)

# Pytorch
This project uses pytorch as the main library for training and building the AI. 

Under the **learning** folder you can find a series of files that detail various aspects of pytorch. 

The biggest and most comprehensive file is `optimization_learning.py`. This file goes over the in depth process of building a model, training it, and the various parameters that go into that. 

I still recommend looking at the other files, especially if you've never built an AI before as there is critical information expaining tensors, data, and other fundamentals. 

All of these learning files are following along [this pytorch tutorial](https://docs.pytorch.org/tutorials/beginner/basics/intro.html). There are even more resources featured on this link than I've put into the learning folder. 

For some extra information about the process of iterative learning and backpropagation, [this video](https://www.youtube.com/watch?v=tIeHLnjs5U8) from 3Blue1Brown was recommended by the pytorch tutorial and I highly recommend it as well. 

# Setting up python
Hopefully you'll have python3 downloaded and access to the console commands, if not, visit [here](https://www.geeksforgeeks.org/python/download-and-install-python-3-latest-version/).

## Python venv
Python is all about libraries and we need somewhere to download those libraries. Not system wide, because that could completely break all of our libaries.

This is what the python virtual environment is for. Separate project require separate virtual environments, so let's set up one for this project. 


### Create a venv
In your terminal, navigate to this project folder. If you're using VS Code, you can click Terminal > New Terminal, and a terminal will pull up, already in the correct directory. 
```
python3 -m venv venv
```
This will create a new folder called venv. The gitignore for this project already includes python so whenever you push your changes, this directory won't be included. We don't want the venv included in the github as everyone's will be different.

Some windows machines use `python` instead of `python3` but I can't test that as I don't have a windows machine. If the above command doesn't work, try:
```
python -m venv venv
```

## Python venv activation
After creating your venv, you'll want to make sure your machine knows you're running python stuff from it. 

#### Windows venv activation
```
# In cmd.exe
venv\Scripts\activate.bat
# In PowerShell
venv\Scripts\Activate.ps1
```
#### Mac or linux venv activation 
```
source venv/bin/activate
```
If you'd like to learn more about the python venv, you can look [here](https://python.land/virtual-environments/virtualenv). 

## Install dependencies 
```
pip3 install -r requirements.txt
```
Similar thing here as `python3`, if `pip3` doesn't work then try using `pip`. 

## Downloading libraries
If for any reason you need to download more libaries than are here, you'll usually follow a similar pattern as above and download them with 
```
pip3 install libraryname
```

### Updating requirements.txt
After downloading a new libary, everyone who downloads the project needs to also be able to download that libary. 

You can update the requirements.txt with
```
pip3 freeze > requirements.txt
```
Again, on windows, I believe it's `pip` instead of `pip3`. 

If you've returned to the project and found that there's more libraries than you currently have, you can run `pip3 install -r requirements.txt` again to update your dependencies. 