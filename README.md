# Selecting-the-Optimal-Model-for-Binary-Classification-of-Mobile-Home-Policies

# Python Project Setup Guide

This guide provides step-by-step instructions for setting up a virtual environment and installing the necessary Python libraries for your project on Unix-like operating systems (Linux or macOS) or Windows.

## Step 1: Install Python

Ensure that Python is installed on your system. If it is not installed, you can download it from [python.org](https://python.org).

## Step 2: Create a Virtual Environment

Open your terminal or command prompt and navigate to the project directory where you want to set up the virtual environment.

```bash
# Navigate to your project directory
cd path/to/your/project
```

# For Unix-like Systems

```bash
# Navigate to your project directory
python3 -m venv env
```
# For Windows


```bash
# Navigate to your project directory
python -m venv env
```
*Or*
```bash
# Navigate to your project directory
py -m venv env
```
# Step 3: Activate the Virtual Environment
Before installing libraries, you need to activate the virtual environment.

# For Unix-like Systems

```bash
# Navigate to your project directory
source env/bin/activate
```
# For Windows


```bash
# Navigate to your project directory
.\env\Scripts\activate
```
#Step 4: Install Required Libraries

```bash
# Navigate to your project directory
pip install pandas joblib scikit-learn
```

```bash
# Navigate to your project directory
pip install torch torchvision torchaudio pytorch-tabnet
```