# setup.py
from setuptools import setup, find_packages

# Read requirements.txt and use it as install_requires
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name             = "PowerForecast",
    version          = "0.1.0",
    author           = "PowerForecast Team",
    description      = "Electricity price forecasting for European markets",
    packages         = find_packages(),
    python_requires  = ">=3.10",
    install_requires = requirements,
)
