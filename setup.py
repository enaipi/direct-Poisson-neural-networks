"""Setup configuration for the Direct Poisson Neural Networks package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dpnn",
    version="1.0.0",
    author="Michal Šípka",
    description="Direct Poisson Neural Networks - Learning non-symplectic mechanical systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.karlin.mff.cuni.cz/~pavelka/direct-poisson-neural-networks/",
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.24.0",
        "torchmetrics>=0.8.0",
        "matplotlib>=3.3.0",
    ],
)
