"""
Setup configuration for Robot Test Bench.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="robot_testbench",
    version="0.1.0",
    author="Eric Rosenfeld",
    author_email="ericalanrosenfeld@gmail.com",
    description="A comprehensive testing framework for robotic systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/earosenfeld/robot-test-bench",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "pyyaml>=5.4.0",
        "simple-pid>=2.0.0",
        "pytest>=6.2.0",
        "pytest-cov>=2.12.0",
    ],
    extras_require={
        "dev": [
            "black>=21.5b2",
            "isort>=5.9.0",
            "mypy>=0.910",
            "pylint>=2.9.0",
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "dashboard": [
            "dash>=2.0.0",
            "plotly>=5.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "robot-testbench=robot_testbench.main:main",
        ],
    },
) 