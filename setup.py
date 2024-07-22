import os
from setuptools import setup, find_packages


def read_version():
    version_dict = {}
    with open(os.path.join("vsl_utils", "version.py")) as f:
        exec(f.read(), version_dict)
    return version_dict["__version__"]


setup(
    name="vsl-utils",
    version="0.1.0",  # Dynamically read the version
    author="IAS - Vessel Dev Team",
    author_email="ias@pfizer.com",
    description="A package containing all reusable vessel utilities.",
    long_description_content_type="text/markdown",
    packages=find_packages(where="app"),
    install_requires=["httpx", "cachetools", "boto3", "python-dateutil", "fastapi"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)
