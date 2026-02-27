"""
Setup script for LayeredRetriever.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="layered-retriever",
    version="1.0.0",
    author="LayeredRetriever Team",
    description="Multi-Stage Retrieval Assistant - 层级检索助手",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/layered-retriever",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "layered-retriever=api.app:main",
        ],
    },
)
