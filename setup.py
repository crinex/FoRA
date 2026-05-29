"""Setup script for peft-fora."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="peft-fora",
    version="0.1.0",
    description="FoRA: Fisher-orthogonal Rank Adaptation for HuggingFace PEFT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="jyoung.park",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=5.0",
            "ruff>=0.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["peft", "lora", "fora", "llm", "fine-tuning", "stiefel", "fisher"],
)
