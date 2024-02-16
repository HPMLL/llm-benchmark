import setuptools
import os
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

_deps = [
'bitsandbytes',
'datasets',
'deepspeed',
'evaluate',
'InquirerPy',
'numpy',
'nvidia_ml_py',
'packaging',
'pandas',
'peft',
'pynvml',
'torch==2.0.1',
'tqdm',
'transformers',
'SentencePiece',
'protobuf',
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}

install_requires = [
    deps["bitsandbytes"],
    deps["datasets"],
    deps["deepspeed"],
    deps["evaluate"],
    deps["InquirerPy"],
    deps["numpy"],
    deps["nvidia_ml_py"],
    deps["packaging"],
    deps["peft"],
    deps["pynvml"],
    deps["torch"],
    deps["tqdm"],
    deps["transformers"],
    deps["pandas"],
    deps["SentencePiece"],
    deps["protobuf"],
]

setuptools.setup(
    name="llmbenchmark",
    version="0.0.1",
    author="HPMLL",
    author_email="aaron_zlt@outlook.com",
    description="This is the package for benchmarking Large Language Models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AaronZLT/llm-benchmark",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6, <3.11',
)