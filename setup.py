"""Setup python package"""
import pathlib

import setuptools

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setuptools.setup(
    name="rag_qa",
    version="0.0.1",
    description="""A Question and Answer model to find relevant context and generate answer for given queries.""",  # noqa
    long_description=README,
    packages=[
        "rag_qa",
        "rag_qa.model_validation",
        "rag_qa.modelling",
        "rag_qa.preprocessing",
    ],
    author="neha gupta",
    author_email="nehagupta.28jan@gmail.com",
    long_description_content_type="text/markdown",
    url="https://github.com/nneehhaa123/RAG_POC",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.8",
    install_requires=[],
)
