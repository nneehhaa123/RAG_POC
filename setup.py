"""Setup python package"""
import pathlib

import setuptools

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setuptools.setup(
    name="dpp_helpline_qa",
    version="0.0.1",
    description="""A Question and Answer model to help the DPP Helpline team answer
                   low-complexity and routine internal audit queries.""",  # noqa
    long_description=README,
    packages=[
        "dpp_helpline_qa",
        "dpp_helpline_qa.model_validation",
        "dpp_helpline_qa.modelling",
        "dpp_helpline_qa.preprocessing",
    ],
    author="KPMGUK",
    author_email="many@kpmg.co.uk",
    long_description_content_type="text/markdown",
    url="https://github.com/KPMG-UK/dpp_helpline_qa",
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
