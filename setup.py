from setuptools import setup, find_packages

setup(
    name="findr",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click",
        "numpy",
        "mygrad",
        "pathlib",
        "pillow",
        "matplotlib",
        "mynn",
        "gensim",
        "setuptools",
    ],
    entry_points={
        # Add all your commands and groups here from the script
        "console_scripts": ["findr=cli_script:cli"]
    },
)
