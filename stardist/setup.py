from pathlib import Path

from setuptools import find_packages, setup


exec(open("runstardist/__version__.py", encoding="utf-8").read())

setup(
    name="run-stardist",
    version=__version__,
    author="Qin Yu",
    author_email="qin.yu@embl.de",
    license="MIT",
    description="Train and use StarDist models",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/kreshuklab/go-nuclear",
    project_urls={
        "Documentation": "https://kreshuklab.github.io/go-nuclear/",
        "Source": "https://github.com/kreshuklab/go-nuclear",
        "Bug Tracker": "https://github.com/kreshuklab/go-nuclear/issues",
    },
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train-stardist=runstardist.train:main",
            "predict-stardist=runstardist.predict:main",
        ],
    },
)
