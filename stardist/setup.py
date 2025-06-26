from setuptools import setup, find_packages

exec(open('runstardist/__version__.py', encoding='utf-8').read())  # pylint: disable=exec-used

setup(
    name='run-stardist',
    version=__version__,  # pylint: disable=undefined-variable
    author='Qin Yu',
    author_email='qin.yu@embl.de',
    license='MIT',
    description='Train and use StarDist models',
    url='https://github.com/kreshuklab/go-nuclear',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'train-stardist=runstardist.train:main',
            'predict-stardist=runstardist.predict:main',
        ],
    },
)
