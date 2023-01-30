import codecs
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

LICENSE = 'BSD 3-clause'
VERSION = '0.1.0'
DESCRIPTION = 'Taycho Crater Detection Model (CDM)'
LONG_DESCRIPTION = \
    'A package that contains code to run the crater detection model of group Taycho from the commandline.'

# Setting up
setup(
    name="taycho_cdm",
    version=VERSION,
    author="Group Taycho",
    author_email="<acds.taycho@gmail.com>",
    license=LICENSE,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['setuptools'],  # TODO
    keywords=['python', 'yolo', 'object detection', 'crater detection', 'mars crater detection',
              'moon crater detection', 'computer vision'],
    classifiers=[
        "Development Status :: 2 - Production",  # TODO
        "Intended Audience :: ESE Students",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
