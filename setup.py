import setuptools
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), 'r') as fh:
    long_description = fh.read()

requirementPath = os.path.join(here, 'requirements.txt')
install_requires = [] # Examples: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name='blechpy',
    version='2.0.65',
    author='Roshan Nanu',
    author_email='roshan.nanu@gmail.com',
    description='Package for exrtacting, processing and analyzing Intan and OpenEphys data',
    long_description_content_type='text/markdown',
    long_description=long_description,
    url='https://github.com/nubs01/blechpy',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License'],
    keywords='blech katz_lab Intan electrophysiology neuroscience',
    python_requires='>=3.6',
    install_requires=install_requires,
    include_package_data=True,
    package_data={'': ['environment.yml'], 'dio': ['defaults/*.json']}
)




