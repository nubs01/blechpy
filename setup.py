import setuptools
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='blechpy',
    version='1.0.0',
    author='Roshan Nanu',
    author_email='roshan.nanu@gmail.com',
    description='Package for exrtacting, processing and analyzing Intan and OpenEphys data',
    long_description=long_description,
    long_desription_content_type='text/markdown',
    url='https://github.com/nubs01/blechpy',
    packages=setuptools.find_packages('blechpy', exclude=['bin', 'docs', 'tests']),
    package_dir={'':'blechpy'},
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Linux'],
    keywords='blech katz_lab Intan electrophysiology neuroscience',
    python_requires='>=3.6',
    install_requires=['datashader', 'numpy', 'pandas',
                      'scipy', 'tkinter', 'matplotlib',
                      'tables', 'scikit-learn', 'easygui',
                      'pyqt5', 'tqdm', 'seaborn'],
    include_package_data=True,
    package_data={'': ['environment.yml'], 'dio': ['defaults/*.json']}
)




