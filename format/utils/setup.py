from setuptools import setup, find_packages

setup(
    name="synthproj",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'soundfile',
        'pydub',
        'librosa',
        'demucs',
        'os',
        'pytorch'
    ],
) 