"""from setuptools import setup, find_packages

setup(
    name='math_func',  # Name of your package
    version='0.1',  # Version of your package
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=[],  # List dependencies if any, e.g., ['numpy', 'requests']
    author='Manuel Rao',
    author_email='manumanolo178@gmail.com',
    description='A short description of your module',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/matiszpek/Proyecto-4to',  # URL to your project's homepage
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum Python version requirement
)
"""

from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "precence_map_module",
        ["E:/projects/tecnico/proyecto ORT2024/Proyecto-4to/precence_map_module.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
]

setup(
    name="precence_map_module",
    ext_modules=ext_modules,
    zip_safe=False,
)
