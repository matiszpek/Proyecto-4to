from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "precence_map_module",
        ["C:/Users/47872500/Documents/Proyecto-4to/precence_map_module.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
]

setup(
    name="precence_map_module",
    ext_modules=ext_modules,
    zip_safe=False,
)
