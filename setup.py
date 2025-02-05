from setuptools import setup, Extension

setup(
    ext_modules=[Extension("_dummy_ext", sources=[])],
)
