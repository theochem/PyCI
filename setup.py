from setuptools import setup, Distribution

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class MyWheel(_bdist_wheel):

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            return _bdist_wheel.get_tag(self)

    class MyDistribution(Distribution):

        def __init__(self, *attrs):
            Distribution.__init__(self, *attrs)
            self.cmdclass['bdist_wheel'] = MyWheel

        def is_pure(self):
            return False

        def has_ext_modules(self):
            return True

except ImportError:
    class MyDistribution(Distribution):
        def is_pure(self):
            return False

        def has_ext_modules(self):
            return True

setup(
    distclass=MyDistribution
)