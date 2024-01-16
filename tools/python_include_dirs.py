import sysconfig

import numpy


print(f"-I{sysconfig.get_paths()['include']} -I{numpy.get_include()}")
