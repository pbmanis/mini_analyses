from setuptools import setup
from setuptools import find_packages
# from distutils.extension import Extension
from Cython.Build import cythonize
# from Cython.Distutils import build_ext
# from setuptools.extension import Extension
# from Cython.Build import cythonize

# ext_modules=[ Extension("clembek",
#               ["minis/clembek.pyx"],
#               libraries=["m"],
#               extra_compile_args = ["-ffast-math"])]

setup(name='minis',
      version='0.4.0',
      description='Mini event analysis routines and support',
      url='https://github.com/pbmanis/minis',
      author='Paul B. Manis, Ph.D.',
      author_email='pmanis@med.unc.edu',
      license='MIT',
      packages=['minis'],
      # cmdclass = {"build_ext": build_ext}
      ext_modules=cythonize("minis/clembek.pyx"),
      zip_safe=False,
      )
      