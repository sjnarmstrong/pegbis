from setuptools import setup, find_packages
from torch.utils import cpp_extension


ext = cpp_extension.CppExtension(
   'pegbis._C',
   ['pegbis/csrc/segment-graph.cpp'],
)

setup(
     name='pegbis',
     version='0.1',
     packages=find_packages(),
     ext_modules=[ext],
     cmdclass={'build_ext': cpp_extension.BuildExtension}
 )
