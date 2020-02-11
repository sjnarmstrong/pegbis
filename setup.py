from setuptools import setup, Extension
from torch.utils import cpp_extension


ext = cpp_extension.CppExtension(
   'pegbis.segment_graph',
   ['pegbis/csrc/segment-graph.cpp'],
)

setup(name='segment_graph',
      ext_modules=[ext],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
