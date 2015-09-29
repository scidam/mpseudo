# -*- coding: utf-8 -*-
import os
from distutils.core import setup
import codecs

def read(fname):
    return codecs.open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()

setup(name='mpseudo',
      packages=['mpseudo'],
      version='0.1',
      description='Multicore and precise computation of pseudospectra'
                  + ' of a square matrices',
      long_description=read('README.md'),
      author='Dmitry E. Kislov',
      author_email='kislov@easydan.com',
      url='https://github.com/scidam/mpseudo',
      requires=['numpy (>=1.7)', ],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Mathematics'
          ],
      )