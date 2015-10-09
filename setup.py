# -*- coding: utf-8 -*-
import os
from distutils.core import setup
import codecs

README_FILE = 'README.md'

def parse_requirements(filename):
    return list(filter(lambda line: (line.strip())[0] != '#',
                       [line.strip() for line in open(filename).readlines()]))


def get_long_description():
    if not os.path.isfile(README_FILE):
        return ''
    try:
        import pypandoc
        doc = open(README_FILE).read()
        description = pypandoc.convert(doc, 'rst')
    except Exception:
        description = open(README_FILE).read()
    return description

setup(name='mpseudo',
      packages=['mpseudo'],
      version='0.1',
      description='Computation of pseudospectra of square matrices in parallel',
      keywords='matrix pseudospectra, eigenvalue problem, computational algebra',
      long_description=get_long_description(),
      author='Dmitry E. Kislov',
      author_email='kislov@easydan.com',
      url='https://github.com/scidam/mpseudo',
      requires=['numpy (>=1.7)',],
      install_requires=parse_requirements('requirements.txt'),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Mathematics'
          ],
      )
