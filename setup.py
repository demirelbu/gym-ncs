from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gym-ncs',
      version='0.0.1',
      author='Burak Demirel',
      author_email='burak.demirel@protonmail.com',
      description='Periodic scheduling of independent feedback control loops.',
      url='https://github.com/demirelbu/sampleproject',
      packages=find_packages(),
      install_requires=['gym~=0.17', 'numpy~=1.19', 'scipy~=1.5'],
      python_requires='~=3.8'
)
