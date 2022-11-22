from setuptools import setup

setup(
   name='nystrom',
   version='0.0.0',
   author='Benjamin Spreng',
   author_email='sprengjamin@gmail.com',
   packages=['nystrom', 'nystrom.plane', 'nystrom.sphere', 'nystrom.plane_sphere', 'nystrom.sphere_sphere', 'nystrom.ufuncs', 'nystrom.materials'],
   license='LICENSE.txt',
   description='Casimir interaction involving spheres and planes',
   long_description=open('README.txt').read(),
   install_requires=[],
)
