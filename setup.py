from setuptools import setup

setup(
   name='caspwn',
   version='0.0.0',
   author='Benjamin Spreng',
   author_email='sprengjamin@gmail.com',
   packages=['caspwn', 'caspwn.plane', 'caspwn.sphere', 'caspwn.plane_sphere', 'caspwn.sphere_sphere', 'caspwn.ufuncs', 'caspwn.materials'],
   license='LICENSE.txt',
   description='Casimir interaction involving spheres and planes',
   long_description=open('README.txt').read(),
   install_requires=[],
)
