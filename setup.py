from setuptools import setup, find_packages
import unittest


def test_suite():
    test_loader = unittest.TestLoader()
    return test_loader.discover('tests', pattern='test_*.py')


setup(name='mytorch',
      version='0.1',
      url='https://github.com/choiking10/mytorch',
      license='MIT',
      author='Yunho Choi',
      author_email='choiking10@gmail.com',
      description='Implementation of DeZero (deep learning from scratch-3)',
      packages=find_packages(exclude=['tests', 'example']),
      long_description=open('README.md').read(),
      zip_safe=False,
      setup_requires=['numpy>=1.19.5'],
      test_suite='setup.test_suite')
