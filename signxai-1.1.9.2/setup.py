import platform
from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='signxai',
    version='1.1.9.2',
    packages=['signxai.methods', 'signxai.methods.Innvestigate', 'signxai.methods.Innvestigate.tests', 'signxai.methods.Innvestigate.tests.tools',
              'signxai.methods.Innvestigate.tests.utils', 'signxai.methods.Innvestigate.tests.utils.keras',
              'signxai.methods.Innvestigate.tests.utils.tests', 'signxai.methods.Innvestigate.tests.analyzer',
              'signxai.methods.Innvestigate.tools', 'signxai.methods.Innvestigate.utils', 'signxai.methods.Innvestigate.utils.keras',
              'signxai.methods.Innvestigate.utils.tests', 'signxai.methods.Innvestigate.utils.tests.cases',
              'signxai.methods.Innvestigate.backend', 'signxai.methods.Innvestigate.analyzer',
              'signxai.methods.Innvestigate.analyzer.canonization', 'signxai.methods.Innvestigate.analyzer.relevance_based',
              'signxai.methods.Innvestigate.applications', 'signxai.examples', 'signxai.utils'],
    url='https://github.com/nilsgumpfer/SIGN-XAI',
    license='BSD 2-Clause License',
    author='Nils Gumpfer',
    author_email='nils.gumpfer@kite.thm.de',
    maintainer='Nils Gumpfer',
    maintainer_email='nils.gumpfer@kite.thm.de',
    description='SIGNed explanations: Unveiling relevant features by reducing bias',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['XAI', 'SIGN', 'LRP'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=[
        'tensorflow>=2.8.0,<=2.12.1 ; platform_system=="Linux"',
        'tensorflow>=2.8.0,<=2.12.1 ; platform_system=="Windows"',
        'tensorflow-macos>=2.8.0,<=2.12.0 ; platform_system=="Darwin"',
        'matplotlib>=3.7.0',
        'scipy>=1.10.0',
        'version-parser>=1.0.1'
    ],
    include_package_data=True,
)
