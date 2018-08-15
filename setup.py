from setuptools import setup, find_packages
from leitmotiv._version import version_string

setup(
    name='leitmotiv',
    version=version_string(),
    packages=find_packages(),
    package_data={
        'leitmotiv.webui': ['static/*', 'templates/*']
    },
    install_requires=[
        'flask',
        'click',
        'libsass',
        'peewee',
        'ruamel.yaml',
        'tqdm'
    ],
    entry_points='''
    [console_scripts]
    leitmotiv=leitmotiv.cli:main
    '''
)
