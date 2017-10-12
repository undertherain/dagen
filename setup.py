import setuptools
import typing as t
import os
import shutil
import importlib


_HERE = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Environment :: Console',
    'License :: OSI Approved :: Apache Software License',
    'Natural Language :: English',
    'Operating System :: POSIX',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3 :: Only'
]

description = "syntetic data generation for deep learning experiments"
name = "dagen"
url = "https://github.com/undertherain/dagen"


def parse_requirements(requirements_path: str = 'requirements.txt') -> t.List[str]:
    """Read contents of requirements.txt file and return data from its relevant lines.

    Only non-empty and non-comment lines are relevant.
    """
    requirements = []
    with open(os.path.join(_HERE, requirements_path)) as reqs_file:
        for requirement in [line.strip() for line in reqs_file.read().splitlines()]:
            if not requirement or requirement.startswith('#'):
                continue
            requirements.append(requirement)

    return requirements


def parse_readme(readme_path: str = 'README.rst', encoding: str = 'utf-8') -> str:
    """Read contents of readme file (by default "README.rst") and return them."""
    with open(os.path.join(_HERE, readme_path), encoding=encoding) as readme_file:
        desc = readme_file.read()
    return desc


def clean(build_directory_name: str = 'build') -> None:
    """Recursively delete build directory (by default "build") if it exists."""
    build_directory_path = os.path.join(_HERE, build_directory_name)
    if os.path.isdir(build_directory_path):
        shutil.rmtree(build_directory_path)


def find_version(
        package_name: str, version_module_name: str = '_version',
        version_variable_name: str = 'VERSION') -> str:
    """Simulate behaviour of "from package_name._version import VERSION", and return VERSION."""
    version_module = importlib.import_module('{}.{}'.format(package_name, version_module_name))
    return getattr(version_module, version_variable_name)


def setup():
    setuptools.setup(
        name=name,
        version=find_version(name.replace('-', '_')),
        url=url,
        classifiers=classifiers,
        keywords=['deep learning', 'dataset', 'generation'],
        install_requires=parse_requirements(),
        description=description,
        long_description=parse_readme(),
        packages=setuptools.find_packages(exclude=['contrib', 'docs', 'test*'])
    )


def main() -> None:
    clean()
    setup()


if __name__ == '__main__':
    main()
