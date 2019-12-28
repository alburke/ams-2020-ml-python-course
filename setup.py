"""Setup file for ams-2020-ml-python-course."""

from setuptools import setup

PACKAGE_NAMES = ['interpretation', 'evaluation']

KEYWORDS = [
    'machine learning', 'deep learning', 'artificial intelligence',
    'data mining', 'weather', 'meteorology', 'atmospheric science',
    'thunderstorm', 'tornado'
]

SHORT_DESCRIPTION = (
    'Python library for machine-learning short course at AMS 2020.'
)

LONG_DESCRIPTION = (
    'Python library for short course on machine learning at AMS (American'
    'Meteorological Society) 2020 Annual Meeting.'
)

CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2.7'
]

if __name__ == '__main__':
    setup(
        name='ams-2020-ml-python-course',
        version='0.1',
        description=SHORT_DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license='MIT',
        author='Amanda Burke',
        author_email='aburke1@ou.edu',
        url='https://github.com/alburke/ams-2020-ml-python-course',
        packages=PACKAGE_NAMES,
        scripts=[],
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        include_package_data=True,
        zip_safe=False
    )
