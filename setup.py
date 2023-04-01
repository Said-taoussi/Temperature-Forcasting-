from setuptools import setup, find_packages

setup(
    name='Statistic-Project-2024',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy',
        'statsmodels',
    ],
    description='A package for temperature analysis',
    author='Said ETTOUSY',
    author_email='said.ETTOUSY@emines.um6p.ma',
    url='https://github.com/Said-taoussi/Statistic-Project-2024',
    download_url='https://github.com/Said-taoussi/mStatistic-Project-2024/archive/v0.1.tar.gz',
    keywords=['temperature', 'analysis', 'data'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
