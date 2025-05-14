from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='weak_instruments',  # Package name
    version='0.1.1',  # Initial version
    author='Jonathan Hyatt, Jacob Hutchings',
    author_email='your_email@example.com',  # Replace with your email
    description='A package for analyzing weak instruments in econometrics.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/weak_instruments',  # Replace with your GitHub repo
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'statsmodels',
        'scipy',
        'matplotlib',
        'seaborn',
        'linearmodels',
        'pytest'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)