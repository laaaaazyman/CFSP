from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
        name='CFNCFSP',
        version='1.1',
        description='Empower your workflow with our toolkit: Word Segmentation with target word search, Frame Identification, Argument Identification and Role Identification. Chain these functions in a flexible pipeline, allowing users to search for all target words or specify their own, delivering tailored results.',
        author='',
        author_email='',
        long_description=long_description,
        long_description_content_type="text/markdown",
        install_requires=[
                'pydantic==2.4.2',
                'torch==1.13.1',
                'transformers==4.24.0',
                'ltp==4.2.13',
        ],
        packages=find_packages(),
        include_package_data=True,
        package_data={
        'CFNCFSP': ['data/all_targets.bin'],
        }
)
