from setuptools import setup, find_packages

setup(
        name='CFSP',
        version='1.0',
        description='Empower your workflow with our toolkit: Word Segmentation with target word search, Frame Identification, Argument Identification and Role Identification. Chain these functions in a flexible pipeline, allowing users to search for all target words or specify their own, delivering tailored results.',
        author='',
        author_email='',
        install_requires=[
                'numpy>=1.20.1',
                'pydantic>=1.8.2',
                'torch>=1.9.1',
                'transformers>=4.5.1',
                'ltp>=4.2.13',
        ],
        packages=find_packages(),
        include_package_data=True,
        package_data={
        'CFSP': ['data/all_targets.bin'],
        }
)
