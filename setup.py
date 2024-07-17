from setuptools import setup, find_packages

setup(
    name="AttentionAML",
    version="1.0.0",
    description="Identifying acute myeloid leukemia subtypes based on an attention based MLP model",
    url="https://github.com/wan-mlab/AttentionAML",
    author="Lusheng Li, Shibiao Wan",
    author_email="lli@unmc.edu",
    license="MIT",
    packages=find_packages(where='./AttentionAML'),
    package_dir={
        '': 'AttentionAML'
    },
    include_package_data=True,
    install_requires=[
        "scikit-learn==1.2.1",
        "scipy==1.7.3",
        "torch"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6"
)
