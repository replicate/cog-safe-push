from setuptools import setup, find_packages

setup(
    name="cog-safe-push",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "replicate>=0.31.0,<1",
        "anthropic>=0.21.3,<1",
        "pillow>=10.0.0",
        "ruff>=0.6.1,<1",
    ],
    entry_points={
        "console_scripts": [
            "cog-safe-push=cog_safe_push.main:main",
        ],
    },
    author="Andreas Jansson",
    author_email="andreas@replicate.com",
    description="Safely push a Cog model, with tests",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    url="https://github.com/andreasjansson/cog-safe-push",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
