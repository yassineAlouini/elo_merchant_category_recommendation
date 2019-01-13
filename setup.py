from setuptools import find_packages, setup

NAME = 'elo_merchant_category_recommendation'
VERSION = '0.0.1'
AUTHOR = 'Yassine Alouini'
DESCRIPTION = """The repo for the elo merchant category recommendation challenge."""
EMAIL = "yassinealouini@outlook.com"
URL = ""

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    # Some metadata
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    url=URL,
    license="MIT",
    keywords="kaggle machine-learning tabular regression",
)
