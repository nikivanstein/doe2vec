from setuptools import setup, find_packages


setup(
    name='doe2vec',
    version='0.6',
    license='MIT',
    author="Bas van Stein",
    author_email='b.van.stein@liacs.leidenuniv.nl',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/Basvanstein/doe2vec',
    keywords='autoencoder, representation learning, self supervised learning',
    install_requires=[
          'scikit-learn',
          'tensorflow',
          'matplotlib',
          'mlflow',
          'pandas'
      ],
)