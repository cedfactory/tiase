#!/usr/bin/env python

import tiase

from setuptools import setup,find_packages


setup(
    name='tiase',
    description="tiase",
    version=tiase.__version__,
    author=tiase.__author__,
    author_email=tiase.__email__,
    url="https://github.com/cedfactory/tiase",
    packages=find_packages(),
    include_package_data=True,
    install_requires=['imblearn==0.0', 'lxml==4.6.2', 'matplotlib==3.3.4',
                    'mplfinance==0.12.7a10', 'numpy==1.19.5', 'plotly==4.14.3',
                    'pandas==1.2.2', 'seaborn==0.11.1', 'statsmodels==0.12.2',
                    'stockstats==0.3.2', 'yfinance==0.1.63', 'finta==1.3', 
                    'scikit-learn==0.24.1','xgboost==1.3.3', 'six~=1.15.0',
                    'tensorboard==2.7.0', 'tensorflow==2.6.0', 'rich==10.6.0',
                    'Keras==2.6.0', 'Keras-Preprocessing==1.1.2',
                    'scikeras==0.4.1', 'joblib==1.0.1', 'parse==1.19.0'
    ],
    extras_require={
            "dev": ["pytest", "pytest-cov"],
    },
    python_requires='>=3.8',
)

