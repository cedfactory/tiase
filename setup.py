#!/usr/bin/env python

import tiar

from setuptools import setup,find_packages


setup(
    name='tiar',
    description="tiar",
    version=tiar.__version__,
    author=tiar.__author__,
    author_email=tiar.__email__,
    url="https://github.com/cedfactory/f",
    packages=find_packages(),
    include_package_data=True,
    install_requires=['imblearn==0.0', 'lxml==4.6.2', 'matplotlib==3.3.4',
                    'mplfinance==0.12.7a10', 'numpy==1.19.5', 'plotly==4.14.3',
                    'pandas==1.2.2', 'seaborn==0.11.1', 'statsmodels==0.12.2',
                    'stockstats==0.3.2', 'yfinance==0.1.63', 'finta==1.3', 
                    'scikit-learn==0.24.1','xgboost==1.3.3', 'tensorboard==2.2.2',
                    'tensorboard-plugin-wit==1.8.0', 'tensorflow==2.2.0', 'tensorflow-estimator==2.2.0',
                    'Keras==2.4.3', 'Keras-Preprocessing==1.1.2', 'joblib==1.0.1', 'parse==1.19.0'
    ],
    extras_require={
            "dev": ["pytest", "pytest-cov", "rich"],
    },
    python_requires='>=3.8',
)

