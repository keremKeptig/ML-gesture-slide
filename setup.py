from setuptools import setup, find_packages

setup(
    name="framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
    ],
    author="Team-16",
    description="A neural network framework for gesture recognition",
    url="https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws24/Team-16/framework.git",
    keywords="machine learning, gesture recognition, neural network",
    python_requires=">=3.7",
)