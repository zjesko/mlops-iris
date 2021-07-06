# ML-Ops demo using a FastAPI application

This repository contains code which demonstrates ML-Ops using a `FastAPI` application which predicts the flower class using the IRIS dataset (https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)


## Setup Project

- Create fork from fork button
- Clone the fork using `git clone https://github.com/purnendukar/mlops-iris`
- Install dependency using `pip3 install -r requirements.txt`

## Running Project

- Run application using `python3 main.py`
- Run test using `pytest`

## CI/CD

- `build` (test) for all the pull requests
- `build` (test) and `upload_zip` for all pushes

## Assignment Tasks

1. Change this README to add your name here: Purnendu Kar. Add and commit changes to a new branch and create a pull request ONLY TO YOUR OWN FORK to see the CI/CD build happening. If the build succeeds, merge the pull request with master and see the CI/CD `upload_zip` take place.
2. Add 2 more unit tests of your choice to `test_app.py` and make sure they are passing.
3. Add one more classifier to startup and use only the one with better accuracy.
4. Add the attribute `timestamp` to the response and return the current time with it. 
