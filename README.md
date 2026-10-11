# Student Performance Prediction (Regression)

In this project I built regression models to predict a student's final math grade (G3, out of 20).
The idea is that schools often find out a student is struggling too late to help, so I wanted to see how well a grade can be predicted, and what matters most.

## Dataset
I used the UCI Student Performance dataset (math course), file `student-mat.csv`.
It has 395 students and 30 input features like age, absences, past failures, study time and parents' education.
There are no missing values and no duplicate rows, but 38 students have a final grade of 0.

## Tools used
Python, Pandas, NumPy, Scikit-learn, Matplotlib and Jupyter Notebook.

## What I did
1. Loaded the data and checked its quality (missing values, duplicates, grade distribution, correlations).
2. Turned the text columns into 0/1 columns using `pd.get_dummies`.
3. Made two versions of the inputs: one with the earlier grades G1 and G2, and one without them. G1 and G2 are only known after the term starts, so the version without them shows what could be predicted early.
4. Trained 5 models: Linear Regression, Ridge, Decision Tree, Random Forest and Gradient Boosting. I also added a baseline that always guesses the average grade.
5. Evaluated them with RMSE and R2 on a test set (20% of students, 79 students), and also used 5-fold cross-validation on the training data.
6. Checked which features matter most using permutation importance.
7. Looked at the zero-grade students separately, and tried tuning the Random Forest with GridSearchCV.

## Results (test set)
| Case | Best model | RMSE | R2 |
|---|---|---|---|
| With G1 and G2 | Random Forest | 1.949 | 0.815 |
| Without G1 and G2 | Random Forest | 3.935 | 0.245 |
| Baseline (average guess) | - | 4.550 | -0.010 |

## What I found
- When G1 and G2 are available, the models predict the final grade well, and G2 is by far the most important feature.
- Without G1 and G2, the prediction is weak. The best model is only about 13% better than guessing the average.
- Without earlier grades, past failures and absences were the most important features.
- Tuning the Random Forest did not improve the result, and removing the zero-grade students did not fix the weak result either.

## Limitations
- The dataset is small (395 students) and the test set has only 79 students, so the ranking of close models can change with a different split.
- 38 students have a grade of 0 and are hard to predict.
- Feature importance shows what the model uses, not what causes grades to change.
- The model uses sex as a feature, so it should not be used to treat students differently because of it.
- The data comes from two schools in Portugal, so it may not apply to other places.

## How to run
Install the libraries and open the notebook:

    pip install -r requirements.txt
    jupyter notebook student_performance_regression.ipynb

The notebook reads `student-mat.csv` from the same folder.
