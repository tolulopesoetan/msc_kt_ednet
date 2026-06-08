02-06-2026

I have ste the phase 1 notebook for the smoke test and made several technical decisons to keep the notebook reproducible and aligned with the dissertation plan.

The data pipeline loads a small subset of the user csv files for an initial smoke test, attaches a user_id from each filename, converts timestamps into datetime format, and combines the files into one interaction dataframe. The interaction data is then joined with the question metadata, and rows with missing metadata are removed to avoid invalid training examples. A correctness label is created by cleaning and comparing each learner’s answer with the official correct answer.

For the knowledge-tracing representation, each question is assigned a skill using the first tag from EDNet’s tags field. This is a simplified Phase 1 decision that gives each interaction a single skill label, making it suitable for a first BKT smoke test. The notebook also includes checks for dataset size, number of users, number of questions, number of skills, and overall correctness rate.

For modelling, the data is converted into a pyBKT-compatible format. A student-level train/test split is used so that the same learner does not appear in both training and testing data. Rare skills are filtered using the training set only, which avoids leaking information from the test set into preprocessing. Because pyBKT’s built-in evaluation caused NaN-related issues, the evaluation was changed to use model predictions directly, with AUC and accuracy calculated manually using scikit-learn.

At this stage, the work establishes a clean BKT baseline pipeline for Phase 1. The next technical step is to extend the same preprocessing and evaluation structure to DKT and SAKT so that all three models can be compared under a shared workflow.