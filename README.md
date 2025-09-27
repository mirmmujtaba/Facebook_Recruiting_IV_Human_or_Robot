# Facebook Recruiting IV: Human vs Robot

Kaggle competition solution for distinguishing human bidders from automated bots in online auctions.

## Competition Overview
- **Competition**: Facebook Recruiting IV: Human or Robot
- **Task**: Binary classification to predict whether a bidder is a human (0) or robot (1)
- **Metric**: AUC (Area Under the ROC Curve)
- **Dataset**: ~7.6 million bids from 2,013 training and 4,700 test bidders

## Solution Approach
- **Feature Engineering**: Created 10+ behavioral features including bid frequency, time gaps, device diversity, country distribution, and auction popularity metrics
- **Model**: Optimized Random Forest Classifier with hyperparameter tuning using RandomizedSearchCV
- **Validation**: Stratified 5-fold cross-validation with holdout set

## Results
- **Leaderboard Score**: 0.93583 (Private: 0.91206)
- **Cross-validation AUC**: 0.929
- **Holdout AUC**: 0.942
- **Late Submission**: Achieved competitive performance that would have placed in the top tier during active competition

## Project Structure
- `facebook-human-vs-robot-notebook.ipynb` - Main analysis and modeling notebook
- `submission.csv` - Final competition predictions
- Feature engineering, model training, and hyperparameter optimization included

## Technical Stack
- Python, pandas, Scikit-Learn, XGBoost, NumPy
- Random Forest with stratified k-fold cross-validation
