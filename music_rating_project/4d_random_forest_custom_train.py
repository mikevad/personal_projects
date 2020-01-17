
# libraries necessary for name==main stuff
import argparse
import os

# dataframe libraries
import numpy as np
import pandas as pd

# pipeline building libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

# algorithm library
from sklearn.ensemble import RandomForestClassifier

# persistance function
from sklearn.externals import joblib


# inference functions ---------------
def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf



if __name__ =='__main__':

    parser = argparse.ArgumentParser()

#     hyperparameters to add
#     parser.add_argument('--n-estimators', type=int, default=10)
#     parser.add_argument('--min-samples-leaf', type=int, default=3)

#     Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
#     parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='train.csv')
#     parser.add_argument('--test-file', type=str, default='boston_test.csv')

    args, _ = parser.parse_known_args()

    train = pd.read_csv(f'{args.train}/{args.train_file}')
    y_train = train['liked']
    X_train = train.iloc[:,1]

    # train
    
    tfidf_vectorizer = TfidfVectorizer(stop_words = 'english')
    variance_filter = VarianceThreshold(.0005)
    best_random_forest = RandomForestClassifier(bootstrap=True, class_weight=None,
                                        criterion='gini', max_depth=None,
                                        max_features='auto',
                                        max_leaf_nodes=None,
                                        min_impurity_decrease=0.0,
                                        min_impurity_split=None,
                                        min_samples_leaf=2,
                                        min_samples_split=10,
                                        min_weight_fraction_leaf=0.0,
                                        n_estimators=1200, n_jobs=None,
                                        oob_score=False, random_state=42,
                                        verbose=0, warm_start=False)
    pipeline_steps = [
        ('tfidf_vectorizer', tfidf_vectorizer), # term frequency document infrequency word vectorizer
        ('variance_filter', variance_filter), # removes low variance columns from dataset
        ('classifier', best_random_forest)
    ]
    model = Pipeline(steps = pipeline_steps)
    
    model.fit(X_train, y_train)
        
#     persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    
#     saving model to s3 bucket
    
