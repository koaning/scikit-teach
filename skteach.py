from sklearn.model_selection import train_test_split
from modAL.models import ActiveLearner
from tqdm import tqdm
import uuid
import numpy as np


def _calc_score(estimator, X, y):
    pred = estimator.predict(X)
    return {"score": np.mean(pred == y)}

def run_experiment(name, data, estimator, strategy, batch_size=100, n_batch=10, start_size=50, idx_start=None):
    """
    Runs an active learning experiment and returns a list of results for a single run. 

    Arguments:
        - name: the name for the experiment
        - data: the (X, y) data pair
        - estimator: the ML model to use
        - strategy: a modAL compatible selection function
        - batch_size: size of each labelling batch
        - n_batch: number of batches to run
        - start_size: if no `idx_start` are passed, how big the starting batch should be
        - idx_start: manually passed indices of the starting batch
    """
    run_stats = {'run_id': str(uuid.uuid4())[:8], 'name': name}
    
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    if not idx_start:
        idx_start = np.random.randint(0, X_train.shape[0], start_size)
    
    # Start up an active learner
    learner = ActiveLearner(
        estimator=estimator,
        query_strategy=strategy,
        X_training=X_train[idx_start, :], 
        y_training=y_train[idx_start]
    )
    
    # Make sure we can't sample same datapoints again.
    X_train, y_train = np.delete(X_train, idx_start, axis=0), np.delete(y_train, idx_start)
    
    # Start logging results
    data = [{"batch": 0, **_calc_score(learner, X_test, y_test), **run_stats}]
    
    for batch in tqdm(range(1, n_batch + 1)):
        # Select new items
        idx, _ = learner.query(X_train, n_instances=batch_size)
        
        # Update learner
        learner.teach(X_train[idx], y_train[idx])
        
        # Make sure we can't sample same datapoints again.
        X_train, y_train = np.delete(X_train, idx, axis=0), np.delete(y_train, idx)

        # Add more logs
        data.append({"batch": batch, **_calc_score(learner, X_test, y_test), **run_stats})
    return data
