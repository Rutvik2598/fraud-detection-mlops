"""M4 feedback loop: delayed-label simulation, join-back, Prefect retraining.

The loop that connects the two planes: late-arriving labels become training data,
a challenger is retrained on the freshened data, and it is promoted in the MLflow
registry only if it beats the current champion on the held-out validation window.
"""
