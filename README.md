# NE-DT
Code for A Nash Equilibria Decision Tree for Binary Classification

Decision trees rank among the most popular and efficient classification methods. They are used to represent rules for recursively partitioning the data space into regions from which reliable predictions regarding classes can be made. These regions are usually delimited by axis-parallel or oblique hyperplanes. Axis-parallel hyperplanes are intuitively appealing and have been widely studied. However, there is still room for exploring different approaches. In this paper, a splitting rule that constructs axis-parallel hyperplanes by computing the Nash equilibrium of a game played at the node level is used to induct a Nash Equilibrium Decision Tree for binary classification. Numerical experiments are used to illustrate the behavior of the proposed method.

- `nedt.py` contains the implementation of __NEDT__ classifier and the methods `fit()` and `predict()` used to fit and predict.
- `test_nedt.py` show a usage example (classify some random generated data).
