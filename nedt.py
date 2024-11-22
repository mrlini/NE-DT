"""
code for the paper:
A Nash Equilibria Decision Tree for Binary Classification

Decision trees rank among the most popular and efficient classification
methods. They are used to represent rules for recursively partitioning
the data space into regions from which reliable predictions regarding
classes can be made. These regions are usually delimited by axis-parallel
or oblique hyperplanes. Axis-parallel hyperplanes are intuitively appealing
and have been widely studied. However, there is still room for exploring
different approaches. In this paper, a splitting rule that constructs
axis-parallel hyperplanes by computing the Nash equilibrium of a game played
at the node level is used to induct a Nash Equilibrium Decision Tree for
binary classification. Numerical experiments are used to illustrate the
behavior of the proposed method.
"""

import math
import numpy as np
from scipy.optimize import minimize


def get_entropy(some_value):
    """
    Compute entropy for a given value

    Args:
        some_value (float|list|np.ndarray): value

    Returns:
        float: entropy
    """
    if isinstance(some_value, (list, np.ndarray)):
        y = some_value
        if len(y) != 0:
            p = sum(y) / len(y)
        else:
            return 1
    else:
        p = some_value
    if (p != 0) and (p != 1):
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    else:
        return 0


def get_entropy_two(y_left, y_right):
    """
    Get entropy for two values as in Zaki Meira, page 489.

    Args:
        y_left (float|list|np.ndarray): labels
        y_right (float|list|np.ndarray): labels

    Returns:
        float: entropy
    """
    n_left = len(y_left)
    n_right = len(y_right)
    n = n_left + n_right
    return n_left / n * get_entropy(y_left) + n_right / n * get_entropy(y_right)


def get_gini(some_value):
    """Compute the gini index for a given value.

    Args:
        some_value (float|list|np.ndarray): value

    Returns:
        float: gini index
    """
    if isinstance(some_value, (list, np.ndarray)):
        y = some_value
        if len(y) != 0:
            p = sum(y) / len(y)
        else:
            return 1
    else:
        p = some_value
    return 1 - (p**2 + (1 - p) ** 2)


def weighted_gini(y_left, y_right):
    """
    Compute the weighted Gini index for a given value as
    in Zaki Meira page 489.

    Args:
        y_left (float|list|np.ndarray): labels
        y_right (float|list|np.ndarray): labels

    Returns:
        float: weighted Gini index
    """
    n_left = len(y_left)
    n_right = len(y_right)
    n = n_left + n_right
    return n_left / n * get_gini(y_left) + n_right / n * get_gini(y_right)


class Node:
    """Node for the decision tree, based on NE."""

    def __init__(
        self,
        criterion,
        split_criterion,
        num_samples,
        num_samples_per_class,
        predicted_class,
        predicted_class_prob,
        k_quantile_zero=0.2,
    ):
        self.criterion = criterion
        self.split_criterion = split_criterion
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.predicted_class_prob = predicted_class_prob
        self.k_quantile_zero = k_quantile_zero
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.beta = 0
        self.attribute = -1

    def split_attributes(self, X, y):
        """find the best attribute to split data

        Args:
            X (np.ndarray): data instances
            y (np.ndarray): true labels

        Returns:
            beta, the attribute and the threshold
        """
        n_attributes = X.shape[1]
        Beta = np.zeros((n_attributes, 2))
        Gain = np.zeros(n_attributes)
        attribute_cut = np.zeros(n_attributes)
        n0 = np.sum(y == 0)
        n1 = np.sum(y == 1)
        beta_new = np.zeros(2)

        if self.split_criterion == "entropy":
            opt_func = self.entropy_gain
        elif self.split_criterion == "gini":
            opt_func = self.gini_gain

        for attribute_i in range(n_attributes):
            beta_new[0] = 1 / 8 * (n0 - n1)
            beta_new[1] = 1 / 8 * np.sum(X[:, attribute_i] - 2 * X[:, attribute_i] * y)
            Beta[attribute_i] = beta_new.copy()

            Gain[attribute_i], attribute_cut[attribute_i] = opt_func(
                X, y, Beta[attribute_i], attribute_i, 0
            )

        # select the best attribute and beta
        self.attribute = np.argmax(Gain)

        self.beta = Beta[self.attribute].copy()
        self.threshold = attribute_cut[self.attribute]
        return self.beta, self.attribute, self.threshold

    def split_attributes_no_game(self, X, y):
        """split data without the game

        Args:
            X (np.ndarray): data instances
            y (np.ndarray): true labels

        Returns:
            beta, attribute and threshold
        """
        n_attributes = X.shape[1]
        Beta = np.zeros((n_attributes, 2))
        Gain = np.zeros(n_attributes)
        cuts = np.zeros(n_attributes)
        beta_new = np.zeros(2)

        if self.split_criterion == "entropy":
            opt_func = self.entropy_gain
        elif self.split_criterion == "gini":
            opt_func = self.gini_gain

        for attribute_i in range(n_attributes):
            beta_new[0] = 0
            beta_new[1] = 1
            Beta[attribute_i] = beta_new.copy()

            Gain[attribute_i], cuts[attribute_i] = opt_func(
                X, y, Beta[attribute_i], attribute_i, 0
            )

        self.attribute = np.argmax(Gain)
        self.beta = Beta[self.attribute].copy()
        self.threshold = cuts[self.attribute]
        return self.beta, self.attribute, self.threshold

    def gain_entropy_for_opt(self, cut, prod, y):
        """entropy for optimization"""
        y_left = y[prod <= cut]
        y_right = y[prod > cut]
        return -(get_entropy(y) - get_entropy_two(y_left, y_right))

    def gain_gini_for_opt(self, cut, prod, y):
        """gini for optimization"""
        y_left = y[prod <= cut]
        y_right = y[prod > cut]
        return weighted_gini(y_left, y_right)

    def split_attr_lr_gini_entr_opt(self, X, y, beta, feature, thr=0):
        """split with entropy / Gini and opt

        Args:
            X (np.ndarray): data instances to split
            y (np.ndarray): labels
            beta (np.ndarray): beta parameter
            feature (int): feature

        Returns:
            y_left, y_right, cut
        """
        prod = X[:, feature] * beta[1] + beta[0]
        indexes = np.argsort(prod)
        ordered_prod = prod[indexes].copy()
        y_ordered = y[indexes].copy()

        if self.split_criterion == "entropy":
            fct_to_use = self.gain_entropy_for_opt
        elif self.split_criterion == "gini":
            fct_to_use = self.gain_gini_for_opt

        if (beta[0] == 0) & (beta[1] == 1):
            # no game in use
            cut = minimize(
                fct_to_use,
                np.random.uniform(np.min(prod), np.max(prod)),
                args=(ordered_prod, y_ordered),
            ).x
        else:
            # use game
            prod0 = np.quantile(prod[y == 0], self.k_quantile_zero)
            prod1 = np.quantile(prod[y == 1], 1 - self.k_quantile_zero)
            cut = (prod0 + prod1) / 2

        y_left = y_ordered[ordered_prod <= cut]
        y_right = y_ordered[ordered_prod > cut]

        return y_left, y_right, cut

    def entropy_gain(self, X, y, beta, attribute, thr=0):
        """
        Compute y_left, y_right, and gain based on entropy
        """
        (
            y_left,
            y_right,
            cut,
        ) = self.split_attr_lr_gini_entr_opt(X, y, beta, attribute, thr)

        return get_entropy(y) - get_entropy_two(y_left, y_right), cut

    def gini_gain(self, X, y, beta, attribute, thr=0):
        """Compute y_left, y_right, and gain based on gini"""
        (
            y_left,
            y_right,
            cut,
        ) = self.split_attr_lr_gini_entr_opt(X, y, beta, attribute, thr)

        return -weighted_gini(y_left, y_right), cut


class NEDTClassifier:
    """NEDT - Nash Equilibrium based Decision Tree classifier"""

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = 5,
        game_based: int = 1,
        k_quantile_zero: float = 0.2,
    ):
        """constructor for NEDT

        Args:
            criterion (str, optional): split criterion. Defaults to 'gini'.
            max_depth (int, optional): max depth for the decision tree.
                                        Defaults to 5.
            game_based (int, optional): use game to construct the tree,
                                        1 - yes. Defaults to 1 - yes.
            k_quantile_zero (float, optional): quantile. Defaults to 0.2.
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.game_based = game_based
        self.nr_nodes = 0
        self.k_quantile_zero = k_quantile_zero

    def fit(self, X: np.ndarray, y: np.ndarray):
        """fit the classifier

        Args:
            X (np.ndarray): train instances
            y (np.ndarray): train labels
        """
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y, 0)

    def predict(self, X):
        """Make predictions for X.

        Args:
            X (np.ndarray): test instances

        Returns:
            (np.ndarray, np.ndarray): predicted class and predicted class
              probability for each test instance
        """
        pred_class = list()
        pred_prob = list()
        for input in X:
            a, b = self._predict_instance(input)
            pred_class.append(a)
            pred_prob.append(b)
        return pred_class, pred_prob

    def _predict_instance(self, inputs: np.ndarray):
        """
        Predict the class for the given instance.

        Args:
            inputs (np.ndarray): data instance

        Returns:
            tuple: predicted class and predicted class probability
        """
        node = self.tree_
        while node.left:
            if inputs[node.attribute] * node.beta[1] + node.beta[0] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.predicted_class, node.predicted_class_prob

    def _grow_tree(self, X, y, depth=0):
        """Build a decision tree by recursively finding the best split.

        Args:
            X (np.ndarray): data instances
            y (np.ndarray): true labels pt X
            depth (int, optional): max depth for the tree. Defaults to 0.

        Returns:
            decision tree
        """
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        predicted_class_prob = num_samples_per_class[1] / len(y)
        atr_node = 0

        node = Node(
            criterion=atr_node,
            split_criterion=self.criterion,
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
            predicted_class_prob=predicted_class_prob,
            k_quantile_zero=self.k_quantile_zero,
        )

        # Split recursively until maximum depth is reached.
        if (depth < self.max_depth) & (X.shape[0] >= 1) & np.all(num_samples_per_class):
            if self.game_based == 1:
                node.split_attributes(X, y)
            else:
                node.split_attributes_no_game(X, y)
            prod = X[:, node.attribute] * node.beta[1] + node.beta[0]

            X_left, y_left = X[prod <= node.threshold], y[prod <= node.threshold]
            X_right, y_right = X[prod > node.threshold], y[prod > node.threshold]

            if len(y_left) == 0 or len(y_right) == 0:
                print(" ")
            else:
                if len(y_left) > 0:
                    node.left = self._grow_tree(X_left, y_left, depth + 1)
                if len(y_right) > 0:
                    node.right = self._grow_tree(X_right, y_right, depth + 1)

        return node
