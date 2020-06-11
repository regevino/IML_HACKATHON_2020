"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Authors:

===================================================
"""


class GitHubClassifier:

    def classify(self, X):
        """
        Receives a list of m unclassified pieces of code, and predicts for each
        one the Github project it belongs to.
        :param X: a numpy array of shape (m,) containing the code segments (strings)
        :return: y_hat - a numpy array of shape (m,) where each entry is a number between 0 and 6
        0 - building_tool
        1 - espnet
        2 - horovod
        3 - jina
        4 - PuddleHub
        5 - PySolFC
        6 - pytorch_geometric
        """

    raise NotImplementedError("TODO: Implement this method by 11pm Friday!")
