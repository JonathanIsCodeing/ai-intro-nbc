import numpy as np
from tqdm import tqdm


class NBC:
    def __init__(self, feature_list, labels):
        self.num_labels = 2   # only 1 = True and 0 = False
        self.num_features = len(feature_list[0])
        self.num_items = len(feature_list[:])

        # compute total occurrence of all features for both labels
        features = np.ones((2, self.num_features))  # init with ones for laplace smoothing
        total = [self.num_features, self.num_features]
        for i, X in enumerate(feature_list):
            label = labels[i]
            for j in range(self.num_features):
                features[label][j] += X[j]
                total[label] += X[j]

        # calculate likelihood of each feature
        self.likelihoods = [[features[0][i] / total[0] for i in range(self.num_features)],
                            [features[1][i] / total[1] for i in range(self.num_features)]]
        print(f'Trained model with {self.num_items} data items.')

    def classify(self, items: list):
        predictions = []
        for item in tqdm(items, total=len(items), desc='Make predictions: '):
            max_prob = -1
            for label in range(self.num_labels):
                prob = len(self.likelihoods[label]) / self.num_items
                for i in range(self.num_features):
                    prob *= self.likelihoods[label][i]**item[i]

                if prob > max_prob:
                    max_prob = prob
            predictions.append(max_prob)

        return predictions
