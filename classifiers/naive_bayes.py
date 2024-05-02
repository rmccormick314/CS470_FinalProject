import numpy as np

class MyNB():
    def __init__(self):
        self.train_features = []
        self.train_labels = []
        self.name = "Naive Bayes"
        self.trained_model = None

    def fit(self, X, y):
        # Store training data
        self.train_features = X
        self.train_labels = y

        # Calculate class probabilities
        class_probs = {}
        for label in np.unique(self.train_labels):
            # Filter train data by label
            train_features_with_label = self.train_features[self.train_labels == label]

            # Calculate class prior probability with Laplace smoothing
            prior_prob = (len(train_features_with_label) + 1) / (len(self.train_labels) + 2)

            # Calculate P(a|c) for each class and each word
            word_probs = (np.sum(train_features_with_label > 0, axis=0) + 1) / (len(train_features_with_label) + 2)

            class_probs[label] = {
                'prior_prob': prior_prob,
                'word_probs': word_probs
            }

        # Store trained model
        self.trained_model = class_probs

    def predict(self, test_features):
        # Create array to hold predictions
        predicted_labels = []

        # Loop over test instances
        for test_instance in test_features:
            # Calculate posterior probabilities for each class
            posterior_probs = {}
            for label, class_info in self.trained_model.items():
                prior_prob = class_info['prior_prob']
                word_probs = class_info['word_probs']

                # Calculate likelihood using word probabilities
                likelihood = np.prod(test_instance * word_probs + (1 - test_instance) * (1 - word_probs))

                # Calculate posterior probability
                posterior_probs[label] = prior_prob * likelihood

            # Predict the class with the highest posterior probability
            predicted_label = max(posterior_probs, key=posterior_probs.get)
            predicted_labels.append(predicted_label)

        return predicted_labels
