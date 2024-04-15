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

            # Calculate class prior probability
            prior_prob = len(train_features_with_label) / len(self.train_labels)

            # Calculate mean and standard deviation for each feature
            class_probs[label] = {
                'prior_prob': prior_prob,
                'mean': np.mean(train_features_with_label, axis=0),
                'std_dev': np.std(train_features_with_label, axis=0)
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
                mean = class_info['mean']
                std_dev = class_info['std_dev']

                # Calculate likelihood using Gaussian assumption
                likelihood = np.prod((1 / (np.sqrt(2 * np.pi) * std_dev)) *
                                     np.exp(-(test_instance - mean)**2 / (2 * std_dev**2)))

                # Calculate posterior probability
                posterior_probs[label] = prior_prob * likelihood

            # Predict the class with the highest posterior probability
            predicted_label = max(posterior_probs, key=posterior_probs.get)
            predicted_labels.append(predicted_label)

        return predicted_labels
