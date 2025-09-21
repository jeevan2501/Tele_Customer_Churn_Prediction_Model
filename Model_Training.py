from sklearn.ensemble import RandomForestClassifier

class ModelTraining:
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None

    def train(self, X_train, y_train):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        return self.model
