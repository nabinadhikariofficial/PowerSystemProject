from generator import DataProcessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


class modal():
    def __init__(self, data_name, label_names, feature_names, modal_name):
        self.data = DataProcessor(data_name)
        self.labels = self.data.data[label_names]
        self.features = self.data.data[feature_names]
        self.train, self.test, self.train_labels, self.test_labels = train_test_split(
            self.labels, self.features, test_size=0.2, random_state=42)
        self.modal_ = modal_name.fit(self.train, self.train_labels)

    def modal_accuracy(self):
        pred = self.modal_.predict(self.test)
        print(accuracy_score(self.test_labels, pred))


for i in range(10):
    modal1 = modal('data_less_trunc', [
        'Po_SG_MVA', 'Po_GFM_MVA', 'Po_GFL_MVA'], 'Total_MVA', SVC())

    modal1.modal_accuracy()
