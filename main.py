from generator import DataProcessor
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

from sklearn.svm import SVC
from sklearn import ensemble
from sklearn import multioutput

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error


class modal():
    def __init__(self, data_name, label_names, feature_names, modal_name):
        self.data = DataProcessor(data_name)
        self.labels = self.data.data[label_names]
        self.features = self.data.data[feature_names]
        self.train, self.test, self.train_labels, self.test_labels = train_test_split(
            self.labels, self.features, test_size=0.2, random_state=42)
        self.modal_ = modal_name.fit(self.train, self.train_labels)

    def modal_accuracy_c(self):
        pred = self.modal_.predict(self.test)
        print("Acurracy is: "+str(accuracy_score(self.test_labels, pred)))

    def modal_accuracy_r(self):
        pred = self.modal_.predict(self.test)
        print("Mean Squared Error is: " +
              str(mean_squared_error(self.test_labels, pred)))

# we can keep on adding to this list for different algo test


algo_list = [multioutput.MultiOutputRegressor(ensemble.GradientBoostingRegressor()),
             multioutput.MultiOutputRegressor(
                 ensemble.HistGradientBoostingRegressor()),
             multioutput.MultiOutputClassifier(
                 ensemble.RandomForestClassifier()),
             multioutput.MultiOutputClassifier(ensemble.GradientBoostingClassifier())]

for algo in algo_list:
    modal1 = modal('data_less_trunc', ['Total_MVA', 'Po_GFM_MVA'], [
        'Po_SG_MVA', 'Po_GFL_MVA'], algo)
    try:
        modal1.modal_accuracy_r()
    except:
        modal1.modal_accuracy_c()
