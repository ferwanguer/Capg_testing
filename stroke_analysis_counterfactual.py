import pandas
import numpy
import dice_ml
from dice_ml import Dice
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import IPython  # for dataframe visualization
from sklearn.datasets import load_iris

class StrokeCounterFactuals:
    def __init__(self, path):
        self.healthcare_dataset = load_iris(as_frame=True).frame
        self.continuous_features = self.healthcare_dataset.drop('target', axis=1).columns.tolist()

        pandas.set_option('display.max_columns', 7)
        pandas.set_option('display.width', 200)

    def preprocess_dataset(self):
        simplified_dataset = self.healthcare_dataset.loc[:,
                             ['age', 'hypertension', 'heart_disease', 'avg_glucose_level',
                              'bmi', 'smoking_status', 'stroke']]

        simplified_dataset.dropna(subset=['bmi'], inplace=True)
        print(f'Shape of simplified dataset {simplified_dataset.shape}')
        mapping = {'formerly smoked': 1.0,
                   'smokes': 1.0,
                   'never smoked': 0.0,
                   'Unknown': 0.0}
        simplified_dataset.replace({'smoking_status': mapping}, inplace=True)

        return simplified_dataset

    def ai_pipeline(self, simplified_dataset):
        target = simplified_dataset['target']
        dataset_X = simplified_dataset.drop('target', axis=1)
        print(f'Shape of dataset X for training {dataset_X.shape}')
        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(dataset_X,
                                                                                                        target,
                                                                                                        test_size=0.2,
                                                                                                        random_state=0,
                                                                                                        stratify=target)
        print(f'Shape of x train {self.x_train.shape}')


        ai_pipeline = Pipeline(steps=[('preprocessor', StandardScaler()),
                                      ('classifier', RandomForestClassifier())])
        modelo = ai_pipeline.fit(self.x_train, self.y_train)
        print(type(modelo))
        return modelo

    def counterfactual_generator_dice(self, model, dataset, input_datapoint):
        data_dice = dice_ml.Data(dataframe=dataset,
                                 continuous_features=self.continuous_features,
                                 outcome_name='target')
        model_dice = dice_ml.Model(model=model, backend='sklearn', model_type='classifier')


        explainer = Dice(data_dice, model_dice, method='genetic')

        features_to_vary = ['avg_glucose_level', 'bmi', 'heart_disease', 'hypertension']
        permited_ranges = {'avg_glucose_level': [50, 250], 'bmi': [18, 35]}  # Only of the continuous variables
        n_samples = 5
        print(f'Start of the prediction of {n_samples} counterfactuals')
        counterfactuals = explainer.generate_counterfactuals(input_datapoint, total_CFs=n_samples,
                                                             desired_class=2)

        return counterfactuals


if __name__ == '__main__':
    d = StrokeCounterFactuals("healthcare-dataset-stroke-data.csv")
    simplified_data = d.healthcare_dataset


    model = d.ai_pipeline(simplified_data)


    quer = d.x_test[2:3]
    print(f' Prediction {model.predict(quer)}')

    counterfactuals = d.counterfactual_generator_dice(model, simplified_data, quer)
    counterfactuals.visualize_as_dataframe()

    print('---Process ended---')
