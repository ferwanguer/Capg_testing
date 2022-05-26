import pandas
import numpy
import dice_ml
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import IPython  # for dataframe visualization


class StrokeCounterFactuals:
    def __init__(self, path):
        self.healthcare_dataset = pandas.read_csv(path)
        self.continuous_features = ['age', 'avg_glucose_level', 'bmi']
        pandas.set_option('display.max_columns', 7)
        pandas.set_option('display.width', 200)

    def preprocess_dataset(self):
        simplified_dataset = self.healthcare_dataset.loc[:,
                             ['age', 'hypertension', 'heart_disease', 'avg_glucose_level',
                              'bmi', 'smoking_status', 'stroke']]
        simplified_dataset.dropna(subset=['bmi'], inplace=True)

        mapping = {'formerly smoked': 1.0,
                   'smokes': 1.0,
                   'never smoked': 0.0,
                   'Unknown': 0.0}
        simplified_dataset.replace({'smoking_status': mapping}, inplace=True)

        return simplified_dataset

    def ai_pipeline(self, simplified_dataset):
        target = simplified_dataset['stroke']
        dataset_X = simplified_dataset.drop('stroke', axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(dataset_X,
                                                                                                        target,
                                                                                                        test_size=0.2,
                                                                                                        random_state=0,
                                                                                                        stratify=target)
        ai_pipeline = Pipeline(steps=[('preprocessor', sklearn.preprocessing.StandardScaler()),
                                      ('classifier', RandomForestClassifier())])
        model = ai_pipeline.fit(self.x_train, self.y_train)

        return model

    def counterfactual_generator_dice(self, model, dataset, input_datapoint):
        data_dice = dice_ml.Data(dataframe=dataset,
                                 continuous_features=self.continuous_features,
                                 outcome_name='stroke')
        model_dice = dice_ml.Model(model=model, backend='sklearn', model_type='classifier')

        counterfactual_algorithm_selector = 'random'  # 3 available. habría q poner aquí un assert o algo para saber si existe

        explainer = dice_ml.Dice(data_dice, model_dice, method=counterfactual_algorithm_selector)

        features_to_vary = ['avg_glucose_level', 'bmi', 'heart_disease', 'hypertension']
        permited_ranges = {'avg_glucose_level': [50, 250], 'bmi': [18, 35]}  # Only of the continuous variables
        n_samples = 5
        print(f'Start of the prediction of {n_samples} counterfactuals')
        counterfactuals = explainer.generate_counterfactuals(input_datapoint, total_CFs=n_samples,
                                                             desired_class='opposite', verbose= True,
                                                             permitted_range = permited_ranges,
                                                             features_to_vary = features_to_vary)

        return counterfactuals


if __name__ == '__main__':
    d = StrokeCounterFactuals("healthcare-dataset-stroke-data.csv")
    simplified_data = d.preprocess_dataset()
    model = d.ai_pipeline(simplified_data)
    counterfactuals = d.counterfactual_generator_dice(model, simplified_data, d.x_test[0:1])
    counterfactuals.visualize_as_dataframe()

    print('---Process ended---')
