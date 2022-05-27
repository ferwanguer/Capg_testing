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


class DiceCounterFactuals:

    def __init__(self,database):
        self.working_dataset = database
        self.continuous_features = self.working_dataset.drop('target', axis=1).columns.tolist()

        pandas.set_option('display.max_columns', 7)
        pandas.set_option('display.width', 200)

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


        n_samples = 5
        print(f'Start of the prediction of {n_samples} counterfactuals')
        counterfactuals = explainer.generate_counterfactuals(input_datapoint, total_CFs=n_samples,
                                                             desired_class=2)

        return counterfactuals


if __name__ == '__main__':
    d = DiceCounterFactuals(load_iris(as_frame=True).frame)
    data = d.working_dataset

    model = d.ai_pipeline(data)

    quer = d.x_test[2:3]
    #print(f' Prediction {model.predict(quer)}')

    counterfactuals = d.counterfactual_generator_dice(model, data, quer)
    counterfactuals.visualize_as_dataframe()

    print('---Process ended---')
