import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_model(data):
    X = data.drop('diagnosis', axis=1)
    Y = data['diagnosis']
    return


def get_clean_data():
    # Load the data
    data = pd.read_csv("../data/data.csv")

    # Clean the data
    data = data.drop_duplicates()
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data


def main():
    get_clean_data()
    # create the model

    # Train the model

    # Evaluate the model

    # Make predictions

    return


if __name__ == "__main__":
    main()