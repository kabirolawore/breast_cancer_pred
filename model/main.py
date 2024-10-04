import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle5 as pickle


def create_model(dataset):
    X = dataset.drop(['diagnosis'], axis=1)
    Y = dataset['diagnosis']
    
    validation_size = 0.20
    seed = 7
    
    # split the data
    X_train, X_validation, Y_train, Y_validation = train_test_split(
                                X, Y, test_size=validation_size, random_state=seed)
    
    # scale the data
    init_scaler = StandardScaler() # return scaler
    scaler = init_scaler.fit(X_train) 
    rescaled_X = scaler.transform(X_train)

    # train the model
    model = LogisticRegression(max_iter=3500) # return model
    model.fit(rescaled_X, Y_train)

    # test the model
    rescaled_validation_X = scaler.transform(X_validation)
    predictions = model.predict(rescaled_validation_X)
    print(f"Accuracy of the model: {accuracy_score(Y_validation, predictions):.3f}")
    print('----------------------\n')
    print(f"Confusion matrix: \n {confusion_matrix(Y_validation, predictions)}")
    print('----------------------\n')
    print(f'Classification report: \n {classification_report(Y_validation, predictions)}')

    return model, init_scaler


def get_clean_data():
    # Load the data
    data = pd.read_csv("../data/data.csv")

    # Clean the data
    data = data.drop_duplicates()
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data


def main():
    # get the data
    data = get_clean_data()
    
    # create the model
    model, scaler = create_model(data)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)



if __name__ == "__main__":
    main()