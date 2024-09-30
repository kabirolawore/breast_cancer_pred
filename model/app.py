import pandas as pd


def get_clean_data():
    # Load the data
    data = pd.read_csv("../data/data.csv")
    print(data.head(10))

    # Clean the data
    data = data.dropna()
    data = data.drop_duplicates()

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