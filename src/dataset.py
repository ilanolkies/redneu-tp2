import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path, header=None)

    # datos de entrada
    data = pd.DataFrame(df[df.columns[1:]]).to_numpy()
    labels = pd.DataFrame(df[df.columns[0:1]]).to_numpy()

    return data, labels
