import os
import pandas as pd
from sklearn.model_selection import train_test_split

data = 'E:/BtlAI/asl_alphabet_train/asl_alphabet_train'

def create_dataframe(data_path):
    filepaths, labels = [], []
    for fold in os.listdir(data_path):
        f_path = os.path.join(data_path, fold)
        for img in os.listdir(f_path):
            filepaths.append(os.path.join(f_path, img))
            labels.append(fold)
    return pd.DataFrame({'Filepaths': filepaths, 'Labels': labels})

df = create_dataframe(data)

# Chia dữ liệu
train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=42)
valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=42)
