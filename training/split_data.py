import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data_path, train_size=0.8, random_state=42):
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(df, train_size=train_size, random_state=random_state)
    
    # Save the splits to CSV files
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    return train_df, val_df

if __name__ == "__main__":
    split_data("../gen_dataset/data/fin_dataset.csv")
    