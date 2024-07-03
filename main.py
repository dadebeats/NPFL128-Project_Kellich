from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Input, Dropout
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from explo_regression import PerformanceExplorer
from config import load_gamestats

def prepare_data(pe, position, target_column, fs, n, th=None, concat_text=False):
    df = pe.feature_pool[position]
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data = df.drop(columns=["totalScore", "devFromL40"]).select_dtypes(include=numerics).fillna(
        df.drop(columns=["totalScore", "devFromL40"]).select_dtypes(include=numerics).median()).fillna(0)
    data = data.iloc[:, :-12]
    target = df[target_column].replace([np.inf, -np.inf], df[target_column].median()).fillna(45)
    l40s = df["L40"].replace([np.inf, -np.inf], df["L40"].median()).fillna(45)

    col_trans = ColumnTransformer(
        [("scaler", RobustScaler(), [i for i in range(data.shape[1])])],
        remainder="passthrough"
    )

    data = pd.DataFrame(col_trans.fit_transform(data), columns=data.columns, index=data.index)

    relevant_cols = None  # Initialize to None

    if fs == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n)
        pca.fit(data)
        data = pca.transform(data)

    elif fs == "corr":
        corrs = pe.filter_correlations(top_n_corrs=n, threshhold_of_importance=th)
        relevant_cols = corrs[position].keys()

    elif fs == "rf":
        if th is not None:
            th /= 100
        rfs = pe.filter_rf_importances(positions, top_n=n, threshold_of_importance=th)
        relevant_cols = rfs[position].keys()

    elif fs == "handmade":
        keys = pe.handmade_feature_selection(positions)
        relevant_cols = keys[position]

    if relevant_cols:
        data = data[relevant_cols]
    if concat_text:
        text_df = pd.read_csv(f"dataset/2024-03-12_textual/{position}.csv", index_col="gameId")
        data = text_df.merge(data, left_index=True, right_index=True, how='inner')
        target = data["totalScore"]
        data.drop(columns=["totalScore"], inplace=True)
        l40s = l40s[:len(data)]

    x_train, x_temp, t_train, t_temp, _, l40s_temp = train_test_split(data, target, l40s, train_size=0.7, shuffle=False)
    x_val, x_test, t_val, t_test, _, l40s_test = train_test_split(x_temp, t_temp, l40s_temp, train_size=0.5,
                                                                  shuffle=False)

    return x_train, x_val, x_test, t_train, t_val, t_test, l40s_test, relevant_cols


def create_sequences(df: pd.DataFrame, series_target: pd.Series, timestep: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences of a specified length for LSTM model training, ensuring each player has enough data points.

    :param df: DataFrame containing features for each player and game.
    :param series_target: Series containing the target variable for each player and game.
    :param timestep: Length of the sequence to be created for each sample.
    :return: A tuple of numpy arrays (sequences, targets) ready for LSTM training.
    """
    sequences = []
    targets = []
    target_ids =[]
    df_data = df.copy()
    # Create new columns for Player and Date by splitting the index
    player_names = df_data.index.map(lambda x: x.split('_')[0])
    game_dates = df_data.index.map(lambda x: x.split('_')[1])

    # Assign these columns to the DataFrame
    df_data['Player'] = player_names
    df_data['Date'] = pd.to_datetime(game_dates)
    series_target.index = df_data.index  # Ensure target series has the same index for alignment

    # Group by player
    grouped = df_data.groupby('Player')

    for player, group in grouped:
        # Ensure the group is sorted by date
        group = group.sort_values(by='Date')

        # Check if the group has enough data points
        if len(group) > timestep:
            # Iterate over the group to create sequences
            for i in range(len(group) - timestep):
                seq = group.iloc[i:i + timestep].drop(columns=['Player', 'Date']).values  # Extract sequence of features
                target_index = group.index[i + timestep]  # Get the index of the target corresponding to the end of the sequence
                target = series_target.loc[target_index]  # Get the target value

                target_ids.append(target_index)
                sequences.append(seq)
                targets.append(target)

    # Convert lists to numpy arrays
    sequences = np.array(sequences)
    targets = np.array(targets)

    return sequences, targets, target_ids


if __name__ == "__main__":
    game_stats = load_gamestats(is_clustering=False)
    positions = ["Defender", "Midfielder", "Forward"]
    pe = PerformanceExplorer(game_stats)
    pe.load_feature_pool(positions, "dataset/2024-03-12_comprehensive")
    pe.load_rf_feature_importance("dataset/2023-04-27_comprehensive_version/feature_importances.json", plot=False)

    model = Sequential()

    lstm_timesteps = None
    use_text_data = True
    fs = "rf"
    print(lstm_timesteps, use_text_data, fs)
    if fs:
        data_dim = 50
    else:
        data_dim = 658

    if use_text_data:
        data_dim += 10
    model = Sequential()
    if lstm_timesteps:
        model.add(Input(shape=(lstm_timesteps, data_dim)))
        model.add(LSTM(32, dropout=0.5, unroll=True))
    else:
        model.add(Input(shape=[data_dim]))
    for i in range(2):
        model.add(Dense(8, activation="relu"))
        model.add(Dropout(0.5))
    model.add(Dense(1, activation=None))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mean_squared_error'])

    for position in positions:
        pe.feature_importances[position] = {k: v for k, v in pe.feature_importances[position].items() if "u_" not in k and "f_" not in k}
        x_train, x_val, x_test, t_train, t_val, t_test, l40s_test, relevant_cols = prepare_data(pe, position, "totalScore", fs, 50, concat_text=use_text_data)
        _, _, lstm_data_index = create_sequences(x_test, t_test, 2)
        if lstm_timesteps:
            x_train, t_train, _ = create_sequences(x_train, t_train, lstm_timesteps)
            x_test, t_test, _ = create_sequences(x_test, t_test, lstm_timesteps)
            x_val, t_val, _ = create_sequences(x_val, t_val, lstm_timesteps)

        x_test, t_test = x_test.loc[lstm_data_index].sort_index().astype('float32'), t_test.loc[lstm_data_index].sort_index().astype('float32')
        history = model.fit(x_train, t_train, epochs=200, batch_size=8, validation_data=(x_val, t_val))
        test_loss = model.evaluate(x_test, t_test)
        print("Test data size:", len(x_test))
        print(f'Test loss {position}(RMSE): {np.sqrt(test_loss)}')

        average_train_target = np.mean(t_train)
        baseline_predictions = np.full(shape=t_test.shape, fill_value=average_train_target)
        baseline_rmse = np.sqrt(np.mean((t_test - baseline_predictions) ** 2))
        print(f'Baseline RMSE (predicting average of train targets): {baseline_rmse}')
        #predictions = model.predict(x_test)