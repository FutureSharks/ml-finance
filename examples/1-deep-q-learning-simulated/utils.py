from sklearn.preprocessing import StandardScaler


def get_scaler(env):
    """
    Takes a env and returns a scaler for its observation space
    """

    low = []
    high = []

    # Add position range
    low.append(0)
    high.append(2)

    # Add unrealised account balance range
    low.append(-300)
    high.append(300)

    # Add range for environment columnns
    low.extend([0] * len(env.environment_columns))
    high.extend(env.price_data[env.environment_columns].max().tolist())

    # Create a fit the scaler
    scaler = StandardScaler()
    scaler.fit([low, high])

    return scaler
