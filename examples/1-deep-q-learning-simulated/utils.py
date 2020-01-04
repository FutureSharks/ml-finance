from sklearn.preprocessing import StandardScaler


def get_scaler(env):
    low = [0] * 3
    high = []
    max_price = env.price_data['price'].max()
    max_cash = 10000
    high.append(max_price)
    high.append(max_cash)
    high.append(max_cash)
    scaler = StandardScaler()
    scaler.fit([low, high])
    return scaler
