from tabulate import tabulate
import pandas as pd
import numpy as np

class Metrics():
    def __init__(self):
        self.results = {}

    def performance_metrics(
        self,
        predictions: np.array, 
        labels:np.array,
        name: str
    ):
        """
        This function is used to compute key metrics
        -predictions: the predictione from the model
        -labels: the real observations
        """
        mse = ((predictions - labels)**2).mean()
        rmse = np.sqrt(mse)
        mae = (abs(predictions - labels)).mean()
        r2 = 1 - mse/(labels**2).mean()
        self.results[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
    
    def build_tabulate(self):
        df = pd.DataFrame(self.results)
        df = df.loc[["MSE", "RMSE", "MAE", "R2"]]
        df = df.applymap(lambda x: f"{x:.3%}")
        print(tabulate(df, headers="keys", tablefmt="fancy_grid"))

def real_vol(
        data: np.array, 
        daily = True
        ):
    """
    This function is used to compute the realized volatility
    """
    if daily:
        real_vol = np.sqrt(252 * data ** 2)
    else:
        real_vol = np.sqrt(252/len(data) * np.sum(data ** 2))
    return real_vol

def resample(
        data: pd.DataFrame, 
        period: str = 'M', 
        daily = False
        ):
    resampled_df = pd.DataFrame()
    resampled_df['Real_vol'] = data['Return'].resample(period).apply(lambda x: real_vol(x, daily=daily))
    resampled_df['Return'] = data['Return'].resample(period).sum()
    return resampled_df