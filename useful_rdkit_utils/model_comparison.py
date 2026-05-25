import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.stats.multicomp import pairwise_tukeyhsd


class WrapperFactory:
    @staticmethod
    def create_wrapper_class(model_class, descriptor_dict, class_name="DynamicMLWrapper", **model_kwargs):
        """
        Dynamically creates a wrapper class with a pre-instantiated model_class.
        Any extra keyword arguments passed to this factory will be forwarded 
        to the model constructor when the wrapper is instantiated.
        """
        
        # 1. __init__ accepts y_col and instantiates the model with the stored kwargs
        def __init__(self, y_col):
            # The model is instantiated using the arguments captured by the outer factory function
            self.model = model_class(**model_kwargs)  
            self.y_col = y_col
            self.fp_name = "fp"
            self.descriptor_dict = descriptor_dict

        # 2. Define the fit method using dictionary map
        def fit(self, train):
            train[self.fp_name] = train.SMILES.map(self.descriptor_dict)
            self.model.fit(np.stack(train[self.fp_name]), train[self.y_col])

        # 3. Define the predict method using dictionary map
        def predict(self, test):
            test[self.fp_name] = test.SMILES.map(self.descriptor_dict)
            pred = self.model.predict(np.stack(test[self.fp_name]))
            return pred

        # 4. Define the validate method
        def validate(self, train, test):
            self.fit(train)
            return self.predict(test)

        # Pack into attributes dictionary
        class_attributes = {
            "__init__": __init__,
            "fit": fit,
            "predict": predict,
            "validate": validate
        }

        # 5. Dynamically construct and return the new class type
        return type(class_name, (object,), class_attributes)

def get_performance_stats(df: pd.DataFrame, y_col: str, method_list: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate R^2 and MAE statistics for each method in the method list.

    :param df: The input DataFrame containing the data.
    :type df: pd.DataFrame
    :param y_col: The name of the column containing the true values.
    :type y_col: str
    :param method_list: A list of method names to evaluate.
    :type method_list: list[str]
    :return: Two DataFrames containing the R^2 and MAE values for each method.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    r2_list = []
    mae_list = []
    result_df = df.query("dset == 'test'").copy()
    for k, v in result_df.groupby("fold"):
        r2_list.append([r2_score(v[y_col], v[x]) for x in method_list])
        mae_list.append([mean_absolute_error(v[y_col], v[x]) for x in method_list])
    r2_df = pd.DataFrame(r2_list, columns=method_list)
    mae_df = pd.DataFrame(mae_list, columns=method_list)
    r2_melt_df = r2_df.melt()
    r2_melt_df.columns = ["method", "r2"]
    mae_melt_df = mae_df.melt()
    mae_melt_df.columns = ["method", "mae"]
    return r2_melt_df, mae_melt_df

def make_tukey_plot(df,y_col,ax=None,method_col="method",xlim=None,title=""):
    if ax == None:
        figure, ax = plt.subplots(1,1)
    tukey = pairwise_tukeyhsd(endog=df[y_col], groups=df[method_col], alpha=0.05)
    ascending=False
    if y_col == "mae":
        ascending=True
    best = df.groupby("method").mean().reset_index().sort_values(y_col,ascending=ascending)[method_col].values[0]
    _ = tukey.plot_simultaneous(comparison_name=best,ax=ax)
    if xlim:
        ax.set_xlim(xlim)
    if y_col == "r2":
        ax.set_xlabel("$R^2$")
    elif y_col == "mae":
        ax.set_xlabel("MAE")
    ax.set_title(title)

