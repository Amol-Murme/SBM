import pandas as pd
import numpy as np
import scorecardpy as sc
from timeit import default_timer as timer


class WOE:

    def __init__(self) -> None:
        pass


    def cal_woebin_iv(self,file_path:str,y,no_of_col_to_read_at_time:int=2,min_iv:int=0.2):
        """
        calculate the WoE and IV for all features of given Dataframe.It uses scorecardpy library.

        Parameters
        ----------
        file_path:str
            Input file path
        y:str
            Dependent variable name.
        no_of_col_to_read_at_time : int
            It specify the number of columns to read in instead of reading whole dataset.
        min_iv : float
            minimum value of IV to select the feature
        """
        start = timer()
        
        iv_new = []
        for group in self.chunker(cols, no_of_col_to_read_at_time):
            group.append(y) 
            print(group)
            df = pd.read_parquet(file_path,columns=group)
            print(df.shape)
            # woe binning ------
            bins = sc.woebin(df, y=y)
            iv_new.append(bins)

        flatten_iv_new ={k: v for d in iv_new for k, v in d.items()}
        col_selected = []
        for key in flatten_iv_new:
            iv_val = flatten_iv_new[key]['total_iv'][0]
            #print(iv_val)
            if ((iv_val >= min_iv) & (iv_val <= 0.5)):
                col_selected.append(key)
        end = timer()
        print("Time to run the function is : ",end-start)
        return col_selected,flatten_iv_new

    def chunker(self,seq, size):
            return (seq[pos:pos + size] for pos in range(0, len(seq), size))


    # def woe_encoding(df,bins_dict):
    #     categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    #     print(categorical_cols)
    #     for i in categorical_cols:
    #         bin = bins_dict[i]
    #         mapping = dict(zip(bin['bin'],bin['woe']))
    #         df[i] = df[i].replace(mapping)
    #     return df

    # def one_hot_encoding(df):
    #     categorical_cols = df.select_dtypes(include=['object']).columns.tolist()  # include 'bool' and 'category' dtype
    #     print(categorical_cols)
    #     df = pd.get_dummies(df, columns = categorical_cols)
    #     return df
    def woe_encoding(self,df:pd.DataFrame,bins_dict:dict):
        """
        Transform the original features to WoE encodeded features

        Parameters
        ----------
        df : dataframe
            DF to be encoded
        bins_dict : dict
            bins dict return by cal_woebin_iv function

        Returns
        -------
        dataframe
            WoE encoded dataframe
        """
        # converting to bin
        dt_bin = sc.woebin_ply(df, bins=bins_dict, to = 'bin')
        str(dt_bin)
        return dt_bin

    def one_hot_encoding(self,df:pd.DataFrame,bins:dict) ->pd.DataFrame:
        """
        Perform one hot encoding on binned features

        Parameters
        ----------
        df : pd.DataFrame
            dataset to be encoded
        bins : dict
            bins dict return by cal_woebin_iv function 

        Returns
        -------
        pd.DataFrame
            One hot encoded dataframe
        """
        df_transform = pd.DataFrame()
        categorical_cols:list = df.select_dtypes(include=['object','category']).columns.tolist()  # include 'bool' and 'category' dtype
        print(categorical_cols)
        if categorical_cols:
            df_transform = pd.get_dummies(df[categorical_cols], columns = categorical_cols)
        cols:list = df.select_dtypes(exclude=['object','category']).columns.tolist()  # include 'bool' and 'category' dtype
        print(cols)
        for col in cols:
            temp = bins[col]
            for i in range(len(temp)):
                l= (bins[col]['bin'][i]).replace('[','').replace(')','').split(',')
                if l[0] != 'missing':
                    min = float(l[0])
                    max = float(l[1])
                    df_transform[col + '_' +temp['bin'][i]] = np.where((df[col] <= max) & (df[col] > min),1,0)
                    

        return df_transform


if __name__ == "__main__":
    woe = WOE()
    input_file_path_parquet = 'Bank Customer Churn Prediction.parquet'
    input_file_path_csv = 'Bank Customer Churn Prediction.csv'
    dependent_var = 'churn'
    cols = pd.read_csv(input_file_path_csv, index_col=0, nrows=0).columns.tolist()
    cols.remove(dependent_var)
    selected_features,bins = woe.cal_woebin_iv(file_path= input_file_path_parquet,y=dependent_var,no_of_col_to_read_at_time=4,min_iv=0.1)
    print(len(selected_features))
    print(selected_features)
    df = pd.read_parquet(input_file_path_parquet,columns=selected_features)
    woe_encoded_df = woe.woe_encoding(df.copy(),bins_dict=bins)
    print(woe_encoded_df)
    one_hot_encoded_df = woe.one_hot_encoding(df=df.copy(),bins=bins)
    print(one_hot_encoded_df)