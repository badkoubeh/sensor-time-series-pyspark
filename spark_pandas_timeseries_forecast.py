import sys
import warnings
from datetime import datetime, timedelta
import itertools
import pandas as pd
import numpy as np
from pyspark.sql.functions import pandas_udf, PandasUDFType
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

# import statsmodels.api as sm

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('sensor data ETL').getOrCreate()
assert spark.version >= '2.4'  # make sure we have Spark 2.4+
sc = spark.sparkContext
sc.setLogLevel('WARN')

schema = types.StructType([
    types.StructField("datetime", types.TimestampType(), True),
    types.StructField("O2", types.FloatType(), True)
])


@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def forecast_udf(pd_df):
    ts = pd_df.set_index(pd.DatetimeIndex(pd_df['datetime']), drop=True)
    ts_train = ts.loc[ts.index < datetime(2020, 11, 1)]
    ts_val = ts.loc[ts.index >= datetime(2020, 11, 1)]

    ts_train_col = ts_train['O2'].resample('30T').mean().interpolate(method='linear')
    ts_val_col = ts_val['O2'].resample('30T').mean().interpolate(method='linear')

    ts_train_col = ts_train_col[ts_train_col.between(ts_train_col.quantile(.2), ts_train_col.quantile(.8))]

    # model_fit = ARIMA(ts_train_col, order=(1, 1, 1)).fit()
    model_fit = ExponentialSmoothing(ts_train_col, trend='add').fit()
    # model_fit = Holt(ts_train_col).fit()

    num_forecast_steps = ts_val_col.count()
    forecast_res, stderr, conf_int = model_fit.forecast(num_forecast_steps)

    forecast_series = pd.Series(forecast_res, index=ts_val_col.index)
    # print(model_fit.summary())

    output = pd.DataFrame({'O2': forecast_series})
    return output


def main():
    sensor_data_df = spark.read.parquet('sensor_data_ts')
    sensor_data_df = sensor_data_df.select(sensor_data_df['datetime'],
                                           sensor_data_df['H2S'],
                                           sensor_data_df['CO'],
                                           sensor_data_df['LEL'],
                                           sensor_data_df['O2']).where(
        sensor_data_df['datetime'] > datetime(2020, 10, 1))
    # train_df = sensor_data_df.where(sensor_data_df['datetime'] < datetime(2020, 10, 1))

    forecast_df = sensor_data_df.groupby('datetime', 'O2').apply(forecast_udf)
    forecast_df.show()


if __name__ == "__main__":
    main()
