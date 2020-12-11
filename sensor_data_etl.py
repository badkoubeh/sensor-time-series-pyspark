import sys

from datetime import datetime, timedelta

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types

# Cassandra configs
cluster_seeds = ['199.60.17.103', '199.60.17.105']
sensor_config_table = "sensor_configs"
sensor_data_table = "sensor_data"
keyspace = "bba37"

# spark = SparkSession.builder.master('local[*]').appName('SQL_Cassandra') \
#     .config('spark.cassandra.connection.host', ','.join(cluster_seeds)) \
#     .getOrCreate()
spark = SparkSession.builder.appName('sensor data ETL').getOrCreate()
assert spark.version >= '2.4'  # make sure we have Spark 2.4+
sc = spark.sparkContext
sc.setLogLevel('WARN')


def main():
    sensor_config_df = spark.read.parquet('sensor_configs')
    message_code_df = spark.read.parquet('message_code_table')
    sensor_config_df = sensor_config_df.select(sensor_config_df['terminal_id'],
                                               sensor_config_df['sensor_type_id'],
                                               sensor_config_df['sensor_name'],
                                               sensor_config_df['sensor_unit']).repartition(50)
    sensor_config_df = sensor_config_df.withColumn('sensor_name',
                                                   functions.when(functions.col('sensor_name') == 'CO ', 'CO')
                                                   .otherwise(functions.col('sensor_name')))

    raw_data_schema = types.StructType([
        types.StructField('terminal_id', types.LongType()),
        types.StructField('terminal_type_id', types.IntegerType()),
        types.StructField('asset_id', types.IntegerType()),
        types.StructField('asset_data_id', types.LongType()),
        types.StructField('division_id', types.IntegerType()),
        types.StructField('datetime', types.TimestampType()),
        types.StructField('latitude', types.FloatType()),
        types.StructField('longitude', types.FloatType()),
        types.StructField('message_code_id', types.IntegerType()),
        types.StructField('sensor_reading', types.FloatType()),
        types.StructField('sensor_type_id', types.IntegerType()),
        types.StructField('site_id', types.IntegerType()),
        types.StructField('site_transition_state', types.BooleanType()),
        types.StructField('site_transition_datetime', types.TimestampType())])

    new_data_points_df = spark.read.csv('sensor_raw_data', schema=raw_data_schema).repartition(200)

    # Filter data for testing
    # new_data_points_df = new_data_points_df.where(new_data_points_df['datetime'] < datetime(2019, 1, 6))

    # Filter outlier data around one operation site
    new_data_points_df = new_data_points_df.where((new_data_points_df['latitude'] < 28.7)
                                                  & (new_data_points_df['longitude'] > -98.8))

    sensor_data = new_data_points_df.join(sensor_config_df,
                                          (new_data_points_df['terminal_id'] == sensor_config_df['terminal_id'])
                                          & (new_data_points_df['sensor_type_id'] == sensor_config_df[
                                              'sensor_type_id']))
    print("No. Records after join operation: ", sensor_data.count())
    drop_extra_cols = sensor_data.select(sensor_data['asset_data_id'],
                                         sensor_data['datetime'],
                                         sensor_data['latitude'],
                                         sensor_data['longitude'],
                                         sensor_data['message_code_id'],
                                         sensor_data['sensor_name'],
                                         sensor_data['sensor_reading']).orderBy(sensor_data['asset_data_id'])

    target_df = drop_extra_cols.groupBy(drop_extra_cols['datetime'],
                                        drop_extra_cols['latitude'],
                                        drop_extra_cols['longitude']) \
        .agg(functions.max(drop_extra_cols['message_code_id']).alias('message_code_id'))

    features_df = drop_extra_cols.groupBy(drop_extra_cols['datetime'],
                                          drop_extra_cols['latitude'],
                                          drop_extra_cols['longitude']).pivot('sensor_name') \
        .agg(functions.first(drop_extra_cols['sensor_reading'], ignorenulls=True))

    features_df = features_df.fillna(0, subset=['CO', 'LEL', 'H2S'])
    features_df = features_df.fillna(21, subset=['O2'])

    output = features_df.join(target_df, ['datetime', 'latitude', 'longitude'])

    output = output.join(functions.broadcast(message_code_df), 'message_code_id')

    # # dropping some columns
    output = output.drop('Battery level')
    output = output.drop('Gas Concentration #6')

    print("Total records: ", output.count())
    # output.where(output['message_code_id'].isin([66, 67, 68, 69, 100])).show(30)
    output.write.parquet('sensor_data_ts')
    # sensor_data.write.format("org.apache.spark.sql.cassandra") \
    #     .mode('append') \
    #     .options(**{'confirm.truncate': True}) \
    #     .options(table='sensor_data', keyspace="bba37").save()


if __name__ == "__main__":
    main()
