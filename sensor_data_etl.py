import operator
import time, random, threading
import sys
import uuid
from datetime import datetime, timedelta

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types

# Cassandra configs
cluster_seeds = ['199.60.17.103', '199.60.17.105']
sensor_config_table = "sensor_configs"
sensor_data_table = "sensor_data"
keyspace = "bba37"

# SQL server configs
server_name = "jdbc:sqlserver://reportingprod.database.windows.net"
username = "trakadmin"
password = "T@l@m@t!cs_10"
connectionProperties = {
    "user": username,
    "password": password,
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}
sql_db_sensor_data = "TrakopolisDataProd_0013_CopyLH"
sql_db_sensor_configs = "TrakopolisProd_CopyLH"
url_sensor_data = "{0}; databaseName= {1};".format(server_name, sql_db_sensor_data)
url_sensor_configs = "{0}; databaseName= {1};".format(server_name, sql_db_sensor_configs)

# spark = SparkSession.builder.master('local[*]').appName('SQL_Cassandra') \
#     .config('spark.cassandra.connection.host', ','.join(cluster_seeds)) \
#     .getOrCreate()
spark = SparkSession.builder.master('local[*]').appName('SQL_polling').getOrCreate()
assert spark.version >= '2.4'  # make sure we have Spark 2.4+
sc = spark.sparkContext
sc.setLogLevel('WARN')

# Kafka configs
kafka_topic = "asset_data"

sensor_division_Id = 3550

sql_gas_sensor_data = """(SELECT
                            AssetData.TerminalID [terminal_id],
                            Terminal.TerminalTypeId [terminal_type_id],
                            Terminal.AssetID [asset_id],
                            AssetData.[ID] [asset_data_id], 
                            Asset.DivisionId [division_id],
                            AssetData.[Created] [datetime],
                            AssetData.[Latitude] [latitude], 
                            AssetData.[Longitude] [longitude],
                            AssetData.[MessageCodeID] [message_code_id],
                            AnalogSensorData.[value] [sensor_reading],
                            AnalogSensorData.[Sensor] [sensor_type_id],
                            SiteInfo.GeofenceID [site_id],
                            SiteInfo.IsEnter [site_transition_state],
                            SiteInfo.DateTime [site_transition_datetime]
                        FROM AssetData
                        INNER JOIN AnalogSensorData on AssetData.ID = AnalogSensorData.AssetDataID
                        LEFT JOIN SummaryGeofence SiteInfo ON SiteInfo.AssetDataID = AssetData.ID
                        INNER JOIN Terminal ON AssetData.TerminalID = Terminal.ID
                        INNER JOIN Asset ON Asset.ID = Terminal.AssetId
                        WHERE Terminal.TerminalTypeID IN (58, 49)
                            AND Asset.divisionId = {0}
                            AND AssetData.MessageCodeId IN (10, 100, 64, 66, 67, 68, 69)
                            AND AssetData.Created > '{1}'
                            AND AssetData.Created < '{2}') RawDataPoints"""

sql_sensor_config_query = """(SELECT
                                   Terminal.ID [terminal_id],
                                   Terminal.TerminalTypeId [terminal_type_id], 
                                   tas.Sensor [sensor_type_id], 
                                   tas.Name [sensor_name],
                                   tas.Units [sensor_unit]
                                 FROM Terminal
                                 INNER JOIN TerminalAnalogSensors tas ON tas.TerminalID = Terminal.ID
                                 WHERE Terminal.TerminalTypeId IN (58, 49)) Terminals"""


def main():
    sensor_config_df = spark.read.jdbc(url=url_sensor_configs,
                                       table=sql_sensor_config_query,
                                       numPartitions=3,
                                       properties=connectionProperties)
    sensor_config_df = sensor_config_df.select(sensor_config_df['terminal_id'],
                                               sensor_config_df['sensor_type_id'],
                                               sensor_config_df['sensor_name'],
                                               sensor_config_df['sensor_unit']).cache()

    # now = datetime.utcnow()
    now = datetime(2019, 3, 1)
    start_date = datetime(2019, 1, 1)
    interval = 10
    sensor_data_count_total = 0

    sensor_data_schema = types.StructType([])
    sensor_data_df = spark.createDataFrame(sc.emptyRDD(), schema=sensor_data_schema)

    # while start_date < now:
    end_date = start_date + timedelta(days=interval)
    new_data_points_df = spark.read.jdbc(url=url_sensor_data,
                                         table=sql_gas_sensor_data.format(sensor_division_Id,
                                                                          start_date.strftime("%Y-%m-%d"),
                                                                          end_date.strftime("%Y-%m-%d")),
                                         numPartitions=6,
                                         properties=connectionProperties)

    sensor_data_count = new_data_points_df.count()
    print("Polled {0} data points from {1} to {2}".format(sensor_data_count,
                                                          start_date.strftime("%Y-%m-%d"),
                                                          end_date.strftime("%Y-%m-%d")))
    sensor_data_count_total += sensor_data_count
    start_date = end_date

    sensor_data = new_data_points_df.join(sensor_config_df,
                                          (new_data_points_df['terminal_id'] == sensor_config_df['terminal_id'])
                                          & (new_data_points_df['sensor_type_id'] == sensor_config_df[
                                              'sensor_type_id']))

    sensor_reads_df = sensor_data_df.groupBy(sensor_data_df['asset_data_id'])\
        .agg(functions.collect_set())

    sensor_data.write.format("org.apache.spark.sql.cassandra") \
        .mode('append') \
        .options(**{'confirm.truncate': True}) \
        .options(table='sensor_data', keyspace="bba37").save()

if __name__ == "__main__":
    main()
