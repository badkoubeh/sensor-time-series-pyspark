import sys

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types

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

spark = SparkSession.builder.master('local[*]').appName('SQL_polling').getOrCreate()
assert spark.version >= '2.4'  # make sure we have Spark 2.4+
sc = spark.sparkContext
sc.setLogLevel('WARN')

sql_sensor_config_query = """(SELECT
                                   Terminal.ID [terminal_id],
                                   Terminal.TerminalTypeId [terminal_type_id], 
                                   tas.Sensor [sensor_type_id], 
                                   tas.Name [sensor_name],
                                   tas.Units [sensor_unit]
                                 FROM Terminal
                                 INNER JOIN TerminalAnalogSensors tas ON tas.TerminalID = Terminal.ID
                                 WHERE Terminal.TerminalTypeId IN (58, 49)) Terminals"""

sql_message_code_query = """(SELECT 
                                ID [message_code_id],
                                Name [message_code_name]
                            FROM MessageCode
                            WHERE ID IN (64, 66, 67, 68,69, 10, 100)) MessageCodes"""


def main():
    sensor_config_df = spark.read.jdbc(url=url_sensor_configs,
                                       table=sql_sensor_config_query,
                                       numPartitions=20,
                                       properties=connectionProperties)
    sensor_config_df = sensor_config_df.select(sensor_config_df['terminal_id'],
                                               sensor_config_df['sensor_type_id'],
                                               sensor_config_df['sensor_name'],
                                               sensor_config_df['sensor_unit']).repartition(20)
    print("No Records: ", sensor_config_df.count())
    sensor_config_df.write.parquet('sensor_configs')

    message_code_df = spark.read.jdbc(url=url_sensor_configs,
                                      table=sql_message_code_query,
                                      numPartitions=1,
                                      properties=connectionProperties)
    message_code_df = message_code_df.withColumn('message_code_name',
                                                 functions.when((functions.col('message_code_name') == 'SensorMsg') |
                                                                (functions.col(
                                                                    'message_code_name') == 'DeviceStatusReport'),
                                                                'Normal')
                                                 .otherwise(functions.col('message_code_name')))
    message_code_df.show()
    message_code_df.write.parquet('message_code_table', mode='overwrite')


if __name__ == "__main__":
    main()
