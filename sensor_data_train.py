import sys

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.regression import GBTRegressor

from pyspark.ml.evaluation import RegressionEvaluator

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

def main(inputs, model_file):
    sensor_data_df = spark.read.format("org.apache.spark.sql.cassandra").options(table=sensor_data_table,
                                                                                 keyspace=keyspace).load()
    # creating a ML pipeline

    sensor_data_df = sensor_data_df.select(sensor_data_df['datetime'],
                                         sensor_data_df['latitude'],
                                         sensor_data_df['longitude'],
                                         sensor_data_df['message_code_id'],
                                         sensor_data_df['sensor_reading'],
                                         sensor_data_df['sensor_name']).orderBy(sensor_data_df['datetime'].asc())
    train_set, validation_set = sensor_data_df.randomSplit([0.75, 0.25])
    train_set.catch()
    validation_set.catch()
    sql_transformer_statement = "SELECT latitude, longitude, sensor_name, sensor_reading, message_code_id" \
                                 "FROM __THIS__"

    sql_transformer = SQLTransformer(statement=sql_transformer_statement)
    assemble_features = VectorAssembler(inputCols=['latitude', 'longitude', 'sensor_name', 'sensor_reading']
                                        , outputCol= 'features')
    classifier = GBTRegressor(featuresCol='features', labelCol='message_code_id')
    pipeline = Pipeline(stages=[sql_transformer, assemble_features, classifier])
    model = pipeline.fit(train_set)

    predictions = model.tranform(validation_set)
    predictions.show()

    r2_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='message_code_id', metricName='r2')
    rmse_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='message_code_id', metricName='rmse')
    r2_score = r2_evaluator.evaluate(predictions)
    rmse_score = rmse_evaluator.evaluate(predictions)
    print('r2 validation score : ', r2_score)
    print('rmse validation score: ', rmse_score)


if __name__ == '__main__':
    cluster_seeds = ['199.60.17.103', '199.60.17.105']
    spark = SparkSession.builder.master('local[*]').appName('sensor data train') \
        .config('spark.cassandra.connection.host', ','.join(cluster_seeds)) \
        .getOrCreate()
    assert spark.version >= '2.4'  # make sure we have Spark 2.4+
    spark.sparkContext.setLogLevel('WARN')
    inputs = sys.argv[1]
    model_file = sys.argv[2]
    main(inputs, model_file)