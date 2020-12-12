import sys
from datetime import datetime

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, SQLTransformer, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier

# Cassandra configs
cluster_seeds = ['199.60.17.103', '199.60.17.105']
sensor_config_table = "sensor_configs"
sensor_data_table = "sensor_data"
keyspace = "bba37"


def main():
    sensor_data_df = spark.read.parquet('sensor_data_ts')

    # creating ML pipelines for classification and regression problems
    sensor_data_df = sensor_data_df.select(sensor_data_df['datetime'],
                                           sensor_data_df['message_code_name'],
                                           sensor_data_df['H2S'],
                                           sensor_data_df['CO'],
                                           sensor_data_df['LEL'],
                                           sensor_data_df['O2']).orderBy(sensor_data_df['datetime'].asc())

    condition = (sensor_data_df['datetime'] < datetime(2020, 10, 1))
    train_df = sensor_data_df.where(condition)
    test_df = sensor_data_df.where(~condition).cache()

    """ CLASSIFICATION PROBLEM FOR GAS EVENT IDENTIFICATION"""
    classification_set = train_df.select(train_df['message_code_name'].alias('target'),
                                         train_df['H2S'],
                                         train_df['CO'],
                                         train_df['LEL'],
                                         train_df['O2'])

    train_set, validation_set = classification_set.randomSplit([0.75, 0.25])
    train_set.cache()
    validation_set.cache()

    # message_code_name is the target column representing gas events (Normal, GasHighAlarm, GasLowAlarm, GasAlarm)
    # sql_transformer_statement = "SELECT datetime ,H2S, LEL, O2, CO, message_code_name AS target " \
    # #                               "FROM __THIS__"

    # sql_transformer = SQLTransformer(statement=sql_transformer_statement_2)
    assemble_features = VectorAssembler(inputCols=['H2S', 'LEL', 'O2', 'CO']
                                        , outputCol='features')
    target_indexer = StringIndexer(inputCol='target', outputCol='label').fit(train_set)
    label_converter = IndexToString(inputCol='label', outputCol='predicted_target')
    print(target_indexer.labels)
    classifier = MultilayerPerceptronClassifier(layers=[4, 20, 4])
    pipeline = Pipeline(stages=[assemble_features, target_indexer, classifier, label_converter])
    model = pipeline.fit(train_set)

    predictions = model.transform(validation_set)
    predictions.select(['target', 'predicted_target']).show(30)

    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label')
    prediction_score = evaluator.evaluate(predictions)
    print('Gas Events Prediction Score: ', prediction_score)


if __name__ == '__main__':
    # cluster_seeds = ['199.60.17.103', '199.60.17.105']
    # spark = SparkSession.builder.master('local[*]').appName('sensor data train') \
    #     .config('spark.cassandra.connection.host', ','.join(cluster_seeds)) \
    #     .getOrCreate()
    spark = SparkSession.builder.appName('sensor data training').getOrCreate()
    assert spark.version >= '2.4'  # make sure we have Spark 2.4+
    spark.sparkContext.setLogLevel('WARN')
    # inputs = sys.argv[1]
    # model_file = sys.argv[2]
    main()
