import sys
from datetime import datetime

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.mllib.feature import PCA

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types, Window

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, SQLTransformer, IndexToString
from pyspark.ml.regression import GBTRegressor

from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator

# Cassandra configs
cluster_seeds = ['199.60.17.103', '199.60.17.105']
sensor_config_table = "sensor_configs"
sensor_data_table = "sensor_data"
keyspace = "bba37"


def main():
    # sensor_data_df = spark.read.format("org.apache.spark.sql.cassandra").options(table=sensor_data_table,
    #                                                                              keyspace=keyspace).load()
    sensor_data_df = spark.read.parquet('sensor_data_ts')

    """ REGRESSION PROBLEM FOR GAS DATA PREDICTION"""
    regression_df = sensor_data_df.select(sensor_data_df['datetime'],
                                          sensor_data_df['H2S'],
                                          sensor_data_df['CO'],
                                          sensor_data_df['LEL'],
                                          sensor_data_df['O2'],
                                          sensor_data_df['message_code_name']).orderBy('datetime')

    day_df = regression_df.withColumn('date', regression_df['datetime'].cast(types.DateType()))
    # day_df.show()
    group_df = day_df.groupby(day_df['date']).agg(functions.max('H2S').alias('H2S_max'))

    # Window operation over date column where max(H2S) of the next day will be assinged to each row
    w = Window.partitionBy().orderBy('date')
    max_H2S_tmrw_df = group_df.withColumn('max_H2S_tmrw', functions.lead('H2S_max').over(w)).cache()
    max_H2S_tmrw_df.show()

    final_df = day_df.join(max_H2S_tmrw_df, 'date').repartition(100)
    final_df = final_df.withColumn('timestamp', functions.unix_timestamp(final_df['datetime'])) \
        .orderBy(final_df['datetime']).dropna()

    split_date = datetime(2020, 11, 1)
    train_set = final_df.where(functions.col('datetime') < split_date) \
        .select(final_df['CO'], final_df['LEL'], final_df['O2'], final_df['H2S'],final_df['H2S_max'], final_df['max_H2S_tmrw'])
    test_set = final_df.where(functions.col('datetime') >= split_date)\
        .select(final_df['CO'], final_df['LEL'], final_df['O2'], final_df['H2S'], final_df['H2S_max'], final_df['max_H2S_tmrw']).cache()

    x_train, x_val = final_df.randomSplit([0.75, 0.25])
    x_train = x_train.cache()
    x_val = x_val.cache()

    feature_assembler = VectorAssembler(inputCols=['H2S', 'CO', 'O2', 'LEL']
                                        , outputCol='features')
    # pca = PCA(k=10, inputCol="Features", outputCol="pcaFeatures")

    gbt = GBTRegressor(featuresCol='features', labelCol='max_H2S_tmrw')
    # dt = DecisionTreeRegressor(featuresCol='features', labelCol='max_H2S_tmrw')
    # rf = RandomForestRegressor(featuresCol='features', labelCol='max_H2S_tmrw')
    # fm = FMRegressor(featuresCol='features', labelCol='max_H2S_tmrw')



    pipeline = Pipeline(stages=[feature_assembler, gbt])
    # pipeline = Pipeline(stages=[feature_assembler, dt])
    # pipeline = Pipeline(stages=[feature_assembler, rf])
    # pipeline = Pipeline(stages=[feature_assembler, fm])

    model_fit = pipeline.fit(x_train)

    y_hat = model_fit.transform(x_val)

    rmse_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='max_H2S_tmrw', metricName='rmse')
    rmse_score = rmse_evaluator.evaluate(y_hat)
    print('rmse validation score: ', rmse_score)

    r2_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='max_H2S_tmrw', metricName='r2')
    r2_score = r2_evaluator.evaluate(y_hat)
    print('r2 validation score : ', r2_score)

    print(model_fit.stages[-1].featureImportances)
    # model_fit.write().overwrite().save('gbt_model')


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
