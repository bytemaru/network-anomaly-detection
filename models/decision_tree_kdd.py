import time

import findspark
findspark.init()

from pyspark import SparkFiles

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder.appName("Decision Tree Model").getOrCreate()

spark.sparkContext.addFile("dataset.csv")

df = spark.read.csv(SparkFiles.get("dataset.csv"), header=False, inferSchema=True)

# Step 1: Convert label (string) to numeric
label_indexer = StringIndexer(inputCol="_c41", outputCol="label")
data = label_indexer.fit(df).transform(df)

# Step 2: Assemble vector from features
num_features = 40
assembler = VectorAssembler(inputCols=[f"_c{i}"for i in range(num_features)], outputCol="features")
data = assembler.transform(data)

accuracies = []
runtimes = []

for i in range(10):
    start_time = time.time()

    # Step 3: Split data into train and test data
    train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)

    # Step 4: Create the classifier
    dt_classifier = DecisionTreeClassifier(labelCol="label", featuresCol="features")

    # Step 5: train the model
    model = dt_classifier.fit(train_data)

    # Step 6: test the model
    predictions = model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    end_time = time.time()

    # Step 7: Store looped data
    accuracies.append(accuracy)
    runtimes.append(end_time - start_time)

    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Runtime: {end_time - start_time:.2f}")
