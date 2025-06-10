import time
import random
import numpy as np

from pyspark import SparkFiles

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder.appName("Decision Tree Model").getOrCreate()


df = spark.read.csv("here  goes the path", header=False, inferSchema=True)

# Step 1: Convert label (string) to numeric
label_indexer = StringIndexer(inputCol="_c41", outputCol="label")
data = label_indexer.fit(df).transform(df)

# Step 2: Assemble vector from features
num_features = 40
assembler = VectorAssembler(inputCols=[f"_c{i}"for i in range(num_features)], outputCol="features")
data = assembler.transform(data)

test_accuracies = []
train_accuracies = []
runtimes = []

for i in range(10):
    start_time = time.time()

    seed = random.randint(1, 9999)

    # Step 3: Split data into train and test data
    train_data, test_data = data.randomSplit([0.7, 0.3], seed=seed)

    # Step 4: Create the classifier
    dt_classifier = DecisionTreeClassifier(labelCol="label", featuresCol="features")

    # Step 5: train the model
    model = dt_classifier.fit(train_data)
    end_time = time.time()

    # Step 6: test the model
    test_predictions = model.transform(test_data)
    train_predictions = model.transform(train_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    test_accuracy = evaluator.evaluate(test_predictions)
    train_accuracy = evaluator.evaluate(train_predictions)

    # Step 7: Store looped data
    test_accuracies.append(test_accuracy)
    train_accuracies.append(train_accuracy)
    runtimes.append(end_time - start_time)

    print(f"Run {i + 1} - Seed: {seed}")
    print(f"   Train Accuracy: {train_accuracy:.4f}")
    print(f"   Test Accuracy:  {test_accuracy:.4f}")
    print(f"   Train Time:     {end_time - start_time:.2f}s\n")

# Step 8: Calculating min, max, avg and st deviation + avg runtime
print("\n=== Summary of 10 Runs ===")
print(f"Test Accuracy: min = {np.min(test_accuracies):.4f}, max = {np.max(test_accuracies):.4f}, "
      f"mean = {np.mean(test_accuracies):.4f}, std = {np.std(test_accuracies):.4f}")
print(f"Train Accuracy: min = {np.min(train_accuracies):.4f}, max = {np.max(train_accuracies):.4f}, "
      f"mean = {np.mean(train_accuracies):.4f}, std = {np.std(train_accuracies):.4f}")
print(f"Avg Runtime: {np.mean(runtimes):.2f} sec")
