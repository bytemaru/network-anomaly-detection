import os
import pandas as pd
import time
import random
import statistics
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

base_dir = "/home/maragahari/prj_kdd/data"
input_file = os.path.join(base_dir, 'kdd.csv')
cleaned_file = os.path.join(base_dir, 'kdd_newdata.csv')

col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

raw_df = pd.read_csv(input_file, header=None)
valid_df = raw_df[~raw_df.apply(lambda x: set(x).issubset(set(col_names)), axis=1)]
valid_df.columns = col_names
valid_df = valid_df.reset_index(drop=True)

features_only = valid_df.columns.difference(['label'])
valid_df[features_only] = valid_df[features_only].apply(pd.to_numeric, errors='coerce')

os.makedirs(base_dir, exist_ok=True)
valid_df.to_csv(cleaned_file, index=False)

spark = SparkSession.builder.appName("KDD_LogisticRegression").getOrCreate()
df = spark.read.csv(f"file://{cleaned_file}", header=True, inferSchema=True)

df = df.withColumn("label", when(col("label") == "normal", 0).otherwise(1))

categorical_cols = ["protocol_type", "service", "flag"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_vec") for col in categorical_cols]

numerical_cols = [c for c in df.columns if c not in categorical_cols + ["label"]]
final_features = [col + "_vec" for col in categorical_cols] + numerical_cols
assembler = VectorAssembler(inputCols=final_features, outputCol="features")

lr = LogisticRegression(featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

results = []

for i in range(10):
    seed = random.randint(1, 9999)
    train_data, test_data = df.randomSplit([0.7, 0.3], seed=seed)

    start = time.time()
    model = pipeline.fit(train_data)
    end = time.time()
    train_time = round(end - start, 2)

    train_predictions = model.transform(train_data)
    test_predictions = model.transform(test_data)

    train_accuracy = evaluator.evaluate(train_predictions)
    test_accuracy = evaluator.evaluate(test_predictions)

    results.append((seed, train_accuracy, test_accuracy, train_time))

    print(f"Run {i + 1} - Seed: {seed}")
    print(f"   Train Accuracy: {train_accuracy:.4f}")
    print(f"   Test Accuracy:  {test_accuracy:.4f}")
    print(f"   Train Time:     {train_time}s\n")

train_accuracies = [acc for _, acc, _, _ in results]
test_accuracies = [acc for _, _, acc, _ in results]
train_times = [t for _, _, _, t in results]

print("\n=== Per-Run Results ===")
print(f"{'Run':<5} {'Seed':<6} {'Train Acc':<10} {'Test Acc':<10} {'Time(s)':<8}")
for i, (seed, train_acc, test_acc, t) in enumerate(results, 1):
    print(f"{i:<5} {seed:<6} {train_acc:<10.4f} {test_acc:<10.4f} {t:<8.2f}")

print("\n=== Summary Metrics ===")
print(f"Train Accuracy - Max: {max(train_accuracies):.4f}, Min: {min(train_accuracies):.4f}, "
      f"Avg: {statistics.mean(train_accuracies):.4f}, Std Dev: {statistics.stdev(train_accuracies):.4f}")

print(f"Test Accuracy  - Max: {max(test_accuracies):.4f}, Min: {min(test_accuracies):.4f}, "
      f"Avg: {statistics.mean(test_accuracies):.4f}, Std Dev: {statistics.stdev(test_accuracies):.4f}")

print(f"Average Training Time: {statistics.mean(train_times):.2f}s")

spark.stop()
