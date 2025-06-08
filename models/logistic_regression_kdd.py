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

    predictions = model.transform(test_data)
    accuracy = evaluator.evaluate(predictions)
    runtime = round(end - start, 2)
    results.append((seed, accuracy, runtime))
    print(f"Run {i + 1} - Seed: {seed}, Accuracy: {accuracy:.4f}, Time: {runtime}s")

accuracies = [acc for _, acc, _ in results]
times = [t for _, _, t in results]

print("\n=== All Seed Results ===")
print(f"{'Run':<5} {'Seed':<6} {'Accuracy':<10} {'Time(s)':<8}")
for i, (seed, acc, runtime) in enumerate(results, start=1):
    print(f"{i:<5} {seed:<6} {acc:<10.4f} {runtime:<8.2f}")

print("\n=== Summary Metrics ===")
print(f"Max Accuracy: {max(accuracies):.4f}")
print(f"Min Accuracy: {min(accuracies):.4f}")
print(f"Avg Accuracy: {statistics.mean(accuracies):.4f}")
print(f"Std Dev Accuracy: {statistics.stdev(accuracies):.4f}")
print(f"Avg Runtime: {statistics.mean(times):.2f}s")

# Stop Spark session
spark.stop()
