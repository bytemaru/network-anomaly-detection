====================================
Network Anomaly Detection with Spark
====================================

Authors: Mariia Fedorova, Harini Maragatha Meenakshi Murugan
Course: AIML427 — Big Data
Assignment 3 — Group Component

This project implements and evaluates two machine learning models — a Decision Tree and a Logistic Regression classifier — using Apache Spark on the KDD network intrusion dataset.

---

## Contents

- models/
  - decision_tree_kdd.py
  - logistic_regression_kdd.py
- data/
  - kdd.csv
  - kdd_newdata.csv (created after preprocessing for logistic_regression_kdd)


---

## 1. Prerequisites

These instructions assume you're working on the ECS Hadoop/Spark cluster (e.g., `co246a-1.ecs.vuw.ac.nz`) and that your environment has Spark and HDFS available.

You must also have:
- Uploaded `dataset.csv` to your HDFS folder. Uploaded python scripts to the same folder. We used scp to copy from our local machines, where these files are stored:

 ```console
foo@bar:~$ scp dataset.csv insert_correct_username@barretts.ecs.vuw.ac.nz:~
foo@bar:~$ scp decision_tree_kdd.py insert_correct_username@barretts.ecs.vuw.ac.nz:~
```

Transfer data file to the hadoop cluster. Shh to the cluster with HDFS and Spark:

 ```console
foo@bar:~$ ssh co246a-1
```

Check that env is configured:

```console
foo@bar:~$ source HadoopSetup.csh
foo@bar:~$ need java8
foo@bar:~$ hdfs dfs -mkdir -p /user/insert_correct_username/kdd
foo@bar:~$ hdfs dfs -put dataset.csv /user/insert_correct_username/kdd/
```
- For executing logistic_regression_kdd
Ensure the dataset and code are in the following structure:
~/prj_kdd/
├── models/
│ ├── decision_tree_kdd.py
│ └── logistic_regression_kdd.py
├── data/
│ ├── kdd.csv
---

## 2. Verify the dataset is successfully transferred:

```console
foo@bar:~$ hdfs dfs -ls /user/insert_correct_username/kdd
```

You shall see the dataset.csv if everything is correct.

Fix the path INSIDE the script decision_tree_kdd.py. Find the line:

```console
df = spark.read.csv("hdfs:///user/insert_correct_username/kdd/dataset.csv", header=False, inferSchema=True)
```

and insert valid path.

For executing logistic_regression_kdd, Copy files from your local system to the ECS cluster for :

scp -r /path/to/project/ maragahari@barretts.ecs.vuw.ac.nz:~/

## 3. Activate Spark env

Since it is not described in tutorial, we needed to activate Spark ourselves:

```console
foo@bar:~$ export SPARK_HOME=/local/spark
foo@bar:~$ export PATH=$SPARK_HOME/bin:$PATH
```

To verify that Spark is installed and working in the env:

```console
foo@bar:~$ spark-submit --version
```

If everything is up and running, you will see something like:

```console
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 3.5.1
      /_/

Using Scala version 2.12.18, Java HotSpot(TM) 64-Bit Server VM, 1.8.0_172
Branch HEAD
Compiled by user heartsavior on 2024-02-15T11:24:58Z
Revision fd86f85e181fc2dc0f50a096855acf83a6cc5d9c
Url https://github.com/apache/spark
Type --help for more information.
```

For executing logistic_regression_kdd, Make sure Spark is available and add it to your path:

export SPARK_HOME=/local/spark3/spark-3.5.1-bin-hadoop3
export PATH=$SPARK_HOME/bin:$PATH

## 4. Submit Spark job

```console
foo@bar:~$ spark-submit decision_tree_kdd.py
```
For executing logistic_regression_kdd, run the logistic regression training script

spark-submit models/logistic_regression_kdd.py

Make sure that the script points to the correct local file path:

base_dir = "/home/maragahari/prj_kdd/data"
input_file = os.path.join(base_dir, "kdd.csv")
cleaned_file = os.path.join(base_dir, "kdd_newdata.csv")
