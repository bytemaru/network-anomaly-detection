====================================
Network Anomaly Detection with Spark
====================================

Authors: Mariia Fedorova, Harini Maragatha Meenakshi Murugan
Course: AIML427 — Big Data
Assignment 3 — Group Component

This project implements and evaluates two machine learning models — a Decision Tree and a Logistic Regression classifier — using Apache Spark on the KDD network intrusion dataset.

---

## Contents

- decision_tree_kdd.py
- logistic_regression_kdd.py

---

## 1. Prerequisites

These instructions assume you're working on the ECS Hadoop/Spark cluster (e.g., `co246a-1.ecs.vuw.ac.nz`) and that your environment has Spark and HDFS available.

You must also have:
- Uploaded `dataset.csv` to your HDFS folder. Uploaded python scripts to the same folder. We used scp to copy from our local machines:

 ```console
foo@bar:~$ scp dataset.csv insert_correct_username@barretts.ecs.vuw.ac.nz:~
foo@bar:~$ scp decision_tree_kdd.py insert_correct_username@barretts.ecs.vuw.ac.nz:~
```

Transfer data file to the hadoop cluster. Check that env is configured:

```console
foo@bar:~$ source HadoopSetup.csh
foo@bar:~$ need java8
foo@bar:~$ hdfs dfs -mkdir -p /user/insert_correct_username/kdd
foo@bar:~$ hdfs dfs -put dataset.csv /user/insert_correct_username/kdd/
```

---

## 2. Verify the dataset is successfully transferred:

```console
foo@bar:~$ hdfs dfs -ls /user/insert_correct_username/kdd
```

You shall see the dataset.csv if everything is correct.

Fix the path INSIDE the script. Find the line:

```console
df = spark.read.csv("hdfs:///user/insert_correct_username/kdd/dataset.csv", header=False, inferSchema=True)
```

and insert valid path.

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

## 4. Submit Spark job

```console
foo@bar:~$ spark-submit decision_tree_kdd.py
```
