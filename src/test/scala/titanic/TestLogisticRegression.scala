package titanic

import org.apache.spark.sql
import org.apache.spark.sql._
import org.scalatest.{BeforeAndAfterEach, FunSuite}
import org.apache.spark.sql.types._
import raywu60kg.kaggles.titanic.TitanicLogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import scala.math.max

class TestLogisticRegression extends FunSuite {
  val spark = SparkSession.builder
    .appName("Test-Titanic-Logistic-Regression")
    .master("local[*]")
    .getOrCreate()
  val trainSchema = StructType(
    Array(
      StructField("PassengerId", LongType, true),
      StructField("Survived", LongType, true),
      StructField("Pclass", StringType, true),
      StructField("Name", StringType, true),
      StructField("Sex", StringType, true),
      StructField("Age", FloatType, true),
      StructField("SibSP", LongType, true),
      StructField("Parch", LongType, true),
      StructField("Ticket", StringType, true),
      StructField("Fare", FloatType, true),
      StructField("Cabin", StringType, true),
      StructField("Embarked", StringType, true)
    )
  )
  val testSchema = StructType(
    Array(
      StructField("PassengerId", LongType, true),
      StructField("Pclass", StringType, true),
      StructField("Name", StringType, true),
      StructField("Sex", StringType, true),
      StructField("Age", FloatType, true),
      StructField("SibSP", LongType, true),
      StructField("Parch", LongType, true),
      StructField("Ticket", StringType, true),
      StructField("Fare", FloatType, true),
      StructField("Cabin", StringType, true),
      StructField("Embarked", StringType, true)
    )
  )
  test("Test load data") {
    val spark = SparkSession.builder
      .appName("Test-Titanic-Logistic-Regression")
      .master("local[*]")
      .getOrCreate()
    val trainData = TitanicLogisticRegression.loadData(
      spark = spark,
      fileDir = "data/titanic/train.csv",
      scheme = trainSchema
    )
    val testData = TitanicLogisticRegression.loadData(
      spark = spark,
      fileDir = "data/titanic/test.csv",
      scheme = testSchema
    )
    assert(trainData.count() == 891)
    assert(testData.count() == 418)
    spark.stop()
  }

  test("Test parseData") {
    val spark = SparkSession.builder
      .appName("Test-Titanic-Logistic-Regression")
      .master("local[*]")
      .getOrCreate()
    val trainData = TitanicLogisticRegression.loadData(
      spark = spark,
      fileDir = "data/titanic/train.csv",
      scheme = trainSchema
    )
    val testData = TitanicLogisticRegression.loadData(
      spark = spark,
      fileDir = "data/titanic/test.csv",
      scheme = testSchema
    )

    val (parsedTrainData, parsedTestData) = TitanicLogisticRegression.parseData(
      trainData = trainData,
      testData = testData
    )
    parsedTrainData.show()
    parsedTestData.show()

    val trainColumns = parsedTrainData.columns.toSeq
    val testColumns = parsedTestData.columns.toSeq
    assert(
      trainColumns == Array(
        "label",
        "features"
      ).toSeq
    )
    assert(
      testColumns == Array(
        "features"
      ).toSeq
    )
    spark.stop()
  }

  test("Test train") {
    // Load training data
    val spark = SparkSession.builder
      .appName("Test-Titanic-Logistic-Regression")
      .master("local[*]")
      .getOrCreate()

    val trainData = TitanicLogisticRegression.loadData(
      spark = spark,
      fileDir = "data/titanic/train.csv",
      scheme = trainSchema
    )
    val testData = TitanicLogisticRegression.loadData(
      spark = spark,
      fileDir = "data/titanic/test.csv",
      scheme = testSchema
    )

    val (parsedTrainData, parsedTestData) = TitanicLogisticRegression.parseData(
      trainData = trainData,
      testData = testData
    )

    val prediction = TitanicLogisticRegression.trainAndPredict(
      trainData = parsedTrainData,
      testData = parsedTestData
    )
    prediction.show()
    assert(prediction.count() == 418)
    spark.stop()
  }
  test("Test write to csv") {
    // Load training data
    val spark = SparkSession.builder
      .appName("Test-Titanic-Logistic-Regression")
      .master("local[*]")
      .getOrCreate()

    val trainData = TitanicLogisticRegression.loadData(
      spark = spark,
      fileDir = "data/titanic/train.csv",
      scheme = trainSchema
    )
    val testData = TitanicLogisticRegression.loadData(
      spark = spark,
      fileDir = "data/titanic/test.csv",
      scheme = testSchema
    )

    val (parsedTrainData, parsedTestData) = TitanicLogisticRegression.parseData(
      trainData = trainData,
      testData = testData
    )

    val prediction = TitanicLogisticRegression.trainAndPredict(
      trainData = parsedTrainData,
      testData = parsedTestData
    )

    val res = TitanicLogisticRegression.write2CSV(
      prediction = prediction,
      testData = testData,
      outputDir = "/tmp/submit",
      isWrite = false
    )
    res.show()

    assert(res.count() == 418)
    spark.stop()
  }

}
