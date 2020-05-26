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

class TestLogisticRegression extends FunSuite {
  test("Test load data") {
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
  // test("Test one hot encoder") {
  //   val spark = SparkSession.builder
  //     .appName("Test-Titanic-Logistic-Regression")
  //     .master("local[*]")
  //     .getOrCreate()
  //   val trainSchema = StructType(
  //     Array(
  //       StructField("PassengerId", LongType, true),
  //       StructField("Survived", LongType, true),
  //       StructField("Pclass", StringType, true),
  //       StructField("Name", StringType, true),
  //       StructField("Sex", StringType, true),
  //       StructField("Age", FloatType, true),
  //       StructField("SibSP", LongType, true),
  //       StructField("Parch", LongType, true),
  //       StructField("Ticket", StringType, true),
  //       StructField("Fare", FloatType, true),
  //       StructField("Cabin", StringType, true),
  //       StructField("Embarked", StringType, true)
  //     )
  //   )
  //   val trainData = TitanicLogisticRegression.loadData(
  //     spark = spark,
  //     fileDir = "data/titanic/train.csv",
  //     scheme = trainSchema
  //   )

  //   // drop
  //   var parsedData = trainData.drop("PassengerId", "Name", "Ticket", "Cabin")
  //   // parsedData.show()

  //   // null
  //   val averageAge =
  //     parsedData.select(mean(parsedData("Age"))).first().getDouble(0)
  //   // println("@@@@@@@@", averageAge)
  //   parsedData = parsedData.na.fill(averageAge, Array("Age"))
  //   // parsedData.show()

  //   // one-hot
  //   val oneHotEncodeFeatures = Array("Pclass", "Sex", "Embarked")

  //   val indexers = oneHotEncodeFeatures.map(c =>
  //     new StringIndexer().setInputCol(c).setOutputCol(c + "_idx")
  //   )
  //   val encoders = oneHotEncodeFeatures.map(c =>
  //     new OneHotEncoderEstimator()
  //       .setInputCols(Array(c + "_idx"))
  //       .setOutputCols(Array(c + "_vec"))
  //   )

  //   // val encoders = new OneHotEncoderEstimator().setInputCols(oneHotEncodeFeatures).setOutputCols(oneHotEncodeFeatures.map(x => x+"Vec"))
  //   val pipeline = new Pipeline().setStages(indexers ++ encoders)

  //   var transformed = pipeline
  //     .fit(parsedData)
  //     .transform(parsedData)

  //   for (feature <- oneHotEncodeFeatures) {
  //     transformed = transformed.drop(feature)
  //     transformed = transformed.drop(feature + "_idx")
  //   }
  //   // .drop(oneHotEncodeFeatures.map(c=>c) ++ oneHotEncodeFeatures.map(c => c + "_idx"))
  //   transformed.show()
  //   spark.stop()
  //   assert(1 == 2)
  // }

  test("Test parseData") {
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
      spark = spark,
      trainData = trainData,
      testData = testData
    )
    parsedTrainData.show()
    parsedTestData.show()

    val trainColumns = parsedTrainData.columns.toSeq
    val testColumns = parsedTestData.columns.toSeq
    spark.stop()
    println("@@@@@", trainColumns, testColumns)
    assert(
      trainColumns == Array(
        "Survived",
        "Age",
        "SibSP",
        "Parch",
        "Fare",
        "Pclass_vec",
        "Sex_vec",
        "Embarked_vec"
      ).toSeq
    )
    assert(
      testColumns == Array(
        "Age",
        "SibSP",
        "Parch",
        "Fare",
        "Pclass_vec",
        "Sex_vec",
        "Embarked_vec"
      ).toSeq
    )
  }
}
