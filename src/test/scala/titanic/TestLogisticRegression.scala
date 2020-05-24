package titanic

import org.apache.spark.sql
import org.apache.spark.sql._
import org.scalatest.{BeforeAndAfterEach, FunSuite}
import org.apache.spark.sql.types._
import raywu60kg.kaggles.titanic.TitanicLogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.functions._
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
    trainData.show()
    trainData.take(1)
    testData.show()

    assert(trainData.count() == 891)
    assert(testData.count() == 418)
  }
  test("Test one hot encoder") {
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
    val trainData = TitanicLogisticRegression.loadData(
      spark = spark,
      fileDir = "data/titanic/train.csv",
      scheme = trainSchema
    )

    // drop
    val parsedData = trainData.drop("PassengerId", "Name", "Ticket", "Cabin")
    parsedData.show()

    // null
    // val features = List("Pclass", "Age", "SibSP", "Parch", "Fare", "Embarked")
    // for (feature <- features) {
    //   parsedData.where(parsedData.col(feature).isNull).show()
    // }
    // feature.foreach()
    val averageAge = parsedData.select(mean(parsedData("Age"))).collectAsList().get(0).get(0)
    val parsedDate = parsedData.na.fill(1.0, Seq("Age"))

    parsedData.show()

    // one-hot
    assert(1 == 2)
  }
}
