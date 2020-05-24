package titanic

import org.apache.spark.sql
import org.apache.spark.sql._
import org.scalatest.{BeforeAndAfterEach, FunSuite}
import org.apache.spark.sql.types._
import raywu60kg.kaggles.titanic.TitanicLogisticRegression

class TestLogisticRegression extends FunSuite {
  test("Test load data") {
    val spark = SparkSession.builder
      .appName("Test-Titanic-Logistic-Regression")
      .master("local[*]")
      .getOrCreate()
    val trainSchema = StructType(
      Array(
        StructField("PassengerId", StringType, true),
        StructField("Survived", StringType, true),
        StructField("Pclass", StringType, true),
        StructField("Name", StringType, true),
        StructField("Sex", StringType, true),
        StructField("Age", StringType, true),
        StructField("SibSP", StringType, true),
        StructField("Parch", StringType, true),
        StructField("Ticket", StringType, true),
        StructField("Fare", StringType, true),
        StructField("Cabin", StringType, true),
        StructField("Embarked", StringType, true)
      )
    )
    val data = TitanicLogisticRegression.loadData(
      spark = spark,
      fileDir = "data/titanic",
      scheme = trainSchema
    )
    println("@@@@@", data.show())
    assert(data.count() == 1727)
  }
}
