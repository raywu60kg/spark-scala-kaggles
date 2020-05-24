package raywu60kg.kaggles.titanic

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.classification.{
  LogisticRegressionModel,
  LogisticRegressionWithLBFGS
}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types._

object TitanicLogisticRegression {
  private val trainSchema = StructType(
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

  private val testSchema = StructType(
    Array(
      StructField("PassengerId", StringType, true),
      StructField("Survived", StringType, false),
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
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Titanic-Logistic-Regression")
      .master("local[*]")
      .getOrCreate()

    val trainDataDir = args(0)
    val testDataDir = args(1)
    val trainData = loadData(
      spark=spark,
      fileDir=trainDataDir,
      scheme = trainSchema)
    val testData = loadData(
      spark=spark,
      fileDir=testDataDir,
      scheme=trainSchema)
    // val parsedData = parseData(trainData)
    // val model = train()

  }

  def loadData(
      spark: SparkSession,
      fileDir: String,
      scheme: StructType
  ): DataFrame = {

    val df = spark.read
      .format("csv")
      .option("header", true)
      .schema(scheme)
      .load(fileDir)
    df

  }

  // def parseData(data: DataFrame): DataFrame = {
  //   data
  // }

  // def train(data: DataFrame): Unit = { 1 }
}
