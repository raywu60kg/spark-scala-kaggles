package titanic

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame



object LogisticRegression {

  def main(args: Array[String]): Unit = {
    val trainDataDir = "../data/titanic/train.csv"
    val data = loadData(fileDir = trainDataDir)
    val parsedData = parseData(data)

  }

  def loadData(fileDir: String): DataFrame = {
    val spark = SparkSession
      .builder
      .appName("Spark CSV Reader")
      .master("local[*]")
      .getOrCreate()

    val df = spark.read
      .format("csv")
      .option("header", true)
      .load(fileDir)
    df
  }

  def parseData(data: DataFrame): DataFrame = {
    data
  }
}