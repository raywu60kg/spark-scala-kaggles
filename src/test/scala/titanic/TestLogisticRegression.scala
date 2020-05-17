package titanic

import org.apache.spark.sql
import org.apache.spark.sql._
import org.scalatest.{BeforeAndAfterEach, FunSuite}
import titanic.LogisticRegression

class TestLogisticRegression extends FunSuite with BeforeAndAfterEach{
  private val master = "local[*]"
  private val appName = "TestLogisticRegression"

  override def beforeEach(): Unit = {
    var spark = new sql.SparkSession
      .Builder()
      .appName(appName)
      .master(master)
      .getOrCreate()

  }
}
