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
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler

object TitanicLogisticRegression {
  private val trainSchema = StructType(
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

  private val testSchema = StructType(
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
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Titanic-Logistic-Regression")
      .master("local[*]")
      .getOrCreate()

    val trainDataDir = args(0)
    val testDataDir = args(1)
    val trainData =
      loadData(spark = spark, fileDir = trainDataDir, scheme = trainSchema)
    val testData =
      loadData(spark = spark, fileDir = testDataDir, scheme = trainSchema)
    // val parsedData = parseData(trainData)
    // val model = train()

  }

  def loadData(
      spark: SparkSession,
      fileDir: String,
      scheme: StructType
  ): (DataFrame) = {

    val df = spark.read
      .format("csv")
      .option("header", true)
      .schema(scheme)
      .load(fileDir)
    df

  }

  def parseData(
      spark: SparkSession,
      trainData: DataFrame,
      testData: DataFrame
  ): (DataFrame, DataFrame) = {
    var parsedTrainData = trainData
    var parsedTestData = testData
    val dropFeatures = List("PassengerId", "Name", "Ticket", "Cabin")
    val oneHotEncodeFeatures = Array("Pclass", "Sex", "Embarked")
    val featuresName = Array("Age", "SibSP", "Parch", "Fare", "Pclass_vec", "Sex_vec", "Embarked_vec")
    val averageAge =
      parsedTrainData.select(mean(parsedTrainData("Age"))).first().getDouble(0)

    // drop unuse columns
    for (feature <- dropFeatures) {
      parsedTrainData = parsedTrainData.drop(feature)
      parsedTestData = parsedTestData.drop(feature)
    }
    // fill null value
    parsedTrainData = parsedTrainData.na.fill(averageAge, Array("Age"))
    parsedTestData = parsedTestData.na.fill(averageAge, Array("Age"))

    // one hot encoder
    val indexers = oneHotEncodeFeatures.map(c =>
      new StringIndexer().setHandleInvalid("skip").setInputCol(c).setOutputCol(c + "_idx")
    )
    val encoders = oneHotEncodeFeatures.map(c =>
      new OneHotEncoderEstimator()
        .setInputCols(Array(c + "_idx"))
        .setOutputCols(Array(c + "_vec"))
    )
    val assembler = new VectorAssembler()
      .setInputCols(featuresName)
      .setOutputCol("features")
    
    val pipeline = new Pipeline().setStages(indexers ++ encoders ++ Array(assembler))
    var transformedTrainData = pipeline
      .fit(parsedTrainData)
      .transform(parsedTrainData)
    var transformedTestData = pipeline
      .fit(parsedTrainData)
      .transform(parsedTestData)
    
    // cleanup features
    // for (feature <- oneHotEncodeFeatures) {
    //   transformedTrainData = transformedTrainData.drop(feature)
    //   transformedTrainData = transformedTrainData.drop(feature + "_idx")

    //   transformedTestData = transformedTestData.drop(feature)
    //   transformedTestData = transformedTestData.drop(feature + "_idx")
    // }
    
    
    transformedTrainData = transformedTrainData.select(col("Survived").as("label"), col("features"))
    transformedTestData = transformedTestData.select("features")
    
    // rename label
    // transformedTrainData.withColumn("Survived", col("label")).

    (transformedTrainData, transformedTestData)
  }

  // def train(trainData: DataFrame, testData: DataFrame): Unit = {
  //   val lr = new LogisticRegressionModel()
  //     .setMaxIter(10)
  //     .setRegParam()
  //     .setElasticNetParam(0.8)
  // }
}
