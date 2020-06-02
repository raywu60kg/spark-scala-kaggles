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
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.Model

object TitanicLogisticRegression {
  private val trainSchema = StructType(
    Array(
      StructField("PassengerId", LongType, false),
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
      StructField("PassengerId", LongType, false),
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
    import spark.implicits._
    val trainDataDir = args(0)
    val testDataDir = args(1)
    val outputDataDir = args(2)
    val trainData =
      loadData(spark = spark, fileDir = trainDataDir, scheme = trainSchema)
    val testData =
      loadData(spark = spark, fileDir = testDataDir, scheme = trainSchema)
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
      outputDir = outputDataDir,
      isWrite = true
    )

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
      trainData: DataFrame,
      testData: DataFrame
  ): (DataFrame, DataFrame) = {
    var parsedTrainData = trainData
    var parsedTestData = testData
    val dropFeatures = List("PassengerId", "Name", "Ticket", "Cabin")
    val oneHotEncodeFeatures = Array("Pclass", "Sex", "Embarked")
    val featuresName = Array(
      "Age",
      "SibSP",
      "Parch",
      "Fare",
      "Pclass_vec",
      "Sex_vec",
      "Embarked_vec"
    )
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
      new StringIndexer()
        .setHandleInvalid("skip")
        .setInputCol(c)
        .setOutputCol(c + "_idx")
    )
    val encoders = oneHotEncodeFeatures.map(c =>
      new OneHotEncoderEstimator()
        .setInputCols(Array(c + "_idx"))
        .setOutputCols(Array(c + "_vec"))
    )
    val assembler = new VectorAssembler()
      .setInputCols(featuresName)
      .setOutputCol("features")
      .setHandleInvalid("keep")

    val pipeline =
      new Pipeline().setStages(indexers ++ encoders ++ Array(assembler))
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

    transformedTrainData =
      transformedTrainData.select(col("Survived").as("label"), col("features"))
    transformedTestData = transformedTestData.select("features")

    // rename label
    // transformedTrainData.withColumn("Survived", col("label")).

    (transformedTrainData, transformedTestData)
  }

  def trainAndPredict(
      trainData: DataFrame,
      testData: DataFrame
  ): DataFrame = {
    val lr = new LogisticRegression()
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()
    // .addGrid(lr.elasticNetParam, Array(0.8, 0.7, 0.6, 0.5))

    val cv = new CrossValidator()
      .setEstimator(lr)
      .setEvaluator(
        // new BinaryClassificationEvaluator().setMetricName("areaUnderPR")
        new BinaryClassificationEvaluator()
      )
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvModel = cv.fit(trainData)
    cvModel.transform(testData)
  }

  def write2CSV(
      prediction: DataFrame,
      testData: DataFrame,
      outputDir: String,
      isWrite: Boolean
  ): DataFrame = {
    var df1 = testData
      .withColumn("id", monotonically_increasing_id())
      .select("PassengerId", "id")

    df1.printSchema()
    val df2 = prediction
      .withColumn("id", monotonically_increasing_id())
      .select("id", "prediction")
    val res = df1
      .join(df2, df1("id") === df2("id"), "outer")
      .drop("id")
      .orderBy(asc("PassengerId"))
    if (isWrite) {

      res.write
        .option("header", "True")
        .mode("overwrite")
        .save(outputDir)
    }
    res
  }
}
