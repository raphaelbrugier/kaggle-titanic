import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{SparkSession, _}
import org.apache.spark.sql.functions._

object TitanicMain {

  val conf = new SparkConf()
    .setAppName("mlpoc")
    .setMaster("local[*]")
    .set("spark.sql.shuffle.partitions", "2")

  val sc = new SparkContext(conf)
  val sqlContext = SparkSession.builder.config(conf).getOrCreate()

  def computeAverageAge(training: DataFrame, test: DataFrame) = {
    training.select("Age")
      .union(test.select("Age"))
      .agg(avg("Age"))
      .head().getDouble(0)
  }

  def main(args: Array[String]) {

    var training = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .load("src/main/resources/data/titanic/train.csv")

    var test = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .load("src/main/resources/data/titanic/test.csv")

    training.printSchema()
    training.show()

    // clean age
    val averageAge = computeAverageAge(training, test)
    training = training.na.fill(averageAge, Array("Age"))
    test = test.na.fill(averageAge, Array("Age"))

    // clean Embarked
    training = training.na.fill("S", Array("Embarked"))
    test = test.na.fill("S", Array("Embarked"))

    // clean null Fare
    training = training.na.drop("any", Array("Fare"))
    test = test.na.fill(8.05, Array("Fare"))

    // add title
    val nameToTitle: String => String = _.replaceAll(""".*, |\..*""", "") match {
      case "Miss" => "Miss"
      case "Mr" => "Mr"
      case "Mrs" => "Mrs"
      case _ => "RareTitle"
    }
    training = training.withColumn("title", udf(nameToTitle).apply(training("Name")))
    test = test.withColumn("title", udf(nameToTitle).apply(test("Name")))

    // Index Sex
    val sexIndexer = new StringIndexer()
      .setInputCol("Sex")
      .setOutputCol("SexIndex")
      .fit(training)

    // Index Embarked
    val embarkedIndexer = new StringIndexer()
      .setInputCol("Embarked")
      .setOutputCol("EmbarkedIndex")
      .fit(training)

    // Title Embarked
    val titleIndexer = new StringIndexer()
      .setInputCol("title")
      .setOutputCol("titleIndex")
      .fit(training)

    // Index Survived as the Label
    val labelIndexer = new StringIndexer()
      .setInputCol("Survived")
      .setOutputCol("Label")
      .fit(training)

    val assembler = new VectorAssembler()
      .setInputCols(Array("Pclass", "Age", "Fare", "SexIndex", "EmbarkedIndex", "titleIndex"))
      .setOutputCol("Features")

    val randomForest = new RandomForestClassifier()
      .setLabelCol("Label")
      .setFeaturesCol("Features")

    val pipeline = new Pipeline().setStages(
      Array(sexIndexer,
        embarkedIndexer,
        titleIndexer,
        labelIndexer,
        assembler,
        randomForest
      )
    )

    val model = pipeline.fit(training)

    val predictions = model.transform(test)
//    predictions.selectExpr("PassengerId", "cast(prediction as int) Survived")
//      .repartition(1)
//      .write
//      .format("com.databricks.spark.csv")
//      .option("header", "true")
//      .save("target/predictions.csv")

    predictions.show()

    // With cross validation

//    val params = new ParamGridBuilder()
//      .addGrid(randomForest.maxDepth, Array(4, 8, 12))
//      .addGrid(randomForest.numTrees, Array(15, 30, 50))
//      .addGrid(randomForest.maxBins, Array(16, 32, 64))
//      .addGrid(randomForest.impurity, Array("entropy", "gini"))
//      .build()
//
//    val evaluator = new BinaryClassificationEvaluator()
//      .setLabelCol("Label")
//      .setRawPredictionCol("prediction")
//      .setMetricName("areaUnderPR")
//
//    val cv = new CrossValidator()
//      .setEstimator(pipeline)
//      .setEvaluator(evaluator)
//      .setEstimatorParamMaps(params)
//      .setNumFolds(10)
//
//    val crossValidatorModel = cv.fit(training)
//    val predictionsCV = crossValidatorModel.transform(test)
//
//    predictionsCV.selectExpr("PassengerId", "cast(prediction as int) Survived")
//      .repartition(1)
//      .write
//      .format("com.databricks.spark.csv")
//      .option("header", "true")
//      .save("target/predictionsCV.csv")
//
//    predictionsCV.show()
  }
}