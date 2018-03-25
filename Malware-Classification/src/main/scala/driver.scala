/**
  * Created by UNisar on 9/2/2016.
  */

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import scala.collection.mutable

object driver {
  var appName = "MalwareClassifier"
  var master = "local[8]"
  var executor = "10g"
  var accessKey = ""
  var secretKey = ""
  var conf: SparkConf = null
  var sc: SparkContext = null
  def initialize()
  {
    conf = new SparkConf()
      .setAppName(appName)
      .setMaster(master)
      .set("spark.executor.memory", executor)
      .set("spark.sql.warehouse.dir", "spark-warehouse")
      .set("spark.network.timeout", "9000")
      .set("spark.executor.heartbeatInterval", "800")

    sc = new SparkContext(conf)
    sc.hadoopConfiguration.set("fs.s3n.awsAccessKeyId", accessKey)
    sc.hadoopConfiguration.set("fs.s3n.awsSecretAccessKey", secretKey)
  }
}

class driver (xTrain: String, yTrain: String, xTest: String, metadataPath: String, resultsPath: String)
  extends Serializable {
  import driver._
  val opCodes = sc.textFile("Opcodes.txt").map(_.toLowerCase()).collect().toSet
  val opCodesMap = new mutable.HashMap[String, Int]()
  val segmentsMap = new mutable.HashMap[String, Int]()
  var index = 1
  for ( code <- opCodes)
  {
    opCodesMap.put(code, index )
    index = index + 1
  }
  index = 200
  sc.textFile("Segments.txt").collect().foreach ( line => {
    segmentsMap.put(line, index)
    index = index + 1
  })

  val corpusIterator = sc.textFile(xTrain).zipWithIndex().map(c => (c._2, c._1)).cogroup(sc.textFile(yTrain).zipWithIndex().map(c => (c._2, c._1))).map ( x => (x._2._2.head, x._2._1.head))

  /*
  This method returns all the opcodes in the given assembly file in the sequential order
   */
  def readASM(inputPath: String) = {
    val fullPath = metadataPath + inputPath + ".asm"
    val collection = sc.textFile(fullPath).flatMap ( line => {
      var index = line.indexOf(';')
      if (index == -1)
        index = line.length
      val tokens = line.dropRight ( line.length - index).split("[\\s]+").toSet
      val result = tokens.intersect(opCodes)
      if (result.isEmpty)
        None
      else
        result
    }).map(opCodesMap(_)).countByValue().toSeq.sortWith((a,b) => a._1 < b._1)
    var output = ""
    for ( col <- collection)
      output = output.concat(" " + col._1 + ":" + col._2)
    output
  }

  def readSegmentInformation(path: String) = {
    val fullPath = metadataPath + path + ".asm"
    val map = sc.textFile(fullPath).map (_.split(" ")(0).split(":")(0)).filter(x => segmentsMap.contains(x))
      .map(segmentsMap(_)).countByValue().toSeq.sortWith((a,b) => a._1 < b._1)
    var output = ""
    for (m <- map)
      output = output + m._1 + ":" + m._2 + " "
    output
  }

  /***
    * This method generates the feature vectors over all the training assembly files where the feature vectors consist of all opcodes
    */
  def runASM = corpusIterator.map ( item => item._1 + readASM(item._2) + " " + readSegmentInformation(item._2)).saveAsTextFile("trainingFeatures")


  /***
    * This method generates the feature vectors over all the testing assembly files
    */
  def runASMTest = sc.textFile(xTest).map ( item => 0 + " " + readASM(item) + " " + readSegmentInformation(item)).saveAsTextFile("testingFeatures")

  def classifyForest() {
    val data = MLUtils.loadLibSVMFile(sc, "trainingFeatures")
    val testingData = MLUtils.loadLibSVMFile(sc, "testingFeatures", 600)

    // Random Forest parameters
    val numClasses = 10
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 250   // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 24
    val maxBins = 32

    // Initiate the classifier
    val model = RandomForest.trainClassifier(data, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Dump the final results
    testingData.map ( x => model.predict(x.features).toInt).coalesce(1).saveAsTextFile(resultsPath)
  }
}
