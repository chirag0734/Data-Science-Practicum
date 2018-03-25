import java.io.PrintWriter

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.{Map, mutable}

class naiveBayes (sc: SparkContext, x: String, y:String, testInput: String, useTFIDF: Boolean) extends java.io.Serializable {

  /**
    * This method is passed a document as an array of words that runs Naive Bayes algorithm and returns the single target class with the highest score
    * @param words Describes the document as an array of words
    * @return The target class
    */
  def getScoreForTargetType(words: Array[String]) = targetClasses.map(target => (target, (math.log10(classProbability.get(target)) + words.map(word => getProbabilityOfWordInTarget(word, target)).reduceLeft(_ + _)))).maxBy(_._2)._1

  /**
    * Method get the length of a given class
    * @param targetType
    * @return length
    */
  def getTotalWords(targetType: String): Double = {
    var total = 0.0
    for ((k, v) <- aggregated(targetType)._2)
      total += v._1
    total
  }

  /**
    * Primarily, this method fills up the following data structure: finalTrainingResult
    * This is a map(x1) of map(x2)[String, Int], where x1 is the class, x2 is the word and Int is the count
    */
  def train() {
    aggregated = corpusIterator.map(x => (x._1, x._2)).combineByKey(createCombiner, mergeValue, mergeCombiners).collectAsMap

    aggregated.foreach(numberOfDocuments += _._2._1)

    // normalizing all the values
    normalizationValues = targetClasses.map(x => (x, aggregated(x)._2.maxBy(b => b._2)._2._1)).toMap
    totalWordsPerClass = targetClasses.map(x => (x, getTotalWords(x))).toMap

    targetClasses.foreach(target => classProbability.put(target, (aggregated(target)._1.toDouble / numberOfDocuments)))
  }

  var numberOfDocuments = 0L
  var wordCounts: Map[String, Int] = null

  /**
    * This method performs the classification for all the documents and then dump the results in an output file
    */
  def classify(filePath:String): Unit = {
    val results = testData.flatMap(document => {
      val words = document.split(tokenizer).filter(_.length > 0).filter(c => !stopWordList.contains(c)).map(_.toLowerCase).toSet.intersect(vocabulary).toArray
      if (!words.isEmpty)
        Some(getScoreForTargetType(words))
      else
        None
    }).coalesce(1).saveAsTextFile(filePath)
  }

  def getCountInTarget(word: String, targetType: String): Double = {
    try {
      return aggregated(targetType)._2(word)._1
    }
    catch {
      case e: Exception => return 0
    }
  }

  /**
    * Returns the total occurrences of a given word in the whole corpus
    *
    * @param word
    * @return
    */
  def totalCounts(word: String): Double = {
    var total = 0.0
    for (t <- targetClasses) {
      if (aggregated(t)._2.contains(word))
        total += aggregated(t)._2(word)._2
    }
    total
  }

  /**
    * This method calculates the P (w|v) value where w is a word and v is a target value
    *
    * @param word       Word that we need the probability for
    * @param targetType Target class that we are interested in
    * @return Returns a double value
    */
  def getProbabilityOfWordInTarget(word: String, targetType: String): Double = {
    if (useTFIDF)
      {
        val tf = math.log10(getCountInTarget(word, targetType)/totalWordsPerClass(targetType))
        val idf = math.log10(math.log10(numberOfDocuments/totalCounts(word)))
        return tf + idf
      }
    else
      return math.log10(((getCountInTarget(word, targetType) + 1)/ (totalWordsPerClass(targetType) + vocabulary.size)))
  } // P(wk|vj)

  /**
    * Helper method to determine if the passed string is a number
    * @param text
    * @return
    */
  def isAllDigits(text: String) = text forall Character.isDigit

  //region DATA reading stuff
  val tokenizer = """[\s\W]"""  // The tokenizer to split the strings,
  val classProbability = new java.util.HashMap[String, Double] // This structure holds the mapping from CLASS -> P(V)
  var normalizationValues: Map[String, Double] = null
  var totalWordsPerClass: Map[String, Double] = null
  val stopWordList = sc.textFile("StopWordList.txt").collect.toSet

  var vocabulary = sc.textFile(x).flatMap(x => x.split(tokenizer)).filter(_.length > 0).map(_.toLowerCase()).distinct.collect
      .filter(!isAllDigits(_)).filter(c => !stopWordList.contains(c)).toSet
  sc.broadcast(vocabulary)

  var aggregated: Map[String, (Long, Map[String, (Double, Double)])] = null
  var testData =  sc.textFile(testInput)
  val documents = sc.textFile(x).zipWithIndex().map(x => (x._2, x._1))
  val labels = sc.textFile(y).zipWithIndex().map(x => (x._2, x._1))
  val merger = documents.cogroup(labels).map(x => (x._2._1, x._2._2))
  val mergeIterator = for (x <- merger) yield new Tuple2(x._2.head, x._1.head)
  val targetClasses = Set("CCAT", "ECAT", "GCAT", "MCAT")

  val corpusIterator: RDD[(String, Vector[String])] = mergeIterator.flatMap(x => {
    val targetTypes = x._1.split(",").toSet.intersect(targetClasses)
    if (targetTypes.isEmpty)
      None
    else
      targetTypes.map(b => (b, x._2.split(tokenizer).filter(_.length > 0).map(_.toLowerCase).filter ( f => vocabulary.contains(f)).toVector))
  })

  /**
    * Creates a map for the passed strings
    * @param vals
    * @return
    */
  def getMap(vals: Vector[String]): mutable.Map[String, (Double, Double)] = {
    var map = mutable.Map[String, (Double, Double)]()
    for (i <- vals) map.put(i, (map.getOrElse(i, (0.0, 0.0))._1 + 1, 1))
    map
  }

  def mergeMaps(output: mutable.Map[String, (Double, Double)], input: Vector[String]) = {
    input.foreach(x => {
      output.put(x, (output.getOrElse(x, (0.0, 0.0))._1 + 1, output.getOrElse(x, (0.0, 0.0))._2))
    })

    input.distinct.foreach(x => {
      output.put(x, (output.getOrElse(x, (0.0, 0.0))._1, output.getOrElse(x, (0.0, 0.0))._2 + 1))
    })
  }

  /**
    * Creates a combiner against a new key for the main combineByKey method
    * @param values
    * @return
    */
  def createCombiner(values: Vector[String]): (Long, mutable.Map[String, (Double, Double)]) = (1, getMap(values))

  /**
    * This method merges an entry against an existing Combiner
    * @param combiner
    * @param values
    * @return
    */
  def mergeValue(combiner: (Long, mutable.Map[String, (Double, Double)]), values: Vector[String]) = {
    mergeMaps(combiner._2, values)
    (combiner._1 + 1, combiner._2)
  }

  /**
    * This method merges two existing combiners
    * @param firstCombiner
    * @param secondCombiner
    * @return
    */
  def mergeCombiners(firstCombiner: (Long, mutable.Map[String, (Double, Double)]), secondCombiner: (Long, mutable.Map[String, (Double, Double)])) = {
    mergeMaps(firstCombiner._2, secondCombiner._2)
    (firstCombiner._1 + secondCombiner._1, firstCombiner._2)
  }

  def mergeMaps(output: mutable.Map[String, (Double, Double)], input: mutable.Map[String, (Double, Double)]) = {
    for ((k,v) <- input) output.put(k, (output.getOrElse(k, (0.0, 0.0))._1 + v._1, output.getOrElse(k, (0.0, 0.0))._2 + v._2))
  }

}