# DocClassification

Document classification (Project 1) using Naive Bayes

Team # n00bs

Steps to get it up and running:

0. Assuming SBT is all setup
1. Check out the code. In case you want to access S3, add the access key/password in spark context in main file
2. Run it as "run {X} {Y} {Testing_X} {Output_Path} UseTFIDF(Boolean)" 
3. Fingers crossed!


# Highlights

1. Naive Bayes
2. TF/IDF
3. Stop word list 
4. Used stemmer but didn't improve the results considerably
5. Runs super-fast. If files are remote, download them and access locally. 



# How this thing works?

Following are the main steps in the run-up to classification: 

1. Build the two inputs (X & Y) based on ZipIndex  

2. Build the vocabulary based on X

3. Read the stop word list that is provided 

4. Since, Naive Bayes needs different statistics to compute the probability, we run a sequence of transformations/actions in the training step through the whole corpus that eventually returns the following structure:

   Map[String, (Long, Map[String, (Double, Double)])], where the constituents are:
   
   Map[TargetType (T), (# of docs with label T, Map[Word (W), (# of total occurrences against T in the whole corpus including duplicates against a single document, # of T docs with word W in it)])]

5. Once we have the structure above along with some other auxiliary structures that hold class probabilities, we can answer every stat that NB may need. 

6. In the classification, we log-sum all the probabilities after removing the words that are in StopList

7. Based on the highest value, emit the result. 
               

# Things we could have done different?

1. The main training method that calculates the convoluted structure above can be definitely refactored for better readability and easier maintainability. FWIW, it still is quite performant and successfully runs the larger testing dataset on the local machine in under two minutes! Maybe should have transformed the whole corpus to a matrix?

2. Data cleansing could have been improved with stemming, better stop list, surprisal? 

3. Other estimation methods like Bernoulli?

# Challenges

1. First time with Spark/Scala for the whole group

2. Setting up Spark on EC2 via Flintrock to make it read of S3





