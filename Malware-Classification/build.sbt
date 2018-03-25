//name := "Classification"
//
//version := "1.0"
//
//scalaVersion := "2.10.5"

lazy val root = (project in file(".")).
  settings(
    name := "Classification",
    version := "1.0",
    scalaVersion := "2.10.5",
    mainClass in Compile := Some("main")
  )

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.6.2" % "provided"
libraryDependencies += "org.scala-lang" % "scala-library" % scalaVersion.value
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.6.2" % "provided"

// META-INF discarding
mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
{
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}
}