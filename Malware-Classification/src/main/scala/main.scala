/**
  * Created by UNisar on 9/2/2016.
  */
object main {

  def main (args: Array[String]): Unit =
  {
    if (args.length < 9)
      {
        println ("Incorrect number of arguments. ")
        println ("Run as: run X_Train Y_Train X_Test ASM_Folder Output_Folder AppName Master Memory Access_KeyId Secret_Access_Key ")
        sys.exit(1)
      }

    driver.appName = args(5)
    driver.master = args(6)
    driver.executor = args(7)
    driver.accessKey = args(8)
    driver.secretKey = args(9)
    driver.initialize()

    val d  = new driver(args{0}, args{1}, args{2}, args{3}, args{4})
    d.runASM
    d.runASMTest
    d.classifyForest()
  }
}
