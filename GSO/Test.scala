import breeze.linalg._
import breeze.numerics._
import scala.collection.mutable.ArrayBuffer
import scala.math
import scala.util.control.Breaks._
class GSO {
  def Objvalue(x: DenseMatrix[Double], y: Int): DenseMatrix[Double] = {
    val objvalue = DenseMatrix.zeros[Double](y, 1)
    for (i <- 0 until y) {
      objvalue(i, 0) = 100 * pow(pow(x(i, 0), 2) - x(i, 1), 2) + pow(1 - x(i, 0), 2)
    }
    objvalue
  }

  def best(obj_Value: DenseMatrix[Double]) = {
    val len = obj_Value.rows
    var index_best = 0
    var best_Fit = obj_Value(0, 0)
    for (i <- 0 until len) {
      if (obj_Value(i, 0) > best_Fit) {
        best_Fit = obj_Value(i, 0)
        index_best = i
      }
    }
    (index_best, best_Fit)
  }
  def range1(X:DenseVector[Double],Y:DenseMatrix[Double]): Unit ={
//    print(X(0))

    for (i <-0 until  Y.cols){
      if (X(i)<Y(i,0)){
        X(i) = Y(i,0)
      }
      if(X(i)>Y(i,1)){
        X(i) = Y(i,1)
      }
    }
  }
}
object GSO {
  def apply() = new GSO()
}
object Test {
  def main(args: Array[String]): Unit = {
    var generation = 0
    val domx = DenseMatrix((-2.048,2.048),(-2.048,2.048))
    //    print(domx(0,::))
    val rho = 0.4
    val gamma = 0.6
    val beta = 0.08
    val nt = 5
    val s = 0.03
    val l0 = 5
    val rs = 2.048
    val r0 = 2.048
    val maxGeneration = 200
    val m = domx.cols
    //    print(m)
    val n =80
    val gAddress = DenseMatrix.zeros[Double](n,m)
    val gValue = DenseMatrix.zeros[Double](n,1)
    val li = DenseMatrix.zeros[Double](n,1)
    val rdi = DenseMatrix.zeros[Double](n,1)
    for (i <- 0 until n) {
      for (j <- 0 until m) {
        gAddress(i, j) = domx(j, 0) + (domx(j, 1) - domx(j, 0)) * math.random()
      }
    }
    for(i <- 0 until n) {
      li(i, 0) = l0
    }
    for(i <- 0 until n){
      rdi(i,0) = r0
    }
    val obj = GSO()
    val objvalue = obj.Objvalue(gAddress,n)
    //    print(objvalue)
    val best = obj.best(objvalue)
    var currentbestfit:Double = best _2
    var currentbestindex = best._1
    while (generation<maxGeneration){
      generation+=1
      val objvalue1 = obj.Objvalue(gAddress,n)
      for (i <- 0 until n){
        li(i,0) = (1-rho)*li(i,0)+gamma*objvalue1(i,0)
      }
      for (i <- 0 until n){
        val Nit = ArrayBuffer[Int]()
        for (j <- 0 until n){
             val a = norm(gAddress(j,::).t-gAddress(i,::).t)
             if ((a<rdi(i,0))&&(li(i,0)<li(j,0))){
               Nit.append(j)
             }
        }
        val num = Nit.length
        if(num != 0){
          val Nitioto = ArrayBuffer[Double]()
          val numerator = ArrayBuffer[Double]()
          val Pij = ArrayBuffer[Double]()
          for (j <- 0 until num){
            Nitioto.append(li(Nit(j),0))
          }
          //求荧光素和
          val sumNitioto = sum(Nitioto)
        //求分子和分母
          for (j <- 0 until num){
            numerator.append(Nitioto(j)-li(i,0))
          }
//          print(numerator)
          val denominator = sumNitioto - li(i,0)
          for (j <-0 until num){
            Pij.append(numerator(j)/denominator)
          }
//          print(Pij)
          //概率归一化
//          accumulate(Pij)
          for (j <- 1 until num){
            Pij(j) = Pij(j-1)+Pij(j)
//            print(j)
          }
          for (j<-0 until num){
            Pij(j) = Pij(j)/Pij(num-1)
          }
//          print(Pij)
          var Pos = 0

          for (j <- 0 until num) {
            if(math.random()<Pij(j)){
              Pos = j

//              j = num+1
//              break
            }
          }
//          print(Pos+"s")
         val J = Nit(Pos)
//          print(J+"s")
//         gAddress(i,::) = gAddress(i,::) + s*((gAddress(J,::)-gAddress(i,::))/norm(gAddress(J,::).t-gAddress(i,::).t))
          gAddress(i,0) = gAddress(i,0)+s*((gAddress(J,0)-gAddress(i,0))/norm(gAddress(J,::).t-gAddress(i,::).t))
          gAddress(i,1) = gAddress(i,1)+s*((gAddress(J,1)-gAddress(i,1))/norm(gAddress(J,::).t-gAddress(i,::).t))
//          print(gAddress)
          obj.range1(gAddress(i,::).t,domx)

          // 更新决策半径
            rdi(i,0) = rdi(i,0) + beta * (nt - num)
            rdi(i,0) = min(rs,max(0.0,rdi(i,0)))
          }

        }

      //           获取最优个体
      val objvalue2 = obj.Objvalue(gAddress,n)
      //            objectvalue2 = ObjValue(gAddress)
      val best1 = obj.best(objvalue2)
      if (best1._2 > currentbestfit) {
        currentbestfit = best1._2
      }
      if (best1._2 > currentbestfit){
        currentbestfit = objvalue2(best1._1,0)
      }

      print(generation + "+"+ currentbestfit)
      print("\n")
      }
//      val re = obj.Objvalue(gAddress,n)
//      print(re)
    }
  }

