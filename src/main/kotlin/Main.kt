package org.example

import com.sksamuel.scrimage.ImmutableImage
import kotlinx.coroutines.runBlocking
import org.example.networkStructure.Layer
import org.example.networkStructure.Network
import org.example.networkStructure.Neuron
import org.jetbrains.kotlinx.dl.dataset.embedded.mnist
import java.io.File
import kotlin.math.max
import kotlin.math.sqrt

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
suspend fun main() {
    test1()
//    println(net.toFileString())
//    tester()
//    importTest()
//    trainingTest()
//    trainingTest2();
//    trainingTest2Inverted()
//    manualTest()
//    pngTest()
}

fun pngTest() {

    val image = File("src/main/resources/gimp.png")
    val inputImage = ImmutableImage.loader().fromFile(image)
    val listImage = mutableListOf<Int>()

    inputImage.forEach {listImage.add(it.toColor().toGrayscale().gray) }
//    inputImage.forEach {print(it.toColor().toGrayscale().gray) }
//    var newImage=to2d(listImage)
//    println(newImage.size)



    var networkFile = File("src/main/resources/trained")
    var net = Network(784,10,2,10,true)
    var importedNet = networkFile.readLines().map { line -> line.split(" ")}.map { layer -> layer.map { it.substring(1,it.length-1).split(",") } }

    net = Network(importedNet.map {layer -> Layer(layer.map
    {neuron -> Neuron(neuron.subList(0,max(0,neuron.size-2)).map
    { it.toDouble() }.toMutableList(),neuron.last().toDouble())}.toMutableList())}.toMutableList())

    println(net.feedforward(listImage.map { it/255.0 }.toMutableList()))
    println(net.feedforward(listImage.map { it/255.0 }.toMutableList()).max())
    /*    for (i in 0 until newImage.size)
        {
            for (j in 0 until newImage.size)
            {
    //            print(newImage[i][j])
    //            print("${toGreyscale(newImage[i][j])}")
                print("\u001b[48;5;${toGreyscale(newImage[i][j])}m")
                print("${newImage[i][j]/255.0}")
            }
            println()

        }*/
}


fun manualTest() {

    val testFile = File("src/main/resources/mnist_test.csv")
    val testData =  testFile.readLines().map { line -> line.split(",").mapIndexed { l, it -> if(l!=0)it.toDouble()/255.0 else it.toDouble() }.toMutableList() }.toMutableList()
    val testingData: MutableList<Pair<MutableList<Double>, MutableList<Double>>> = mutableListOf()
    for(i in 0 until testData.size)
    {
        var expected = MutableList(10) { l -> if((testData[i][0]).toInt()==l) 1.0 else 0.0 }
        testingData.add(Pair((testData[i].subList(1,testData[i].size)).toMutableList(),expected))
    }

    var networkFile = File("src/main/resources/trained")
    var net = Network(784,10,2,10,true)
    var importedNet = networkFile.readLines().map { line -> line.split(" ")}.map { layer -> layer.map { it.substring(1,it.length-1).split(",") } }

    net = Network(importedNet.map {layer -> Layer(layer.map
    {neuron -> Neuron(neuron.subList(0,max(0,neuron.size-2)).map
    { it.toDouble() }.toMutableList(),neuron.last().toDouble())}.toMutableList())}.toMutableList())

    var stats = MutableList<Int>(10) {0}



    val manualFile = File("src/main/resources/testt2.csv")
//    val manualFile = File("src/main/resources/testt.csv")

    val manualData = manualFile.readText().split(",").mapIndexed { l, it -> it.toDouble()/255.0}.toMutableList()

    println(manualData)
    println(net.feedforward(manualData))
    println("start")
    net.evaluate(testingData)



//    for(i in 0 until testData.size) if(net.feedforward(testData[i])==testingData[i].second)stats[testData[i][0].toInt()]++

//    println(stats)

    /*println(testingData[0].second)
    println(net.feedforward(testData[0]))*/

}


suspend fun trainingTest()
{
    val trainFile = File("src/main/resources/mnist_train.csv")
    val testFile = File("src/main/resources/mnist_test.csv")
    val trainData = trainFile.readLines().map { line -> line.split(",").mapIndexed { l, it -> if(l!=0)it.toDouble()/255.0 else it.toDouble() } }.toMutableList()
    val testData =  testFile.readLines().map { line -> line.split(",").mapIndexed { l, it -> if(l!=0)it.toDouble()/255.0 else it.toDouble() }.toMutableList() }.toMutableList()
    var net = Network(784,10,1,30,false)


    val stats1 = mutableListOf<Int>(0,0,0,0,0,0,0,0,0,0)
    val stats2 = mutableListOf<Int>(0,0,0,0,0,0,0,0,0,0)
    trainData.forEach {data -> stats1[data.first().toInt()]++ }
    testData.forEach {data -> stats2[data.first().toInt()]++ }
    println("statsTrain $stats1")
    println("statsTest $stats2")
    val trainingData: MutableList<Pair<MutableList<Double>, MutableList<Double>>> = mutableListOf()

    for(i in 0 until trainData.size)
    {
//        println((trainData[i][0]).toInt())
        var expected = MutableList(10) { l -> if((trainData[i][0]).toInt()==l) 1.0 else 0.0 }
        trainingData.add(Pair((trainData[i].subList(1,trainData[i].size)).toMutableList(),expected))
    }

    val testingData: MutableList<Pair<MutableList<Double>, MutableList<Double>>> = mutableListOf()


    for(i in 0 until testData.size)
    {
        var expected = MutableList(10) { l -> if((testData[i][0]).toInt()==l) 1.0 else 0.0 }
        testingData.add(Pair((testData[i].subList(1,testData[i].size)).toMutableList(),expected))
    }
    val trained = File("src/main/resources/trained")
    for(i in 0 until 10)
    {
/*        println(testData[i][0])
        println(trainingData[0])
        println(testingData[0])*/

        println("epoch $i")
        net.evaluate(testingData)
        runBlocking{
            net.SGD(trainingData,10,3.0)/////////////
        }
        if(i%10==0)
        {
            manualTest()
            trained.writeText(net.toFileString())
        }
    }

}
suspend fun trainingTest2()
{
    val trainFile = File("src/main/resources/mnist_train.csv")
    val testFile = File("src/main/resources/mnist_test.csv")
//    val trainData = trainFile.readLines().map { line -> line.split(",").mapIndexed { l, it -> if(l!=0)it.toDouble()/255.0 else it.toDouble() } }.toMutableList()
//    val testData =  testFile.readLines().map { line -> line.split(",").mapIndexed { l, it -> if(l!=0)it.toDouble()/255.0 else it.toDouble() }.toMutableList() }.toMutableList()
    var net = Network(784,10,1,30,false)

    val (train, test) = mnist()
    val trainingData = train.x.mapIndexed {i,it -> Pair(it.map { it.toDouble() }.toMutableList(), MutableList(10) { l -> if(train.y[i].toInt()==l) 1.0 else 0.0 }) }.toMutableList()
    val testingData = test.x.mapIndexed {i,it -> Pair( it.map { it.toDouble() }.toMutableList(),MutableList(10) { l -> if(test.y[i].toInt()==l) 1.0 else 0.0 }) }.toMutableList()



    val stats1 = mutableListOf<Int>(0,0,0,0,0,0,0,0,0,0)
    val stats2 = mutableListOf<Int>(0,0,0,0,0,0,0,0,0,0)


//    trainingData.forEach {data -> satats1[data.first.toInt()]++ }
//    testingData.forEach {data -> stats2[data.first.toInt()]++ }
    println("statsTrain $stats1")
    println("statsTest $stats2")
/*    val trainingData: MutableList<Pair<MutableList<Double>, MutableList<Double>>> = mutableListOf()


    for(i in 0 until trainData.size)
    {
//        println((trainData[i][0]).toInt())
        var expected = MutableList(10) { l -> if((trainData[i][0]).toInt()==l) 1.0 else 0.0 }
        trainingData.add(Pair((trainData[i].subList(1,trainData[i].size)).toMutableList(),expected))
    }

    val testingData: MutableList<Pair<MutableList<Double>, MutableList<Double>>> = mutableListOf()


    for(i in 0 until testData.size)
    {
        var expected = MutableList(10) { l -> if((testData[i][0]).toInt()==l) 1.0 else 0.0 }
        testingData.add(Pair((testData[i].subList(1,testData[i].size)).toMutableList(),expected))
    }*/
    println(trainingData[0].second)
    val trained = File("src/main/resources/trained")
    for(i in 0 until 10)
    {
/*        println(testData[i][0])
        println(trainingData[0])
        println(testingData[0])*/

        println("epoch $i")
        net.evaluate(testingData)
        runBlocking{
            net.SGD(trainingData,10,3.0)/////////////
        }
        if(i%10==0)
        {
            manualTest()
            trained.writeText(net.toFileString())
        }
    }

}

suspend fun trainingTest2Inverted()
{
    val trainFile = File("src/main/resources/mnist_train.csv")
    val testFile = File("src/main/resources/mnist_test.csv")
//    val trainData = trainFile.readLines().map { line -> line.split(",").mapIndexed { l, it -> if(l!=0)it.toDouble()/255.0 else it.toDouble() } }.toMutableList()
//    val testData =  testFile.readLines().map { line -> line.split(",").mapIndexed { l, it -> if(l!=0)it.toDouble()/255.0 else it.toDouble() }.toMutableList() }.toMutableList()
    var net = Network(784,10,1,30,false)

    val (train, test) = mnist()
    val trainingData = train.x.mapIndexed {i,it -> Pair(it.map { 1.0 - it.toDouble() }.toMutableList(), MutableList(10) { l -> if(train.y[i].toInt()==l) 1.0 else 0.0 }) }.toMutableList()
    val testingData = test.x.mapIndexed {i,it -> Pair( it.map { 1.0 - it.toDouble() }.toMutableList(),MutableList(10) { l -> if(test.y[i].toInt()==l) 1.0 else 0.0 }) }.toMutableList()

//    println(trainingData[0].first)

    val stats1 = mutableListOf<Int>(0,0,0,0,0,0,0,0,0,0)
    val stats2 = mutableListOf<Int>(0,0,0,0,0,0,0,0,0,0)


//    trainingData.forEach {data -> stats1[data.first.toInt()]++ }
//    testingData.forEach {data -> stats2[data.first.toInt()]++ }
    println("statsTrain $stats1")
    println("statsTest $stats2")
    /*    val trainingData: MutableList<Pair<MutableList<Double>, MutableList<Double>>> = mutableListOf()


        for(i in 0 until trainData.size)
        {
    //        println((trainData[i][0]).toInt())
            var expected = MutableList(10) { l -> if((trainData[i][0]).toInt()==l) 1.0 else 0.0 }
            trainingData.add(Pair((trainData[i].subList(1,trainData[i].size)).toMutableList(),expected))
        }

        val testingData: MutableList<Pair<MutableList<Double>, MutableList<Double>>> = mutableListOf()


        for(i in 0 until testData.size)
        {
            var expected = MutableList(10) { l -> if((testData[i][0]).toInt()==l) 1.0 else 0.0 }
            testingData.add(Pair((testData[i].subList(1,testData[i].size)).toMutableList(),expected))
        }*/
    println(trainingData[0].second)
    val trained = File("src/main/resources/trained")
    for(i in 0 until 1)
    {
        /*        println(testData[i][0])
                println(trainingData[0])
                println(testingData[0])*/

        println("epoch $i")
        net.evaluate(testingData)
        runBlocking{
            net.SGD(trainingData,10,3.0)/////////////
        }
        if(i%10==0)
        {
            manualTest()
            trained.writeText(net.toFileString())
        }
    }

}


fun to2d(image: List<Int>,x : Int, y : Int) : List<List<Int>>{
    val returnImage = List<MutableList<Int>>(y) { mutableListOf<Int>() }
    for (i in 0 until y)
    {
        for (j in 0 until x)
        {
            returnImage[i].add(image[j + i*x])
//            println("y $i  x ${j+i*x}")
        }
    }
    return returnImage

}

fun layerTo2d(kernel: Layer) : List<List<Double>>{
    val side = sqrt(kernel.neurons.size.toDouble()).toInt()
    val returnImage = List<MutableList<Double>>(side) { mutableListOf<Double>() }
    for (i in 0 until side)
    {
        for (j in 0 until side)
        {
            returnImage[i].add(kernel.neurons[i+j].bias)
        }
    }
    return returnImage

}


fun importTest()
{
    val file = File("src/main/resources/mnist_test.csv")
    print("\u001b[48;5;249m")
    print(toGreyscale(198))
    print("\u001b[48;5;231m")
    print(toGreyscale(200))
    val data = file.readLines().map { line -> line.split(",").map { it.toInt() } }.toMutableList()
    println(data[0].sum())
    println(data[0].size)
//    for (j in 0..10) {
/*        for (i in 1 until data[35].size)
        {
            print("\u001b[48;5;${toGreyscale(data[35][i])}m")
            print("  ")
//            print("${ data[35][i] }   ")
            if(i%28==0) println()
        }
//    }*/

    //val kernel = (Layer(9,0,false))
/*    val image = data[35].toMutableList()
    image.removeFirst()
    var newImage = to2d(image)
    println("huh")
//    val kernel = Layer(9,0,false)
//    val newKernel = layerTo2d(kernel)
    val newKernel = mutableListOf<MutableList<Double>>(mutableListOf(-1.0,-1.0,-1.0),mutableListOf(-1.0,8.0,-1.0),mutableListOf(-1.0,-1.0,-1.0))
    newImage = convolve(newImage,newKernel)
    for (i in 0 until newImage.size)
    {
        for (j in 0 until newImage.size)
        {
//            print(newImage[i][j])
            print("\u001b[48;5;${toGreyscale(newImage[i][j])}m")
            print("  ")
        }
        println()

    }*/
    //grey 232-255
  //  println("\u001b[38;5;200m")

}



fun convolve(
    image: List<List<Int>>,
    kernel : List<List<Double>>
): List<List<Int>> {
    val kernelSide = (kernel.size).toInt()
    val imageSide = (image.size).toInt()
    val newArray = image.map { it.toMutableList() }.toMutableList()
    for(i in 0 until imageSide)
    {
        for (j in 0 until imageSide)
        {
            var sum = 0.0
            for (k in 0 until kernelSide)
            {
                for (l in 0 until kernelSide)
                {
                    if((i-kernelSide/2+k) < 0 || (i-kernelSide/2+k) > imageSide-1 || (j-kernelSide/2+l) < 0 || (j-kernelSide/2+l) > imageSide-1)
                    {

                    }else sum += (image[i-kernelSide/2+k][j-kernelSide/2+l] * kernel[l][k]) //todo handle edges
                }
            }

            newArray[i][j] = (sum/(kernelSide*kernelSide)).toInt()
        }
    }
    return newArray
}

fun toGreyscale(num : Int) : Int
{
    return ((num*24/255.0) + 231).toInt()
}

fun tester() {


/*    val testlist = mutableListOf<Double>(0.1,0.3,0.2)
    println(testlist.mapIndexed { index, value -> value * 2 })*/


    //testing Neuron
    val neuron1 = Neuron(2, false)
    val neuron2 = Neuron(2, true)
    val list1 = mutableListOf<Double>(0.6, 0.3)
    val neuron3 = Neuron(list1, 0.5)
    println("neuron1 - 2 weights random : $neuron1")
    println("neuron2 - 2 weights zeroes : $neuron2")
    println("neuron3 - 2 weights from list : $neuron3")


    //testing Layer
    val layer1 = Layer(3,2,false)
    val layer2 = Layer(3,2,true)

    val list2 = mutableListOf<Neuron>(neuron1, neuron2, neuron3)
    val layer3 = Layer(list2)

    println("layer1 - 3 neurons 2 inputs random : $layer1")
    println("layer2 - 3 neurons 2 inputs zeroes : $layer2")
    println("layer3 - 3 neurons from list : $layer3")


    //testing Network

    val net1 = Network(3,2,3,4,false)
    val net2 = Network(3,2,3,4,true)
    val net3 = net1 + net2

    println("network1 - 3 inputs 2 outputs 3 hidden 4 h_size random : $net1")
    println("network2 - 3 inputs 2 outputs 3 hidden 4 h_size zeroes : $net2")
    println("network3 - 3 inputs 2 outputs 3 hidden 4 h_size net1+net2 : $net3")

    //testing backpropagation
//    net3.backpropagation(Pair<MutableList<Double>,MutableList<Double>>(mutableListOf<Double>(1.0,0.6,0.7),mutableListOf(0.3,0.4)))

    //test feedForward
    var net4 = Network(3,2,1,2,false)
    println("network 4 ${net4}")
    println("feedforwarded ${net4.feedforward(mutableListOf(0.0,1.1,0.5))}")


}