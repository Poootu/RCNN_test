package org.example

import com.sksamuel.scrimage.ImmutableImage
import com.sksamuel.scrimage.filter.GaussianBlurFilter
import com.sksamuel.scrimage.filter.GrayscaleFilter
import com.sksamuel.scrimage.nio.PngWriter
import kotlinx.coroutines.runBlocking
import networkStructure.Layer
import networkStructure.Network
import networkStructure.Neuron
import org.example.ConvolutionalNetwork.Tensor
import org.example.ConvolutionalNetwork.createCnn
import org.example.ConvolutionalNetwork.softmax
import org.example.ConvolutionalNetwork.testNetwork
import org.jetbrains.kotlinx.dl.dataset.embedded.mnist
import org.tensorflow.op.core.HistogramFixedWidth
import java.io.File
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.text.compareTo
import kotlin.text.get

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
suspend fun main() {
//    test1()
    geminiTest()
//    println(net.toFileString())
//    tester()
//    importTest()
//    trainingTest()
//    trainingTest2();
//    trainingTest2Inverted()
//    manualTest()
//    pngTest()
}



fun geminiTest() {
    val imageFile = File("src/main/resources/mathTestSmall.png")
    var inputImage = ImmutableImage.loader().fromFile(imageFile)
    val listImage = mutableListOf<Int>()
    val filter = GaussianBlurFilter(1)
    inputImage = inputImage.filter(filter)

    //var greyedImage = inputImage.map { p -> java.awt.Color(inputImage.pixel(p.x,p.y).toAverageGrayscale().toInt()) }
    for (i in 0 until inputImage.height)
    {
        for (j in 0 until inputImage.width)
        {
            listImage.add((inputImage.pixel(j,i).toAverageGrayscale().blue()))

        }
    }
    var newImage=to2d(listImage,inputImage.width,inputImage.height)

    var greyedImage2 = inputImage.map { p -> java.awt.Color((newImage[p.y][p.x]),(newImage[p.y][p.x]),(newImage[p.y][p.x])) }
    File("src/main/resources/saved/s1.png.").createNewFile()
    greyedImage2.output(PngWriter.NoCompression,File("src/main/resources/saved/s1.png"))

    var segmentedGraph = (fasterImageSegmentation(listImage.toIntArray(),inputImage.width,inputImage.height,100))
    var segmentedList = vertsToList(segmentedGraph.values.toMutableList(),inputImage.height,inputImage.width)
    var segmentedImage = inputImage.map { p -> java.awt.Color((segmentedList[p.y][p.x].first),(segmentedList[p.y][p.x].second),(segmentedList[p.y][p.x].third)) }

    var merged = GEMINIbetterSelectiveSearch(segmentedGraph,listImage.toIntArray(),inputImage.width,inputImage.height)
    println(segmentedGraph.size)
//    segmentedImage.output(PngWriter.NoCompression,File("src/main/resources/saved/s1.png"))
//    segmentedList = vertsToList(merged,inputImage.height,inputImage.width)



    segmentedImage = inputImage.map { p ->
//        if(((b.x1 == p.x || b.x2 == p.x) && (b.y1 <= p.y && p.y <= b.y2)) || ((b.y1 == p.y || b.y2 == p.y) && (b.x1 <= p.x && p.x <= b.x2))) java.awt.Color(0,0,255)
        /*else*/ java.awt.Color((segmentedList[p.y][p.x].first),(segmentedList[p.y][p.x].second),(segmentedList[p.y][p.x].third))
    }
    var b = merged.map{getBoundingBox(it)}.filter { it.area() > 200 }
    var scaled = inputScaling(listImage.toIntArray(),b,inputImage.width, inputImage.height, 32,0)
    scaled.forEachIndexed {i, it -> saveTensorAsPng(it,"src/main/resources/saved/tests/$i.png")}
    val outlinedImage = inputImage.map { p ->
        val x = p.x
        val y = p.y
        var isBoundary = false

        for (box in b) {
//        if(box.area() < 200) continue
            // Check if the current pixel (x, y) is on any of the four edges of the bounding box.
            // This is the logic you requested, adapted to use minX/maxX/minY/maxY from BoundingBox.
            val onEdge = (
                    (box.x1 == x || box.x2 == x) && (box.y1 <= y && y <= box.y2)
                    ) || (
                    (box.y1 == y || box.y2 == y) && (box.x1 <= x && x <= box.x2)
                    )

            if (onEdge) {
                isBoundary = true
                break
            }
        }

        if (isBoundary) {
            // Color the boundary pixel blue (0, 0, 255)
            java.awt.Color(0, 0, 255)
        } else {
            // Fallback to the original pixel color
            java.awt.Color(p.red(), p.green(), p.blue())
        }
    }

    println(segmentedGraph.size)
    outlinedImage.output(PngWriter.NoCompression,File("src/main/resources/saved/s2.png"))
    segmentedImage.output(PngWriter.NoCompression,File("src/main/resources/saved/s3.png"))

    val inputSize = 32
    val channels = 1
    val classes = 10

    val cnn = createCnn(inputW = inputSize, inputH = inputSize, inputC = channels, numClasses = classes)

    var trainingSamples = loadImageDataset("src/main/resources/dataset/",32).shuffled()
    trainingSamples.forEach {
        println(it.label)
        cnn.train(it.tensor,intToResultTensor(it.label,10)) }
    trainingSamples.shuffled().forEach {
        println(it.label)
        cnn.train(it.tensor,intToResultTensor(it.label,10)) }
    trainingSamples.shuffled().forEach {
        println(it.label)
        cnn.train(it.tensor,intToResultTensor(it.label,10)) }


    // Convert to grayscale

    // Resize to network input size
//    img = img.scaleTo(32, 32)
    scaled.forEachIndexed {i, it ->
        (cnn.forward(scaled[0], applySoftmax = true))
    }

/*    println(cnn.forward(scaled[0], applySoftmax = true))
    println(cnn.forward(scaled[1143], applySoftmax = true))
    println(cnn.forward(trainingSamples[0].tensor, applySoftmax = true))*/


//    var regiontest = getArrayFromBox(b[0],newImage)
//    testNetwork(regiontest)

}

fun intToResultTensor(number : Int, maxNumber : Int) : Tensor   //maybe change numering system later
{
    return Tensor(DoubleArray(maxNumber) {i -> if(i!=number) 0.0 else 1.0},1,1,maxNumber)
}

fun inputScaling(
    image: IntArray,                 // grayscale image, 0..255 values
    boundingBoxes: List<BoundingBox>,
    width: Int,
    height: Int,
    intendedSize: Int,
    padding: Int                    // padding around resized patch
): List<Tensor> {

    val finalSize = intendedSize + 2 * padding
    val outputs = ArrayList<Tensor>()

    for (box in boundingBoxes) {

        // --- 1. Clamp bounding box to image bounds ---
        val x1 = box.x1.coerceIn(0, width - 1)
        val y1 = box.y1.coerceIn(0, height - 1)
        val x2 = box.x2.coerceIn(0, width - 1)
        val y2 = box.y2.coerceIn(0, height - 1)

        if (x2 < x1 || y2 < y1)
            continue

        val bw = x2 - x1 + 1
        val bh = y2 - y1 + 1

        // --- 2. Extract patch from image ---
        val patch = Array(bh) { IntArray(bw) }
        for (dy in 0 until bh) {
            val yy = y1 + dy
            for (dx in 0 until bw) {
                val xx = x1 + dx
                patch[dy][dx] = image[yy * width + xx]
            }
        }

        // --- 3. Resize patch to intendedSize×intendedSize (nearest-neighbor) ---
        val resized = Array(intendedSize) { IntArray(intendedSize) }
        for (y in 0 until intendedSize) {
            val srcY = (y * bh) / intendedSize
            for (x in 0 until intendedSize) {
                val srcX = (x * bw) / intendedSize
                resized[y][x] = patch[srcY][srcX]
            }
        }

        // --- 4. Create padded canvas ---
        val canvas = Array(finalSize) { IntArray(finalSize) { 0 } }

        // place resized patch in the center
        for (y in 0 until intendedSize) {
            for (x in 0 until intendedSize) {
                canvas[y + padding][x + padding] = resized[y][x]
            }
        }

        // --- 5. Convert padded canvas to Tensor (1 channel) ---
        val data = DoubleArray(finalSize * finalSize)
        var idx = 0
        for (y in 0 until finalSize) {
            for (x in 0 until finalSize) {
                data[idx++] = canvas[y][x] / 255.0
            }
        }

        outputs += Tensor(
            data = data,
            h = finalSize,
            w = finalSize,
            c = 1
        )
    }

    return outputs
}

//add padding
fun getArrayFromBox(b : BoundingBox, image : List<List<Int>>) : DoubleArray
{
    var width = abs(b.x2-b.x1)
    var proposedRegion = DoubleArray(b.area())
    for (i in b.x1..b.x2)
    {
        for (j in b.y1..b.y2)
        {
            proposedRegion[i * width+j] = image[i][j].toDouble() / 255
        }
    }
    return proposedRegion
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
    var net = Network(784, 10, 2, 10, true)
    var importedNet = networkFile.readLines().map { line -> line.split(" ")}.map { layer -> layer.map { it.substring(1,it.length-1).split(",") } }

    net = Network(importedNet.map {layer ->
        Layer(
            layer.map
        { neuron ->
            Neuron(
                neuron.subList(0, max(0, neuron.size - 2)).map
                { it.toDouble() }.toMutableList(), neuron.last().toDouble()
            )
        }.toMutableList()
        )
    }.toMutableList())

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
    { it.toDouble() }.toMutableList(),neuron.last().toDouble())
    }.toMutableList())
    }.toMutableList())

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


