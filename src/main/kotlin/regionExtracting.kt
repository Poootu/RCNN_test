package org.example

import com.sksamuel.scrimage.ImmutableImage
import com.sksamuel.scrimage.filter.GrayscaleFilter
import com.sksamuel.scrimage.nio.PngWriter
import com.sksamuel.scrimage.pixels.Pixel
import org.example.ConvolutionalNetwork.Tensor
import java.io.File
import java.util.PriorityQueue
import kotlin.collections.forEach
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.random.Random
var Edges = mutableListOf<Edge>()

fun calculateGaussianKernel(radius: Int, sigma: Double): List<Double> {
    val size = 2 * radius + 1
    val kernel = MutableList(size) { 0.0 }
    var sum = 0.0

    // Calculate and sum un-normalized weights based on the Gaussian equation
    for (i in 0 until size) {
        val x = i - radius
        val weight = exp(-(x * x) / (2 * sigma.pow(2)))
        kernel[i] = weight
        sum += weight
    }

    // Normalize weights
    return kernel.map { it / sum }
}

/**
 * Applies a Gaussian blur filter to a grayscale image (IntArray).
 * Uses a separable kernel (two 1D passes) for efficiency.
 */
fun applyGaussianBlur(
    image: IntArray,
    width: Int,
    height: Int,
    radius: Int = 1,
    sigma: Double = 1.0
): IntArray {

    val kernel = calculateGaussianKernel(radius, sigma)
    val size = kernel.size

    val tempImage = IntArray(width * height)
    val outputImage = IntArray(width * height)

    // PASS 1: Horizontal Blur (Input -> TempImage)
    for (y in 0 until height) {
        for (x in 0 until width) {
            var sum = 0.0

            for (k in 0 until size) {
                val offset = k - radius
                val sampleX = (x + offset).coerceIn(0, width - 1) // Boundary clamping
                sum += image[y * width + sampleX] * kernel[k]
            }
            tempImage[y * width + x] = sum.roundToInt().coerceIn(0, 255)
        }
    }

    // PASS 2: Vertical Blur (TempImage -> OutputImage)
    for (y in 0 until height) {
        for (x in 0 until width) {
            var sum = 0.0

            for (k in 0 until size) {
                val offset = k - radius
                val sampleY = (y + offset).coerceIn(0, height - 1) // Boundary clamping
                sum += tempImage[sampleY * width + x] * kernel[k]
            }
            outputImage[y * width + x] = sum.roundToInt().coerceIn(0, 255)
        }
    }

    return outputImage
}


fun androidSegmentation(listTriple : IntArray, height: Int, width: Int): IntArray
{
        val size = height*width

        var coloredImage = IntArray(size) //because we save it as ARGB_8888 so 32 bits like an Int

        //gemini
        coloredImage = applyGaussianBlur(coloredImage,width,height,1,0.9)

        for (i in 0 until size)
        {
            val greyedColor = (((listTriple[i] shr 16) and 0xFF) * 0.21 + ((listTriple[i] shr 8) and 0xFF) * 0.72  + ((listTriple[i]) and 0xFF) * 0.07).toInt()
            coloredImage[i]= (0xFF shl 24) or (greyedColor shl 16) or (greyedColor shl 8) or (greyedColor shl 0)
        }


        val segmented = fasterImageSegmentation(coloredImage.copyOf(),width,height,700)

//        val merged = betterSelectiveSearch(segmented,coloredImage,width,height)
        val merged = GEMINIbetterSelectiveSearch(segmented,coloredImage,width,height)

        /*segmented.values*/merged.forEachIndexed{i, segment ->
            val randomColor = Random.nextBits(24)
            segment.forEachIndexed{ j,pixel ->
                coloredImage[pixel.first * width + pixel.second] = (0xFF shl 24) or randomColor //getting back to 32 bits for one color
            }

        }

        return coloredImage
}

fun test1()
{

    //maybe colored segmentation??

    //temp
    //temp

    val imageFile = File("src/main/resources/hophoto.png")
    var inputImage = ImmutableImage.loader().fromFile(imageFile)
    val listImage = mutableListOf<Int>()
//    val filter = GaussianBlurFilter(1)
//    inputImage = inputImage.filter(filter)

    //var greyedImage = inputImage.map { p -> java.awt.Color(inputImage.pixel(p.x,p.y).toAverageGrayscale().toInt()) }
        for (i in 0 until inputImage.height)
    {
            for (j in 0 until inputImage.width)
        {
            listImage.add((inputImage.pixel(j,i).toAverageGrayscale().blue()))

        }
    }
    var newImage=to2d(listImage,inputImage.width,inputImage.height)
//    println(newImage.forEachIndexed { index, pixel -> pixel.forEachIndexed { j, pixel -> println("$index  $j") } })

/*
    val region1 = mutableListOf<Edge>(
        Edge(v1 = Pair(0,0), v2 = Pair(2,2), 75),
        Edge(v1 = Pair(0,0), v2 = Pair(1,1), 9),
        Edge(v1 = Pair(1,1), v2 = Pair(2,2), 95),
        Edge(v1 = Pair(2,2), v2 = Pair(3,3), 51),
        Edge(v1 = Pair(1,1), v2 = Pair(3,3), 19),
        Edge(v1 = Pair(1,1), v2 = Pair(4,4), 42),
        Edge(v1 = Pair(4,4), v2 = Pair(3,3), 31)
    )
    println("starting msp")
    println(minimumSpanningTree(region1).size)
    println(minimumSpanningTree(region1))
*/

    var greyedImage2 = inputImage.map { p -> java.awt.Color((newImage[p.y][p.x]),(newImage[p.y][p.x]),(newImage[p.y][p.x])) }
    File("src/main/resources/saved/s1.png.").createNewFile()
    greyedImage2.output(PngWriter.NoCompression,File("src/main/resources/saved/s1.png"))

    var segmentedGraph = (fasterImageSegmentation(listImage.toIntArray(),inputImage.width,inputImage.height,100))
    var segmentedList = vertsToList(segmentedGraph.values.toMutableList(),inputImage.height,inputImage.width)
    var segmentedImage = inputImage.map { p -> java.awt.Color((segmentedList[p.y][p.x].first),(segmentedList[p.y][p.x].second),(segmentedList[p.y][p.x].third)) }

    println(segmentedGraph.size)
    segmentedImage.output(PngWriter.NoCompression,File("src/main/resources/saved/s1.png"))
/*
    segmentedGraph = similarityMerging(segmentedGraph,newImage,Edges)
    segmentedList = vertsToList(segmentedGraph,inputImage.height,inputImage.width)
    segmentedImage = inputImage.map { p -> java.awt.Color((segmentedList[p.y][p.x].first),(segmentedList[p.y][p.x].second),(segmentedList[p.y][p.x].third)) }

    println(segmentedGraph.size)
    segmentedImage.output(PngWriter.NoCompression,File("src/main/resources/saved/s2.png"))

    segmentedGraph = similarityMerging(segmentedGraph,newImage,Edges)
    segmentedList = vertsToList(segmentedGraph,inputImage.height,inputImage.width)
    segmentedImage = inputImage.map { p -> java.awt.Color((segmentedList[p.y][p.x].first),(segmentedList[p.y][p.x].second),(segmentedList[p.y][p.x].third)) }
    println(segmentedGraph.size)
    segmentedImage.output(PngWriter.NoCompression,File("src/main/resources/saved/s3.png"))

    segmentedGraph = similarityMerging(segmentedGraph,newImage,Edges)
    segmentedList = vertsToList(segmentedGraph,inputImage.height,inputImage.width)
    segmentedImage = inputImage.map { p -> java.awt.Color((segmentedList[p.y][p.x].first),(segmentedList[p.y][p.x].second),(segmentedList[p.y][p.x].third)) }
    println(segmentedGraph.size)
    segmentedImage.output(PngWriter.NoCompression,File("src/main/resources/saved/s4.png"))*/
}

fun vertsToList(
    segmentedGraph: MutableList<MutableList<Pair<Int, Int>>>,
    height: Int,
    width: Int
) : MutableList<MutableList<Triple<Int,Int,Int>>>
        {
        val list = MutableList(height) { MutableList(width){ Triple(0,0,0) }}

        segmentedGraph.forEach {
            val color = Triple(Random.nextInt(10,250),Random.nextInt(10,250),Random.nextInt(10,250))
            it.forEach {
                list[it.first][it.second] = color
            } }

        return list
}

fun graphToList(graph : MutableList<MutableList<Edge>>,height : Int, width: Int) : MutableList<MutableList<Triple<Int,Int,Int>>>
{
    val list = MutableList(height) { MutableList(width){ Triple(0,0,0) }}

    graph.forEach {
        val color = Triple(Random.nextInt(10,250),Random.nextInt(10,250),Random.nextInt(10,250))
        it.forEach {
        list[it.v1.first][it.v1.second] = color
        list[it.v2.first][it.v2.second] = color
    } }

    return list
}


fun test2()
{
    val box1 = BoundingBox(20,10,40,40)
    val box2 = BoundingBox(30,20,80,60)
    println(intersectionOverUnion(box1, box2))
}



fun intersectionOverUnion(box1 : BoundingBox, box2 : BoundingBox) : Double
{
    val box1Area = box1.area()
    val box2Area = box2.area()
    val intersectionArea = box1.intersection(box2).area()
    val IOU : Double = (intersectionArea) / (box1Area + box2Area - intersectionArea).toDouble()
    return IOU
}







class BoundingBox(var x1 :Int, var y1 :Int, var x2 :Int, var y2 :Int) {
    fun intersection(other: BoundingBox): BoundingBox {
        return BoundingBox(
            x1 = max(x1,other.x1), //left top ig
            y1 = max(y1,other.y1),
            x2 = min(x2,other.x2), //right bottom ig
            y2 = min(y2,other.y2)
            )
    }
    fun area() : Int
    {
        return if(x2 < x1 || y2 < y1) 0
        else (x2-x1 + 1) * (y2-y1 + 1)
    }
}

data class Edge(
    val v1 : Pair<Int,Int>,
    val v2 : Pair<Int,Int>,
    val weight : Int

) {
    override fun toString(): String {
        return "v1=$v1, v2=$v2, weight=$weight"
    }
}

fun fastImageSegmentation(image : MutableList<MutableList<Int>>,k : Int): MutableList<MutableList<Pair<Int, Int>>> { //implemented with union find
    val height = image.size  //maybe the other way
    val width = image[0].size
//later might add ranking to make tree shallow and fasten thing even more
//    var components: List<Pair<MutableList<Pair<Int, Int>>,Int>> // it is a list of pairs containing list of vertices and a threshold number
    val edges = mutableListOf<Edge>()   //list of edges
    var parenthood = mutableListOf<Int>()  //list of parent indexes
    var sizes = mutableListOf<Int>()  //list 'families' sizes
    var thresholds = mutableListOf<Int>()  //list of thresholds
    println("starting segmentation")
    for (i in 0 until height)
    {
        for (j in 0 until width)
        {
            if(i<height - 1) edges.add(Edge(Pair(i,j),Pair(i+1,j), edgeWeight(image[i][j],image[i+1][j])))
            if(j<width - 1) edges.add(Edge(Pair(i,j),Pair(i,j+1), edgeWeight(image[i][j],image[i][j+1])))
        }
    }
    edges.sortBy { it.weight }
    for (i in 0 until height*width) {
        parenthood.add(i)
        thresholds.add(k)
        sizes.add(1)
    }
    for (i in 0 until edges.size) {
        var currentParent = edges[i].v1.first * width + edges[i].v1.second
        while (parenthood[currentParent] != currentParent) //finding function
        {
            currentParent = parenthood[currentParent]
        }
        val parent1 = currentParent

        currentParent = edges[i].v2.first * width + edges[i].v2.second
        while (parenthood[currentParent] != currentParent) //finding function
        {
            currentParent = parenthood[currentParent]
        }
        val parent2 = currentParent

        if (parent1==parent2) continue
        if(edges[i].weight > thresholds[parent1] || edges[i].weight > thresholds[parent2]) continue

        parenthood[parent1] = parent2
        sizes[parent2] += sizes[parent1]
        thresholds[parent2] = edges[i].weight + k/sizes[parent2]

    }

    val tempMap = mutableMapOf<Int,MutableList<Pair<Int,Int>>>()
    for (i in 0 until height)
    {
        for (j in 0 until width)
        {
            var currentParent = i * width + j
            while (parenthood[currentParent] != currentParent) //finding function
            {
                currentParent = parenthood[currentParent]
            }
            tempMap.getOrPut(currentParent) { mutableListOf() }.add(Pair(i,j))
        }
    }
    println("segmentation finished")
    Edges = edges
    return tempMap.values.toMutableList()
}


fun imageSegmentation(image : MutableList<MutableList<Int>>,k : Int): MutableList<MutableList<Pair<Int, Int>>> {
    val height = image.size  //maybe the other way
    val width = image[0].size
    val edges = mutableListOf<Edge>()   //list of edges
    var components: List<Pair<MutableList<Pair<Int, Int>>,Int>> // it is a list of pairs containing list of vertices and a threshold number
    println("starting segmentation")
    for (i in 0 until height)
    {
        for (j in 0 until width)
        {
            if(i<height - 1) edges.add(Edge(Pair(i,j),Pair(i+1,j), edgeWeight(image[i][j],image[i+1][j])))
            if(j<width - 1) edges.add(Edge(Pair(i,j),Pair(i,j+1), edgeWeight(image[i][j],image[i][j+1])))
        }
    }
    edges.sortBy { it.weight }
//    components = edges.map { mutableListOf(it) } as MutableList<MutableList<Edge>>
    components =( edges.map {Pair(mutableListOf(it.v1),k )} + edges.map { Pair(mutableListOf(it.v2),k) }).distinctBy{it.first}.toMutableList()
        for (i in 0 until edges.size) {
            try{
                val C1 =
                    components.first() { if(it.first.find {it == edges[i].v1 } != null) true else false }
                val C2 =
                    components.first() { if(it.first.find {it == edges[i].v2 } != null) true else false }
                if (C1 != C2 && edges[i].weight <= C1.second && edges[i].weight <= C2.second) {
                    components.remove(C2)
                    components.remove(C1)
                    components.add(Pair(C1.first + C2.first,edges[i].weight + (k/(C1.first.size+C2.first.size))) as Pair<MutableList<Pair<Int, Int>>, Int>)
                }///////merging edge is maximum in the component cause they are ordered

            }
            catch (e :Exception){}
    }
    println("segmentation finished")
    Edges = edges
    return components.map { it.first }.toMutableList()
}

fun findEdges(
    vertices: MutableList<Pair<Int, Int>>,
    edges: MutableList<Edge>,

): MutableList<Edge> {
    return edges.filter {edge -> vertices.find { it == edge.v1 } != null  && vertices.find { it == edge.v2 }!= null  }.toMutableList()
}


fun MIntDiff(C1: MutableList<Edge>, C2: MutableList<Edge>, k: Int) : Int
{
//    println(tau(C1,k))
    return min(internalDifference(C1) + tau(C1,k),internalDifference(C2) + tau(C2,k))
}
fun internalDifference(C : MutableList<Edge>) : Int
{
    if(C.size == 1) return 0
    return minimumSpanningTree(C).maxBy { it.weight }.weight
}
fun tau(C : MutableList<Edge>, k: Int) : Int {
    return k / C.size
} //return should be Int?

fun edgeWeight(i1: Int, i2: Int): Int {
    return abs((i1 and 0xFF)-(i2 and 0xFF)) //change that later ig
}
fun minimumSpanningTree(C: MutableList<Edge>) : MutableList<Edge>
{
    val visitedEdges = mutableListOf<Edge>()
    val D = C/*.map { mutableListOf(it) }*/.sortedBy { it/*[0]*/.weight }.toMutableList()
    val unvisitedVertices = (D.map { it.v1 } + D.map { it.v2 }).distinct().toMutableList()
    visitedEdges.add(D.first())
    unvisitedVertices.remove(D.first().v1)
    unvisitedVertices.remove(D.first().v2)
    D.removeFirst()
    while(unvisitedVertices.isNotEmpty()) {
        val next1 = D.first { first -> visitedEdges.find { it.v1 == first.v1 || it.v1 == first.v2 || it.v2 == first.v1 || it.v2 == first.v2} != null && (unvisitedVertices.find { first.v1 == it } != null || unvisitedVertices.find { first.v2 == it } != null)}
    //println("next $next1     $unvisitedVertices")
            visitedEdges.add(next1)
            unvisitedVertices.remove(next1.v1)
            unvisitedVertices.remove(next1.v2)
            D.remove(next1)



    }
    return visitedEdges
}



/*fun similarityMerging(components : MutableList<MutableList<Pair<Int, Int>>>,image : List<List<Int>>,edges: MutableList<Edge>) : MutableList<MutableList<Pair<Int, Int>>>
{


    var neighbourList = mutableListOf<Pair<MutableList<Pair<Int, Int>>, MutableList<Pair<Int,Int>>>>()   // list containing all components and their neighbours(by first elements reference)
    var similarities = mutableListOf<Triple<Pair<Int, Int>,Pair<Int, Int>,Int>>()  //reference 1, reference 2, similarity

    for(j in 0 until components.size) {
        neighbourList.add(Pair(components[j], mutableListOf()))
    }
    edges.forEach { (v1, v2, weight) -> if(components.find {it.find { it == v1 } != null} != components.find {it.find { it == v2 } != null})
        neighbourList.find { t -> t.first[0] == components.find{it[0]==t.first[0]} }
        neighbourList[j].second.add(component[0])
        neighbourList[j].second.add(component[0])
    }

    for(j in 0 until components.size)
    {
        neighbourList.add(Pair(components[j],mutableListOf()))
        components.forEachIndexed {index, component -> if(index!=j) {
            var bb = getBoundingBox((component + components[j]) as MutableList<Pair<Int, Int>>)
            var b1 = getBoundingBox(components[j])
            var b2 = getBoundingBox(component)
            if(bb.area() != 0 && bb.area() < b1.area() + b2.area())  //change neighbor condition
            {
                neighbourList[j].second.add(component[0])
            }
            }
        }
    }
    var visitedSimilarities = mutableListOf<Pair<Pair<Int, Int>, Pair<Int, Int>>>()
    neighbourList.forEach {nL ->
        nL.second.forEach {neighbour -> if(visitedSimilarities.find {sim -> nL.first[0]==sim.first && neighbour==sim.second || nL.first[0]==sim.second && neighbour==sim.first} == null)
        {
            similarities.add(Triple(nL.first[0],neighbour,calculateSimilarity(nL.first,neighbourList.first{it.first[0]==neighbour}.first    ,image)))
        } }
    }

    while(similarities.isNotEmpty())
    {
        var highestSimilarity = similarities.maxBy { it.third }
        var forMerge = neighbourList.find { it.first[0] == highestSimilarity.second }
        neighbourList.remove(forMerge)
        neighbourList.find { it.first[0] == highestSimilarity.first }?.first?.addAll(forMerge!!.first)
//        neighbourList.find { it.first[0] == highestSimilarity.first }?.second?.addAll(forMerge!!.second)
        similarities.removeIf { it.first == highestSimilarity.first || it.second == highestSimilarity.first || it.first == highestSimilarity.second || it.second == highestSimilarity.second }

        visitedSimilarities.removeAll(visitedSimilarities)
        neighbourList.forEach {nL ->
            if(nL.first[0]==highestSimilarity.first || nL.first[0]==highestSimilarity.second)
            nL.second.forEach {neighbour -> if(visitedSimilarities.find {sim -> nL.first[0]==sim.first && neighbour==sim.second || nL.first[0]==sim.second && neighbour==sim.first} == null)
            {
                try {
                similarities.add(Triple(nL.first[0],neighbour,calculateSimilarity(nL.first,neighbourList.first{it.first[0]==neighbour}.first    ,image)))
                }
                catch (e : Exception){}
            } }
        }


    }
    return neighbourList.map { it.first }.toMutableList() //temp ig
}*/

/*fun similarityMerging(components : MutableList<MutableList<Pair<Int, Int>>>,image : MutableList<MutableList<Int>>,iterations : Int) : MutableList<MutableList<Pair<Int, Int>>>
{
    var neighbourList = mutableListOf<Pair<Pair<Int,Int>, MutableList<Pair<Int,Int>>>>()   // list containing all components(by first element) and their neighbours(by first elements)
    var similarities = mutableListOf<Triple<MutableList<Pair<Int, Int>>,MutableList<Pair<Int, Int>>,Int>>()  //c1, c2, similarity
    for (i in 0 until iterations)
    {
        for(j in 0 until components.size)
        {
            //for j-th region
            neighbourList.add(Pair(components[j][0],mutableListOf()))
            components.forEachIndexed {index, componentList -> if(index!=j){
                var bb = getBoundingBox((componentList + components[j]) as MutableList<Pair<Int, Int>>)
                var b1 = getBoundingBox(components[j])
                var b2 = getBoundingBox(componentList)
                if(bb.area() != 0 && bb.area() < b1.area() + b2.area())
                {
                    neighbourList[j].second.add(components[j][0])
                    similarities.add(Triple(componentList,components[index],calculateSimilarity(componentList,components[index],image)))
                }
            }}
//            components.forEachIndexed {index, componentList -> if(index!=j){similarities.add(Triple(componentList,components[index],calculateSimilarity(componentList,components[index],image)))} } //for every pair of regions(components) it calculates it's similarity
        } //id probably makes duplicates gotta fix that later
    }
    similarities.sortBy { it.third }
    while(similarities.isNotEmpty())
    {
        var maxSimilarity = similarities.maxBy { it.third }
        var mergedComponents = maxSimilarity.first + maxSimilarity.second as MutableList<Pair<Int, Int>>
        similarities.removeIf { it.first == maxSimilarity.first || it.second == maxSimilarity.first || it.first == maxSimilarity.second || it.second == maxSimilarity.second}
        var neighboursMerge = neighbourList.first() { it.first == maxSimilarity.first}!!.second.addAll(neighbourList.first() { it.first == maxSimilarity.second}!!.second) as Pair<Pair<Int, Int>, MutableList<Pair<Int, Int>>>
        neighbourList.removeIf { it.first == maxSimilarity.first[0] || it.second == maxSimilarity.first[0] || it.first == maxSimilarity.second[0] || it.second == maxSimilarity.second[0]}
        neighbourList = neighbourList.map { Pair(it.first,it.second.map { if(it == maxSimilarity.second[0]) maxSimilarity.first[0] else it }) }.toMutableList() as MutableList<Pair<Pair<Int, Int>, MutableList<Pair<Int, Int>>>>// replacing all neighbors where there was second of two regions merged with the first coordinate of first of two regions
        neighbourList.add(neighboursMerge)
        neighboursMerge.second.forEach { similarities.add(Triple(mergedComponents,,calculateSimilarity(neighboursMerge.first,it,image))) }
    }

    return
}*/


//later do that so you dont have to calculate it each time

fun GEMINIcalculateSimilarity(
    component1: MutableList<Pair<Int, Int>>,
    component2: MutableList<Pair<Int, Int>>,
    image: IntArray,
    width: Int,
    height: Int
): Double {
    var similarity = 0.0 // Corrected typo: similairty -> similarity

    // --- Setup for floating-point calculations ---
    val size1 = component1.size.toDouble()
    val size2 = component2.size.toDouble()
    val totalPixels = (width * height).toDouble()
    val BINS = 30
    val BIN_SIZE = 256.0 / BINS // More precise bin divisor

    // --- Color/Intensity Similarity ---
    var histogram1 = MutableList<Int>(BINS) { 0 }
    component1.forEach {
        // Use BIN_SIZE for calculation and coerce to ensure index is valid
        val index = (image[it.first * width + it.second] / BIN_SIZE).toInt().coerceIn(0, BINS - 1)
        histogram1[index]++
    }

    var histogram2 = MutableList<Int>(BINS) { 0 }
    component2.forEach {
        val index = (image[it.first * width + it.second] / BIN_SIZE).toInt().coerceIn(0, BINS - 1)
        histogram2[index]++
    }

    // Calculation Fix: Use normalized frequencies (counts / region size) for intersection.
    var colorIntersection = 0.0
    histogram1.forEachIndexed { index, count1 ->
        val freq1 = count1.toDouble() / size1
        val freq2 = histogram2[index].toDouble() / size2

        // Sum of minimums (Histogram Intersection Kernel)
        colorIntersection += minOf(freq1, freq2)
    }

    // Apply weight
    similarity += 10.0 * colorIntersection

    // --- Texture Similarity (Skipped) ---

    // --- Size Similarity ---

    // Fix: Convert numerator and denominator to Double to prevent integer division.
    val sizeSum = size1 + size2
    similarity += 1.0 * (1.0 - (sizeSum / totalPixels))

    // --- Fill Similarity ---

    // Fix: Convert calculation to Double. Assumes getBoundingBox().area() returns Int.
    val boundingBoxArea =
        getBoundingBox((component1 + component2) as MutableList<Pair<Int, Int>>).area().toDouble()
    val emptySpace = boundingBoxArea - sizeSum

    similarity += 1.0 - (emptySpace / totalPixels)


    return similarity
}
/*fun calculateSimilarity(
    component1: MutableList<Pair<Int, Int>>,
    component2: MutableList<Pair<Int, Int>>,
    image: IntArray,
    width: Int,
    height: Int
): Double {
    var similairty = 0.0

    //color similarity
    var histogram1 = MutableList<Int>(30) {0} //30 bins where each represents a brightness
    component1.forEach {histogram1[((image[it.first * width + it.second])/8.5).toInt()]++}
    var histogram2 = MutableList<Int>(30) {0}
    component2.forEach {histogram2[((image[it.first * width + it.second])/8.5).toInt()]++}

//    println(histogram1)

    histogram1.forEachIndexed { index, it ->   similairty += 10 * min(it/30,histogram2[index])/30}//  /30 - ?

    //texture similarity skip for now

    //size similarity
    similairty += 1 * (1 - ((component1.size + component2.size)/(height*width)))

    //fill similarity
    similairty += 1 - (((getBoundingBox((component1+component2) as MutableList<Pair<Int, Int>>).area()) - component1.size - component2.size)/(height*width))


    return similairty
}*/


fun getBoundingBox(
    component: MutableList<Pair<Int, Int>>,
) : BoundingBox
{
    var maxBoundingBox = BoundingBox(100000,100000,-1,-1)//1 should be top left so it's first set opposite
    component.forEach {
        if (it.second < maxBoundingBox.x1) maxBoundingBox.x1 = it.second
        if (it.second > maxBoundingBox.x2) maxBoundingBox.x2 = it.second

        if (it.first < maxBoundingBox.y1) maxBoundingBox.y1 = it.first
        if (it.first > maxBoundingBox.y2) maxBoundingBox.y2 = it.first

    }
    return maxBoundingBox
}

/*
I tried making components contain only a vertice(as in paper) not entire edge but i came across problem of not having weights in msp
/*

/*var componentsNew = (edges.map { it.v1 } + edges.map { it.v2 }).distinct().map { mutableListOf(it) }
    for(i in 0 until edges.size)
    {
        val C1 = componentsNew.first() { if(it.firstOrNull() { it == edges[i].v1 }!= null) true else false }!!
        val C2 = componentsNew.first() { if(it.firstOrNull() { it == edges[i].v2 }!= null) true else false }!!
        if(C1 != C2 && edges[i].weight <= MIntDiffNew(C1,C2,k))
        {
            println(C1 != C2)
            components[i] = (C1 + C2) as MutableList<Edge>
        }
    }*/


 */
fun MIntDiffNew(C1: MutableList<Pair<Int, Int>>, C2: MutableList<Pair<Int, Int>>, k: Int) : Int
{
    return min(internalDifferenceNew(C1) + k/C1.size,internalDifferenceNew(C2) + k/C2.size)
}
fun internalDifferenceNew(C : MutableList<Pair<Int, Int>>) : Int
{
    return minimumSpanningTreeNew(C).maxBy { it.weight }.weight

}
fun minimumSpanningTreeNew(C: MutableList<Pair<Int, Int>>) : MutableList<Edge>
{
    val visitedEdges = mutableListOf<Edge>()
    val D = C.sortedBy { it.weight }.toMutableList()
    val unvisitedVertices = (D.map { it.v1 } + D.map { it.v2 }).distinct().toMutableList()
    visitedEdges.add(D.first())
    unvisitedVertices.remove(D.first().v1)
    unvisitedVertices.remove(D.first().v2)
    D.removeFirst()
    while(unvisitedVertices.isNotEmpty()) {
        val next1 = D.first { first -> visitedEdges.find { it.v1 == first.v1 || it.v1 == first.v2 || it.v2 == first.v1 || it.v2 == first.v2} != null && (unvisitedVertices.find { first.v1 == it } != null || unvisitedVertices.find { first.v2 == it } != null)}
        //println("next $next1     $unvisitedVertices")
        visitedEdges.add(next1)
        unvisitedVertices.remove(next1.v1)
        unvisitedVertices.remove(next1.v2)
        D.remove(next1)



    }
    return visitedEdges
}
*/



fun fasterImageSegmentation(image : IntArray,width : Int,height: Int,k : Int): MutableMap<Int, MutableList<Pair<Int, Int>>> { //implemented with union find
//later might add ranking to make tree shallow and fasten thing even more
//    var components: List<Pair<MutableList<Pair<Int, Int>>,Int>> // it is a list of pairs containing list of vertices and a threshold number
    val edges = mutableListOf<Edge>()   //list of edges
    var parenthood = IntArray(height*width)  //list of parent indexes
    var sizes = IntArray(height*width)   //list 'families' sizes
    var thresholds = FloatArray(height*width)   //list of thresholds
    for (i in 0 until height)
    {
        for (j in 0 until width)
        {
            if(i<height - 1) edges.add(Edge(Pair(i,j),Pair(i+1,j), edgeWeight(image[i*width + j],image[(i+1)*(width) + j])))
            if(j<width - 1) edges.add(Edge(Pair(i,j),Pair(i,j+1), edgeWeight(image[i*width + j],image[i*width + j+1])))
        }
    }
    edges.sortBy { it.weight }
    for (i in 0 until height*width) {
        parenthood[i] = (i)
        thresholds[i] = (k.toFloat())
        sizes[i] = (1)
    }
    for (i in 0 until edges.size) {
        var currentParent = edges[i].v1.first * width + edges[i].v1.second
        while (parenthood[currentParent] != currentParent) //finding function
        {
            currentParent = parenthood[currentParent]
        }
        parenthood[edges[i].v1.first * width + edges[i].v1.second] = currentParent
        val parent1 = currentParent

        currentParent = edges[i].v2.first * width + edges[i].v2.second
        while (parenthood[currentParent] != currentParent) //finding function
        {
            currentParent = parenthood[currentParent]
        }
        parenthood[edges[i].v2.first * width + edges[i].v2.second] = currentParent
        val parent2 = currentParent

        if (parent1==parent2) continue
        if(edges[i].weight > thresholds[parent1] || edges[i].weight > thresholds[parent2]) continue
        /*
                parenthood[parent1] = parent2
                sizes[parent2] += sizes[parent1]
                thresholds[parent2] = edges[i].weight + k/sizes[parent2]*/

        if (sizes[parent1] < sizes[parent2]) {
            parenthood[parent1] = parent2
            sizes[parent2] += sizes[parent1]
            thresholds[parent2] = (edges[i].weight + k / sizes[parent2]).toFloat()
        } else {
            parenthood[parent2] = parent1
            sizes[parent1] += sizes[parent2]
            thresholds[parent1] = (edges[i].weight + k / sizes[parent1]).toFloat()
        }

    }

    val tempMap = mutableMapOf<Int,MutableList<Pair<Int,Int>>>()
    for (i in 0 until height)
    {
        for (j in 0 until width)
        {
            var currentParent = i * width + j
            while (parenthood[currentParent] != currentParent) //finding function
            {
                currentParent = parenthood[currentParent]
            }
            parenthood[i * width + j] = currentParent
            tempMap.getOrPut(currentParent) { mutableListOf() }.add(Pair(i,j))
        }
    }
    return tempMap/*.values.toMutableList()*/
}

/**
 * Reprezentuje krawędź w grafie podobieństwa. Służy jako element Max-Heap.
 * r1 i r2 to ID połączonych regionów.
 */
private data class SimilarityEdge(val r1: Int, val r2: Int, val similarity: Double) : Comparable<SimilarityEdge> {
    // Implementacja dla Max-Heap: sortowanie malejące (największa podobieństwo na górze)
    override fun compareTo(other: SimilarityEdge): Int {
        // Porównanie z odwrotnej kolejności
        return other.similarity.compareTo(this.similarity)
    }
}

fun GEMINIbetterSelectiveSearch(
    components: MutableMap<Int, MutableList<Pair<Int, Int>>>,
    image: IntArray,
    width: Int,
    height: Int
): MutableList<MutableList<Pair<Int, Int>>> {

    // --- STRATEGIA SZYBKOŚCI: Użycie PriorityQueue (Max-Heap) dla O(log N) ---
    val edgeQueue = PriorityQueue<SimilarityEdge>()

    // Mapa do śledzenia sąsiadów (zastępuje starą mapę similarities, która była używana do iteracji)
    // Używamy jej teraz tylko do aktualizacji po fuzji.
    val neighborMap = components.keys.associateWith { mutableSetOf<Int>() }.toMutableMap()

    // Zestaw aktywnych ID regionów, używany do szybkiej walidacji krawędzi z kolejki
    val activeRegions = components.keys.toMutableSet()

    // Lista wszystkich propozycji regionów
    val allComponents = components.values.toMutableList()

    // Tablica mapująca pozycję piksela na ID jego początkowego regionu.
    val parenthood = IntArray(height * width)
    components.forEach { (regionId, pixels) ->
        pixels.forEach { (r, c) ->
            parenthood[r * width + c] = regionId
        }
    }

    // Zestaw par, aby uniknąć dodawania tej samej krawędzi (R1, R2) więcej niż raz do kolejki
    val addedEdges = mutableSetOf<Pair<Int, Int>>()

    // --- 1. Inicjalizacja Podobieństw i Max-Heap ---
    for (i in 0 until height) {
        for (j in 0 until width) {
            val p1 = parenthood[i * width + j]

            // Funkcja pomocnicza do przetwarzania sąsiada p2
            fun processNeighbor(p2: Int) {
                if (p1 == p2) return

                val pair = if (p1 < p2) Pair(p1, p2) else Pair(p2, p1)

                // Dodanie krawędzi do Max-Heap tylko raz
                if (addedEdges.add(pair)) {
                    val sim = GEMINIcalculateSimilarity(components[p1]!!, components[p2]!!, image, width, height)
                    edgeQueue.add(SimilarityEdge(p1, p2, sim))
                }

                // Aktualizacja mapy sąsiadów
                neighborMap.getOrPut(p1) { mutableSetOf() }.add(p2)
                neighborMap.getOrPut(p2) { mutableSetOf() }.add(p1)
            }

            // Sprawdzenie sąsiada poniżej
            if (i < height - 1) {
                processNeighbor(parenthood[(i + 1) * width + j])
            }

            // Sprawdzenie sąsiada po prawej
            if (j < width - 1) {
                processNeighbor(parenthood[i * width + j + 1])
            }
        }
    }

    // --- 2. Hierarchiczne Łączenie Regionów (Max-Heap Loop) ---

    // Bezpieczne ustawienie ID dla kolejnego nowego regionu
    var nextId = (components.keys.maxOrNull() ?: 0) + 1

    while (edgeQueue.isNotEmpty()) {

        // 2a. Pobierz krawędź o największym podobieństwie w O(log N)
        val maxEdge = edgeQueue.poll() ?: break

        val regionA = maxEdge.r1
        val regionB = maxEdge.r2

        // Walidacja: Jeśli któryś region nie jest już aktywny, krawędź jest nieaktualna. Pomijamy.
        if (!activeRegions.contains(regionA) || !activeRegions.contains(regionB)) {
            continue
        }

        val newRegionId = nextId++

        // 2b. Łączenie Pikseli
        val mergedRegionPixels = components[regionA]!!.toMutableList().apply {
            addAll(components[regionB].orEmpty())
        }

        // Zapis nowego regionu i aktualizacja zestawu aktywnych ID
        components[newRegionId] = mergedRegionPixels
        allComponents.add(mergedRegionPixels)

        // 2c. Aktualizacja Aktywnych Regionów
        activeRegions.remove(regionA)
        activeRegions.remove(regionB)
        activeRegions.add(newRegionId)

        // 2d. Aktualizacja Mapy Sąsiadów i Recalculation

        val neighborsA = neighborMap.remove(regionA) ?: emptySet()
        val neighborsB = neighborMap.remove(regionB) ?: emptySet()

        // Zbierz unikalny zestaw ID sąsiadów, którzy są nadal aktywni i nie są nowym regionem
        val combinedNeighboursIds = (neighborsA + neighborsB)
            .filter { activeRegions.contains(it) && it != newRegionId }
            .toSet()

        // Usuń stare krawędzie z map sąsiadów
        combinedNeighboursIds.forEach { neighbourId ->
            neighborMap[neighbourId]?.remove(regionA)
            neighborMap[neighbourId]?.remove(regionB)
        }

        // Zainicjuj nowy region w mapie sąsiadów
        neighborMap[newRegionId] = mutableSetOf()

        // Oblicz i dodaj nowe krawędzie do Max-Heap i mapy sąsiadów
        for (neighbourId in combinedNeighboursIds) {
            val similarity = GEMINIcalculateSimilarity(components[neighbourId]!!, components[newRegionId]!!, image, width, height)

            // Dodaj nową krawędź do Max-Heap
            edgeQueue.add(SimilarityEdge(newRegionId, neighbourId, similarity))

            // Zaktualizuj mapę sąsiadów
            neighborMap[newRegionId]!!.add(neighbourId)
            neighborMap[neighbourId]!!.add(newRegionId)
        }

        // Usuń stare wpisy z mapy components
        components.remove(regionA)
        components.remove(regionB)
    }

    return allComponents
}

fun betterSelectiveSearch(components : MutableMap<Int, MutableList<Pair<Int, Int>>>,image : IntArray,width : Int,height: Int) : MutableList<MutableList<Pair<Int, Int>>>
{
    var similarities = mutableMapOf<Int,MutableList<Pair<Int,Double>>>()  //map containing neighbours of each region and similarity with them
    var parenthood = IntArray(height*width)  //list of parent indexes
    var allComponents = components.values.toMutableList()
    components.forEach{region ->
        region.value.forEach {
            parenthood[it.first * width + it.second] = region.key
        }
    }


    for (i in 0 until height)
    {
        for (j in 0 until width)
        {
            var parent1 = parenthood[i*width + j]
            var parent2 = 0
            if(i<height - 1){
                parent2 = parenthood[(i + 1) * (width) + j]
                if(parent1!=parent2) {
                    similarities.getOrPut(parent1) { mutableListOf() }.add(Pair(parent2,GEMINIcalculateSimilarity(components[parent1]!!,components[parent2]!!,image,width,height)))
                    similarities.getOrPut(parent2) { mutableListOf() }.add(Pair(parent1,GEMINIcalculateSimilarity(components[parent1]!!,components[parent2]!!,image,width,height)))
                }
            }
            if(j<width - 1)
                parent2 = parenthood[i*width + j+1]
            if(parent1!=parent2) {
                similarities.getOrPut(parent1) { mutableListOf() }.add(Pair(parent2,
                    GEMINIcalculateSimilarity(components[parent1]!!,components[parent2]!!,image,width,height)))
                similarities.getOrPut(parent2) { mutableListOf() }.add(Pair(parent1,GEMINIcalculateSimilarity(components[parent1]!!,components[parent2]!!,image,width,height)))
            }
        }
    }
//maybe later change that so it doesn't store the similarity doubled if possible
    var nextId = similarities.keys.max() + 1
    while (similarities.isNotEmpty())
    {                                                                                       //this (a,b) splits the pair
        var maxSimilarity = similarities.flatMap { (region,neighbours) -> neighbours.map { (neighbour,similarity) -> Triple(region,neighbour,similarity)} }.maxBy { it.third }
//        components[maxSimilarity.first]?.addAll(components[maxSimilarity.second].orEmpty()) //merging two regions


        var newId = nextId //technically not needed
        nextId++

        val mergedRegion = components[maxSimilarity.first]!!.apply { addAll(components[maxSimilarity.second].orEmpty()) }
        components[newId] = mergedRegion // Store the merged region under the new ID
        allComponents.add(mergedRegion)

        val combinedNeighboursIds = (similarities[maxSimilarity.first].orEmpty().map { it.first } +
                similarities[maxSimilarity.second].orEmpty().map { it.first })
            .filter { it != maxSimilarity.first && it != maxSimilarity.second }
            .toSet()

/*        val combinedNeighbours =
            (similarities[maxSimilarity.first].orEmpty() +
                    similarities[maxSimilarity.second].orEmpty())
                .distinct()*/



        similarities.remove(maxSimilarity.first)
        similarities.remove(maxSimilarity.second)
        components.remove(maxSimilarity.first)
        components.remove(maxSimilarity.second)



        similarities.forEach { it.value.removeIf { it.first==maxSimilarity.first || it.first== maxSimilarity.second} }

        combinedNeighboursIds.forEach {neighbour ->
            val similarity = GEMINIcalculateSimilarity(components[neighbour]!!,components[newId]!!,image,width,height)
            similarities.getOrPut(newId) { mutableListOf() }.add(Pair(neighbour,similarity))
            similarities.getOrPut(neighbour) { mutableListOf() }.add(Pair(newId,similarity))
        }
        similarities = similarities.filter { it.value.isNotEmpty()}.toMutableMap()

    }
//technically parenthood should be updated but it dosen't matter here



    return allComponents.toMutableList()
}




fun saveTensorAsPng(tensor: Tensor, outputPath: String) {
    require(tensor.c == 1) { "Expected grayscale tensor with c = 1" }

    val width = tensor.w
    val height = tensor.h

    val pixels = Array(height*width) { Pixel(1,1,1) }
    for (y in 0 until height){
        for (x in 0 until width){
            val idx = (y * width + x)

            var v = tensor.data[idx]

            // Allow both 0–1.0 and 0–255 input
            if (v <= 1.0) v *= 255.0

            val gray = v.coerceIn(0.0, 255.0).toInt()

            pixels[y*width + x] = Pixel(x,y,gray,gray,gray,255)
        }
    }

    val image = ImmutableImage.create(width, height,pixels)

    image.output(PngWriter.NoCompression, outputPath)
}



data class LabeledTensor(
    val tensor: Tensor,
    val label: Int
)

fun loadImageDataset(rootDir: String, size: Int): List<LabeledTensor> {
    val result = mutableListOf<LabeledTensor>()

    for (label in 0..9) {
        val folder = File("$rootDir/$label")
        if (!folder.exists() || !folder.isDirectory) {
            println("Skipping folder: $folder (not found)")
            continue
        }

        folder.listFiles { file -> file.extension.lowercase() in listOf("png", "jpg", "jpeg") }
            ?.forEach { file ->

                // Load image
                var img = ImmutableImage.loader().fromFile(file)

                // Convert to grayscale
                img = img.filter(GrayscaleFilter())

                // Resize to network input size
                img = img.scaleTo(size, size)

                val w = img.width
                val h = img.height
                val data = DoubleArray(w * h)

                val pixels = img.pixels()

                for (y in 0 until h) {
                    for (x in 0 until w) {
                        val idx = y * w + x
                        val p = pixels[idx]

                        // pixel gray already in 0..255 from grayscale filter
                        val gray = p.red() // since grayscale: r=g=b

                        // Normalize to 0..1
                        data[idx] = gray / 255.0
                    }
                }

                result.add(
                    LabeledTensor(
                        tensor = Tensor(data, h = h, w = w, c = 1),
                        label = label
                    )
                )
            }
    }

    return result
}

data class Detection(
    val box: BoundingBox,
    val score: Double,
    val classId: Int
)

fun nonMaximumSuppression(
    detections: List<Detection>,
    iouThreshold: Double = 0.7
): List<Detection> {
    if (detections.isEmpty()) return emptyList()

    // Sort detections by descending score
    val sorted = detections.sortedByDescending { it.score }.toMutableList()
    val finalDetections = mutableListOf<Detection>()

    while (sorted.isNotEmpty()) {
        val best = sorted.removeAt(0) // pick highest score
        finalDetections.add(best)

        // Remove all detections with high IoU with best
        val iterator = sorted.iterator()
        while (iterator.hasNext()) {
            val det = iterator.next()
            if (intersectionOverUnion(best.box, det.box) > iouThreshold && best.classId == det.classId) {
                iterator.remove()
            }
        }
    }

    return finalDetections
}

fun nonMaximumSuppressionRCNN(
    detections: List<Detection>,
    iouThreshold: Double = 0.7,
    scoreThreshold: Double = 0.0          // RCNN zazwyczaj filtruje przed NMS
): List<Detection> {

    if (detections.isEmpty()) return emptyList()

    // 1. Wstępne odfiltrowanie bardzo słabych detekcji (jak w RCNN)
    val filtered = detections.filter { it.score >= scoreThreshold }.toMutableList()
    if (filtered.isEmpty()) return emptyList()

    // 2. Sort by descending confidence
    val sorted = filtered.sortedByDescending { it.score }.toMutableList()
    val finalDetections = mutableListOf<Detection>()

    // 3. Class-agnostic NMS (najważniejsza zmiana!)
    while (sorted.isNotEmpty()) {
        val best = sorted.removeAt(0)
        finalDetections.add(best)

        val iterator = sorted.iterator()
        while (iterator.hasNext()) {
            val det = iterator.next()

            // RCNN usuwa boxy po samym IoU —
            // KLASY SĄ IGNOROWANE na tym etapie
            if (intersectionOverUnion(best.box, det.box) > iouThreshold) {
                iterator.remove()
            }
        }
    }

    return finalDetections
}


fun boundingBoxToTensor(
    image: IntArray,     // grayscale 0..255
    width: Int,          // full image width
    height: Int,         // full image height
    box: BoundingBox,
    targetSize: Int = 32 // output tensor size
): Tensor {

    // ---- 1. Clamp box inside the image ----
    val x1 = box.x1.coerceIn(0, width - 1)
    val y1 = box.y1.coerceIn(0, height - 1)
    val x2 = box.x2.coerceIn(0, width - 1)
    val y2 = box.y2.coerceIn(0, height - 1)

    val bw = x2 - x1 + 1
    val bh = y2 - y1 + 1

    if (bw <= 0 || bh <= 0) {
        // Empty → return blank tensor
        return Tensor(DoubleArray(targetSize * targetSize), targetSize, targetSize, 1)
    }

    // ---- 2. Extract box pixels into buffer ----
    val crop = DoubleArray(bw * bh)

    var idx = 0
    for (yy in y1..y2) {
        val rowOffset = yy * width
        for (xx in x1..x2) {
            val pixel = image[rowOffset + xx]
            crop[idx++] = pixel / 255.0
        }
    }

    // ---- 3. Resize crop → targetSize×targetSize (nearest neighbor) ----
    val out = DoubleArray(targetSize * targetSize)
    val scaleX = bw.toDouble() / targetSize
    val scaleY = bh.toDouble() / targetSize

    for (ty in 0 until targetSize) {
        val srcY = (ty * scaleY).toInt().coerceIn(0, bh - 1)
        val srcYOff = srcY * bw
        for (tx in 0 until targetSize) {
            val srcX = (tx * scaleX).toInt().coerceIn(0, bw - 1)
            out[ty * targetSize + tx] = crop[srcYOff + srcX]
        }
    }

    return Tensor(out, targetSize, targetSize, 1)
}