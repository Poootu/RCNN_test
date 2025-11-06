package org.example

import com.sksamuel.scrimage.ImmutableImage
import com.sksamuel.scrimage.filter.GaussianBlurFilter
import com.sksamuel.scrimage.nio.PngWriter
import java.io.File
import kotlin.collections.forEach
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random
fun test1()
{
    val imageFile = File("src/main/resources/img.png")
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

    val segmentedGraph = (imageSegmentation(newImage as MutableList<MutableList<Int>>,300))
    val segmentedList = vertsToList(segmentedGraph,inputImage.height,inputImage.width)
    val segmentedImage = inputImage.map { p -> java.awt.Color((segmentedList[p.y][p.x].first),(segmentedList[p.y][p.x].second),(segmentedList[p.y][p.x].third)) }

    segmentedImage.output(PngWriter.NoCompression,File("src/main/resources/saved/s1.png"))
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







class BoundingBox(val x1 :Int, val y1 :Int, val x2 :Int, val y2 :Int) {
    fun intersection(other: BoundingBox): BoundingBox {
        return BoundingBox(
            x1 = max(x1,other.x1),
            y1 = max(y1,other.y1),
            x2 = min(x2,other.x2),
            y2 = min(y2,other.y2)
            )
    }
    fun area() : Int
    {
        return if(x2 < x1 || y2 < y1) 0
        else (x2-x1) * (y2-y1)
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
    return abs(i1-i2) //change that later ig
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



