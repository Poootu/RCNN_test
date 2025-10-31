package org.example.networkStructure

import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.lang.Thread.sleep
import java.util.Collections.shuffle
import kotlin.math.exp
import kotlin.random.Random

class Network {

    var layers: MutableList<Layer>

    constructor(inputNumber: Int, outputNumber: Int, hiddenLayerNumber: Int, hiddenLayerSize: Int, onlyZeros: Boolean) {
        layers = mutableListOf<Layer>()
        layers.add(Layer(inputNumber, 0, onlyZeros))

        var s = inputNumber
        for (i in 0 until hiddenLayerNumber) {
            layers.add(Layer(hiddenLayerSize, s, onlyZeros))
            s = hiddenLayerSize
        }

        layers.add(Layer(outputNumber, s, onlyZeros))

    }

    constructor(l: MutableList<Layer>) {
        layers = l
    }


    fun feedforward(input : MutableList<Double>) : MutableList<Double> {
        var previousLayer = input
        for (i in 1 until layers.size) {
//            println(" $i - $previousLayer")
//            println(" $i - ${layers[1].neurons[0]}")
            var list = mutableListOf<Double>()
            for (j in 0 until layers[i].neurons.size)
            {
                var sum = layers[i].neurons[j].bias
                for (k in 0 until layers[i].neurons[j].weights.size)
                {
                    sum+= layers[i].neurons[j].weights[k] * previousLayer[k]
                }
                list.add(sigmoid(sum))
            }
            previousLayer = list
        }
        return previousLayer
    }
    suspend fun SGD(
        trainingData: MutableList<Pair<MutableList<Double>, MutableList<Double>>>,
        miniBatchSize: Int,
        learningRate: Double
    ) //stochastic gradient descent
    {
//        shuffle(trainingData.toList())
            coroutineScope {
        for (i in 0 until trainingData.size step miniBatchSize) {
//            launch {
            val miniBatch = trainingData.toList().subList(i,i + miniBatchSize)
            updateMiniBatch(miniBatch, learningRate)
            println(i)
//            println(miniBatch)

//            }

        }
            }
//        println("trainingData: ${trainingData[0]}")
//        println("trainingData: ${trainingData.subList(0,1)}")
/*        for (i in 0 until trainingData.size) {
            if(trainingData[i].second.first() == 1.0) updateMiniBatch(trainingData.subList(i,i+1), learningRate)

        }*/


    }

    private fun updateMiniBatch(miniBatch: List<Pair<MutableList<Double>, MutableList<Double>>>, learningRate: Double) {
        //new weight = old_weight - (learning_rate * (sum of (changes to cost/changes to weight)) / examples in a batch
        //new bias = old_bias - (learning_rate * (sum of (changes to cost/changes to bias)) / examples in a batch
//            println("$miniBatch .")
        var sum_gradient =
            Network(layers[0].neurons.size, layers.last().neurons.size, layers.size - 2, layers[1].neurons.size, true)
        //pusta sieć jako gradient do aktualizowania
        for (i in miniBatch.indices) {
            sum_gradient += backpropagation(miniBatch[i]) //druga warstwa się nie zmienia chyba
//            println("--------------------")
//            println(sum_gradient.layers[2])
//            println("========================")
        }
//            println(sum_gradient)
        for (i in layers.indices)
        {
            for (j in layers[i].neurons.indices)
            {
                layers[i].neurons[j].bias -= (learningRate/miniBatch.size) * sum_gradient.layers[i].neurons[j].bias
                for (k in layers[i].neurons[j].weights.indices)
                {
                    layers[i].neurons[j].weights[k] -= (learningRate/miniBatch.size) * sum_gradient.layers[i].neurons[j].weights[k]
                }
            }
        }

    }

            /*private*/ fun backpropagation(trainingData: Pair<MutableList<Double>, MutableList<Double>>) : Network {
            val gradient = Network(layers[0].neurons.size, layers.last().neurons.size, layers.size - 2, layers[1].neurons.size, true)
/*            println("{{{{{{")
            println(gradient)
            println("}}}}}}}")*/
                //pusta siec jako gradient do aktualizowania



            var activations = mutableListOf<MutableList<Double>>()
            val weighted_inputs = mutableListOf<MutableList<Double>>()
            activations.add(trainingData.first)
            weighted_inputs.add(activations[0])

            for (i in 1 until layers.size)   //feedforward
            {
                var activation = mutableListOf<Double>()
                var weighted_input = mutableListOf<Double>()
                for (j in layers[i].neurons.indices)
                {
                    var sum = layers[i].neurons[j].bias
                    for (k in layers[i].neurons[j].weights.indices)
                    {
                        sum += layers[i].neurons[j].weights[k]*activations[i-1][k]
                    }
                    weighted_input.add(sum)
//                    println(weighted_input)
                    activation.add(sigmoid(sum))
//                    println(activation)
//                println(sigmoid(sum))
                }
                weighted_inputs.add(weighted_input)
                activations.add(activation)


//                println(weighted_input)
            }
//                println(activations)


            //backpropagation
            //hmm why delta

            var delta = cost_derivative(activations.last(),trainingData.second).mapIndexed { index, d -> d*sigmoid_derivative(weighted_inputs.last()[index]) }.toMutableList() // /sigmoid prime
            for (g in gradient.layers.last().neurons.indices) gradient.layers.last().neurons[g].bias = delta[g]
/*            gradient.layers.last().neurons.mapIndexed {index, neuron ->
                neuron.bias = delta[index]
            }*/
            for (g in gradient.layers.last().neurons.indices) gradient.layers.last().neurons[g].weights = gradient.layers.last().neurons[g].weights.mapIndexed { weightIndex, weight -> delta[g]*activations[activations.size-2][weightIndex] }.toMutableList()
//                println(gradient.layers.last().neurons)
                /*            gradient.layers.last().neurons.mapIndexed {index, neuron ->
                neuron.weights.mapIndexed { weightIndex, weight -> delta[index]*activations[activations.size-2][weightIndex] }
            }*/
//            println(activations)
//            println(delta)
            for (i in 2 until layers.size)  // .. or until?????
            {
//            println(delta)
                var tempDelta = MutableList(layers[layers.size-i].neurons.size){0.0}
//                println(tempDelta)

                for (j in 0 until layers[layers.size - i].neurons.size) {
                    var sum = 0.0
                    for (k in 0 until delta.size)
                    {
                        sum += delta[k] * layers[layers.size-i+1].neurons[k].weights[j]
                    }
                    sum *= sigmoid_derivative(
                        weighted_inputs[layers.size - i][j]
                    )
                    tempDelta[j] = sum


/*                    var sum = 0.0
                    for (k in 0 until layers[layers.size - i + 1].neurons.size) {
                        sum += layers[layers.size - i + 1].neurons[k].weights[j] * delta[k] * sigmoid_derivative(
                            weighted_inputs[layers.size - i][k]
                        ) //emm co się dzieje z sumą i czy to w ogóle ma sens
                    }*/
                }

                delta = tempDelta
//                delta = delta.mapIndexed { index, d -> d }.toMutableList()
                for (g in gradient.layers[layers.size - i].neurons.indices) {
                    gradient.layers[layers.size - i].neurons[g].bias = delta[g]
                }
                /*                gradient.layers[layers.size - i].neurons.forEachIndexed {index, neuron ->
                                    neuron.bias = delta[index]
                                }*/
                for (g in gradient.layers[layers.size - i].neurons.indices) {
/*                    for (k in gradient.layers[layers.size - i].neurons[g].weights.indices) {
                        println(
                            delta[g]
                        )
                        println("pozdro")
                        println(
                            activations[activations.size - i - 1][k]
                        )
                    }*/
                    gradient.layers[layers.size - i].neurons[g].weights =
                        gradient.layers[layers.size - i].neurons[g].weights.mapIndexed { weightIndex, weight -> delta[g] * activations[activations.size - i - 1][weightIndex] }
                            .toMutableList()

                }

/*                for (g in gradient.layers[layers.size - i].neurons.indices) {
                    for (k in gradient.layers[layers.size - i].neurons[g].weights.indices) {
                        gradient.layers[layers.size - i].neurons[g].weights[k] =
                            delta[g] * activations[activations.size - i - 1][k]
                    }
                }*/
//                println(gradient.layers[layers.size - i].neurons)


                /*                gradient.layers[layers.size - i].neurons.forEachIndexed {index, neuron ->
                                    neuron.weights.mapIndexed { weightIndex, weight -> delta[index]*activations[activations.size-i-1][weightIndex] }
                                }*/
                /*runBlocking {
//                    println(delta.size)
//                    println( gradient.layers.last().neurons.size)
                    println(gradient.layers[layers.size - i].neurons)
                    sleep(3000)
                }*/
            }

            return gradient
    }


    fun evaluate(testData : MutableList<Pair<MutableList<Double>, MutableList<Double>>>)
    {
        //softmax?
        var stats = MutableList<Int>(10) {0}
        var correct = 0
        for(i in testData.indices)
        {
            val output = feedforward(testData[i].first)
//            println(" $i - ${output}")
            val expectedOutput = testData[i].second
            for(j in 0..9)
            {
                if(output[j] == output.max())
                {
                    if(expectedOutput[j] == 1.0)
                    {
                        correct++
                        stats[j]++
                    }
                       break
                }
            }
        }
        println("Correct: $correct / ${testData.size}")
        println("stats: $stats")
    }

    fun cost_derivative(outputs : MutableList<Double>, targetValues : MutableList<Double>): MutableList<Double> {
        val difference = outputs.toMutableList()
        for (i in difference.indices)
        {
            difference[i] = difference[i] - targetValues[i]
        }
        return difference
    }

    fun sigmoid(value : Double): Double {
        return 1.0/(1.0 + exp(-value))
    }
    fun sigmoid_derivative(value : Double): Double {
        return sigmoid(value)*(1-sigmoid(value))
    }

    operator fun plus(other: Network): Network {
        //only if same size
        var newNetwork =
            Network(layers[0].neurons.size, layers.last().neurons.size, layers.size - 2, layers[1].neurons.size, true)
        for (i in layers.indices) {
            for (j in layers[i].neurons.indices) {
                newNetwork.layers[i].neurons[j].bias = this.layers[i].neurons[j].bias + other.layers[i].neurons[j].bias
                for (k in layers[i].neurons[j].weights.indices) {
                    newNetwork.layers[i].neurons[j].weights[k] =
                        this.layers[i].neurons[j].weights[k] + other.layers[i].neurons[j].weights[k]
                }
            }
        }
        return newNetwork
    }

    override fun toString(): String {
        return layers.joinToString("\n") { it.toString() }
    }
    fun toFileString(): String {
        return layers.joinToString("\n") { it.toFileString() }
    }
}


class Layer {

    var neurons: MutableList<Neuron>

    constructor(numberOfNeurons: Int, inputNumber: Int, onlyZeros: Boolean) {
        neurons = MutableList(numberOfNeurons) {Neuron(inputNumber,onlyZeros)}
    }

    constructor(n: MutableList<Neuron>) {
        neurons = n
    }

    override fun toString(): String {
        return neurons.joinToString(" ") { it.toString() }
    }
    fun toFileString(): String {
        return neurons.joinToString(" ") { "{${it.toFileString()}}" }
    }
}


class Neuron {


    var weights : MutableList<Double>
    var bias : Double


    constructor(numberOfWeights : Int,onlyZeros:Boolean)
    {
        weights = MutableList(numberOfWeights) { if (onlyZeros) 0.0 else Random.nextDouble(-1.0,1.0) }
        bias = if (onlyZeros) 0.0 else Random.nextDouble(-1.0,1.0)
    }

    constructor(w : MutableList<Double>, b : Double)
    {
        weights = w
        bias = b
    }
    override fun toString(): String {
        return (weights + bias).toString()
    }
    fun toFileString(): String {
        return (weights + bias).joinToString(",") { it.toString() }
    }
}

class Constructors {


    constructor(i: Int) {
    }
}