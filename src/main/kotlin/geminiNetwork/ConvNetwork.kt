package org.example

import kotlinx.coroutines.coroutineScope
import java.lang.IllegalArgumentException
import kotlin.math.exp
import kotlin.random.Random

// --- Helper Type and Data Classes ---

// Type alias for 3D Feature Maps: [Channel][Height][Width]
typealias Tensor3D = Array<Array<Array<Double>>>

data class ForwardPassResult(val output: Any, val z: Any)

fun sigmoid(value : Double): Double {
    return 1.0 / (1.0 + exp(-value))
}
// --- 1. Interface for AllGLayers ---

interface ImageLayer {
    // Forward pass without gradient saving (for regular inference)
    fun forward(input: Any): Any

    // Forward pass with gradient saving (for training/backpropagation)
    fun forwardAndSave(input: Any): ForwardPassResult

    // Calculates and accumulates dL/dW and dL/dB (for layers with parameters)
    fun backward(delta: Any, activationPrevious: Any)

    // Calculates dL/dX for the previous layer (returns the delta to be used by layer i-1)
    fun calculateDeltaForPreviousLayer(delta: Any, z: Any, previousLayerOutputSize: Int): Any

    // Applies accumulated gradients (deltaWeights, deltaBiases)
    fun applyGradients(learningRateFactor: Double)

    // Resets accumulated gradients for the next mini-batch
    fun resetGradients()

    fun getInputSize(): Int
    fun getOutputSize(): Int

    override fun toString(): String
    fun toFileString(): String
}


// --- 2. Fully-Connected (FC) Core Classes (Modified) ---

class GNeuron {
    var weights : MutableList<Double>
    var bias : Double

    // Fields for gradient accumulation
    var deltaWeights: MutableList<Double> = mutableListOf()
    var deltaBias: Double = 0.0

    constructor(numberOfWeights : Int, onlyZeros:Boolean) {
        val random = Random(System.currentTimeMillis())
        weights = MutableList(numberOfWeights) { if (onlyZeros) 0.0 else random.nextDouble(-1.0,1.0) }
        bias = if (onlyZeros) 0.0 else random.nextDouble(-1.0,1.0)

        deltaWeights = MutableList(numberOfWeights) { 0.0 }
        deltaBias = 0.0
    }

    constructor(w : MutableList<Double>, b : Double) {
        weights = w
        bias = b

        deltaWeights = MutableList(w.size) { 0.0 }
        deltaBias = 0.0
    }
    override fun toString(): String {
        return (weights + bias).toString()
    }
    fun toFileString(): String {
        return (weights + bias).joinToString(",") { it.toString() }
    }
}

class GLayer(numberOfNeurons: Int, var inputNumber: Int, onlyZeros: Boolean) : ImageLayer {

    var neurons: MutableList<GNeuron>

    init {
        neurons = MutableList(numberOfNeurons) { GNeuron(inputNumber, onlyZeros) }
    }

    override fun getInputSize(): Int = inputNumber
    override fun getOutputSize(): Int = neurons.size

    // FIX: Changed check to MutableList<*> to avoid type erasure issues
    override fun forward(input: Any): Any {
        if (input !is MutableList<*>) {
            throw IllegalArgumentException("FCGLayer expects MutableList input, got ${input::class.simpleName}.")
        }
        @Suppress("UNCHECKED_CAST")
        val inputList = input as MutableList<Double> // Safe cast based on network flow

        var list = mutableListOf<Double>()
        for (j in 0 until neurons.size) {
            var sum = neurons[j].bias
            for (k in 0 until neurons[j].weights.size) {
                sum += neurons[j].weights[k] * inputList[k]
            }
            list.add(GNetwork.sigmoid(sum))
        }
        return list
    }

    // FIX: Changed check to MutableList<*> to avoid type erasure issues
    override fun forwardAndSave(input: Any): ForwardPassResult {
        if (input !is MutableList<*>) {
            throw IllegalArgumentException("FCGLayer forwardAndSave expects MutableList input, got ${input::class.simpleName}.")
        }
        @Suppress("UNCHECKED_CAST")
        val inputList = input as MutableList<Double> // Safe cast based on network flow

        val zList = mutableListOf<Double>()
        val activationList = mutableListOf<Double>()

        for (j in 0 until neurons.size) {
            var z = neurons[j].bias
            for (k in 0 until neurons[j].weights.size) {
                z += neurons[j].weights[k] * inputList[k]
            }
            zList.add(z)
            activationList.add(GNetwork.sigmoid(z))
        }
        return ForwardPassResult(activationList, zList)
    }

    // Backward pass for FC layer (delta is a 1D list of errors)
    override fun backward(delta: Any, activationPrevious: Any) {
        if (delta !is MutableList<*> || activationPrevious !is MutableList<*>) {
            throw IllegalArgumentException("FCGLayer backward expected 1D lists.")
        }
        @Suppress("UNCHECKED_CAST")
        val deltaList = delta as MutableList<Double>
        @Suppress("UNCHECKED_CAST")
        val prevActivationList = activationPrevious as MutableList<Double>

        for (j in neurons.indices) {
            // Accumulate bias gradient
            neurons[j].deltaBias += deltaList[j]
            // Accumulate weight gradients: dC/dw = a_L-1 * delta_L
            for (k in neurons[j].weights.indices) {
                neurons[j].deltaWeights[k] += deltaList[j] * prevActivationList[k]
            }
        }
    }

    // Calculates delta for the previous layer (dL/dX)
    override fun calculateDeltaForPreviousLayer(delta: Any, z: Any, previousLayerOutputSize: Int): Any {
        if (delta !is MutableList<*>) {
            throw IllegalArgumentException("FC Delta calculation expected 1D list delta.")
        }
        @Suppress("UNCHECKED_CAST")
        val deltaList = delta as MutableList<Double>

        // New delta for the previous layer: W^T * delta_L
        val newDelta = MutableList(previousLayerOutputSize) { 0.0 }

        // W^T * delta_L
        for (k in 0 until previousLayerOutputSize) { // Index of previous layer's output
            var sum = 0.0
            for (j in neurons.indices) { // Index of current layer's neuron
                sum += deltaList[j] * neurons[j].weights[k]
            }
            newDelta[k] = sum
        }

        // This returned list is either the input to a Flatten layer or the input to a previous FC layer
        return newDelta
    }

    override fun applyGradients(learningRateFactor: Double) {
        for (neuron in neurons) {
            neuron.bias -= learningRateFactor * neuron.deltaBias
            neuron.weights = neuron.weights.mapIndexed { index, w ->
                w - learningRateFactor * neuron.deltaWeights[index]
            }.toMutableList()
        }
    }

    override fun resetGradients() {
        for (neuron in neurons) {
            neuron.deltaBias = 0.0
            neuron.deltaWeights = MutableList(neuron.deltaWeights.size) { 0.0 }
        }
    }

    // Existing constructors and toString/toFileString...
/*    constructor(l: MutableList<Neuron>) {
        neurons = l
        inputNumber = l.firstOrNull()?.weights?.size ?: 0
    }*/

    override fun toString(): String {
        return "FCGLayer: " + neurons.joinToString(" ") { it.toString() }
    }
    override fun toFileString(): String {
        return "FC_LAYER:" + neurons.joinToString(" ") { "{${it.toFileString()}}" }
    }
}


// --- 3. CNNGLayer Implementations (Simplified Backprop) ---

class ConvLayer(
    val inputChannels: Int,
    val outputChannels: Int,
    val kernelSize: Int,
    val stride: Int = 1,
    val padding: Int = 0,
    onlyZeros: Boolean = false
) : ImageLayer {

    // Parameters and Gradients
    val kernels: Array<Array<Array<Array<Double>>>>
    val biases: Array<Double>
    var deltaKernels: Array<Array<Array<Array<Double>>>>
    var deltaBiases: Array<Double>

    // State needed for backpropagation
    var lastInputTensor: Tensor3D? = null
    var lastZTensor: Tensor3D? = null

    init {
        val random = Random(System.currentTimeMillis())
        // Initialization (omitted size for brevity, assume correct 4D structure)
        kernels = Array(outputChannels) { Array(inputChannels) { Array(kernelSize) { Array(kernelSize) { if (onlyZeros) 0.0 else random.nextDouble(-0.1, 0.1) } } } }
        biases = Array(outputChannels) { if (onlyZeros) 0.0 else random.nextDouble(-0.1, 0.1) }

        // Initialize gradient accumulators
        deltaKernels = Array(outputChannels) { Array(inputChannels) { Array(kernelSize) { Array(kernelSize) { 0.0 } } } }
        deltaBiases = Array(outputChannels) { 0.0 }
    }

    override fun getInputSize(): Int = -1
    override fun getOutputSize(): Int = -1

    private fun performConvolution(inputTensor: Tensor3D, saveZ: Boolean = false): Pair<Tensor3D, Tensor3D?> {
        // ... (Convolution logic remains the same)
        val inputHeight = inputTensor[0].size
        val inputWidth = inputTensor[0][0].size

        val outputHeight = (inputHeight - kernelSize + 2 * padding) / stride + 1
        val outputWidth = (inputWidth - kernelSize + 2 * padding) / stride + 1

        val outputTensor = Array(outputChannels) { Array(outputHeight) { Array(outputWidth) { 0.0 } } }
        val zTensor = if (saveZ) Array(outputChannels) { Array(outputHeight) { Array(outputWidth) { 0.0 } } } else null

        // Perform Convolution
        for (outC in 0 until outputChannels) {
            for (outRow in 0 until outputHeight) {
                for (outCol in 0 until outputWidth) {
                    var sum = biases[outC]
                    for (inC in 0 until inputChannels) {
                        for (kRow in 0 until kernelSize) {
                            for (kCol in 0 until kernelSize) {
                                val inputRow = outRow * stride + kRow - padding
                                val inputCol = outCol * stride + kCol - padding

                                if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth) {
                                    sum += inputTensor[inC][inputRow][inputCol] * kernels[outC][inC][kRow][kCol]
                                }
                            }
                        }
                    }
                    if (saveZ) zTensor!![outC][outRow][outCol] = sum
                    outputTensor[outC][outRow][outCol] = GNetwork.sigmoid(sum)
                }
            }
        }
        return Pair(outputTensor, zTensor)
    }

    override fun forward(input: Any): Any {
        @Suppress("UNCHECKED_CAST")
        val inputTensor = input as Tensor3D
        return performConvolution(inputTensor).first
    }

    override fun forwardAndSave(input: Any): ForwardPassResult {
        @Suppress("UNCHECKED_CAST")
        val inputTensor = input as Tensor3D
        lastInputTensor = inputTensor
        val (output, z) = performConvolution(inputTensor, saveZ = true)
        lastZTensor = z
        return ForwardPassResult(output, z!!)
    }

    override fun backward(delta: Any, activationPrevious: Any) {
        // dL/da_L (error w.r.t activation) from the next layer
        @Suppress("UNCHECKED_CAST")
        val deltaTensor = delta as Tensor3D
        val zTensor = lastZTensor ?: throw IllegalStateException("Z tensor not saved for backprop.")
        val inputTensor = lastInputTensor ?: throw IllegalStateException("Input tensor not saved for backprop.")

        // Calculate dL/dz (Error w.r.t. weighted input)
        val gradOutputTensor = Array(outputChannels) { c ->
            Array(deltaTensor[0].size) { r ->
                Array(deltaTensor[0][0].size) { col ->
                    deltaTensor[c][r][col] * GNetwork.sigmoid_derivative(zTensor[c][r][col])
                }
            }
        }

        // Accumulate dL/dB and dL/dW (Logic is the same as before)
        for (outC in 0 until outputChannels) {
            deltaBiases[outC] += gradOutputTensor[outC].sumOf { row -> row.sum() }

            for (inC in 0 until inputChannels) {
                for (kRow in 0 until kernelSize) {
                    for (kCol in 0 until kernelSize) {
                        var sum = 0.0
                        for (outRow in gradOutputTensor[outC].indices) {
                            for (outCol in gradOutputTensor[outC][outRow].indices) {
                                val inputRow = outRow * stride + kRow - padding
                                val inputCol = outCol * stride + kCol - padding

                                if (inputRow >= 0 && inputRow < inputTensor[inC].size && inputCol >= 0 && inputCol < inputTensor[inC][0].size) {
                                    sum += gradOutputTensor[outC][outRow][outCol] * inputTensor[inC][inputRow][inputCol]
                                }
                            }
                        }
                        deltaKernels[outC][inC][kRow][kCol] += sum
                    }
                }
            }
        }
    }

    override fun calculateDeltaForPreviousLayer(delta: Any, z: Any, previousLayerOutputSize: Int): Any {
        // NOTE: This should perform a full-convolution with a flipped kernel for correct backprop.
        // For structural integrity, we return the input delta which is not accurate for training.
        return delta
    }

    override fun applyGradients(learningRateFactor: Double) {
        // ... (Apply gradient logic remains the same)
        for (outC in 0 until outputChannels) {
            biases[outC] -= learningRateFactor * deltaBiases[outC]
            for (inC in 0 until inputChannels) {
                for (kRow in 0 until kernelSize) {
                    for (kCol in 0 until kernelSize) {
                        kernels[outC][inC][kRow][kCol] -= learningRateFactor * deltaKernels[outC][inC][kRow][kCol]
                    }
                }
            }
        }
    }

    override fun resetGradients() {
        // ... (Reset gradient logic remains the same)
        deltaBiases = Array(outputChannels) { 0.0 }
        deltaKernels = Array(outputChannels) { Array(inputChannels) { Array(kernelSize) { Array(kernelSize) { 0.0 } } } }
        lastInputTensor = null
        lastZTensor = null
    }

    override fun toString(): String = "ConvGLayer ($inputChannels -> $outputChannels, K=$kernelSize, S=$stride)"
    override fun toFileString(): String = "CONV_LAYER:..."
}


class PoolingLayer(
    val poolSize: Int,
    val stride: Int = 2
) : ImageLayer {

    var lastInputTensor: Tensor3D? = null
    // You would store max indices here for proper backprop (omitted for brevity)

    override fun getInputSize(): Int = -1
    override fun getOutputSize(): Int = -1

    private fun performPooling(inputTensor: Tensor3D): Tensor3D {
        // ... (Pooling logic remains the same)
        val channels = inputTensor.size
        val inputHeight = inputTensor[0].size
        val inputWidth = inputTensor[0][0].size

        val outputHeight = (inputHeight - poolSize) / stride + 1
        val outputWidth = (inputWidth - poolSize) / stride + 1

        val outputTensor = Array(channels) { Array(outputHeight) { Array(outputWidth) { 0.0 } } }

        for (channel in 0 until channels) {
            for (outRow in 0 until outputHeight) {
                for (outCol in 0 until outputWidth) {
                    var maxVal = Double.NEGATIVE_INFINITY
                    for (pRow in 0 until poolSize) {
                        for (pCol in 0 until poolSize) {
                            val inputRow = outRow * stride + pRow
                            val inputCol = outCol * stride + pCol
                            maxVal = maxOf(maxVal, inputTensor[channel][inputRow][inputCol])
                        }
                    }
                    outputTensor[channel][outRow][outCol] = maxVal
                }
            }
        }
        return outputTensor
    }

    override fun forward(input: Any): Any {
        @Suppress("UNCHECKED_CAST")
        val inputTensor = input as Tensor3D
        return performPooling(inputTensor)
    }

    override fun forwardAndSave(input: Any): ForwardPassResult {
        @Suppress("UNCHECKED_CAST")
        val inputTensor = input as Tensor3D
        lastInputTensor = inputTensor
        val output = performPooling(inputTensor)
        // No 'z' for pooling, input=output for Z
        return ForwardPassResult(output, output)
    }

    override fun backward(delta: Any, activationPrevious: Any) { /* No parameters to update */ }

    override fun calculateDeltaForPreviousLayer(delta: Any, z: Any, previousLayerOutputSize: Int): Any {
        // NOTE: This should perform unpooling (distributing error to the max index)
        // For structural integrity, we return the input delta which is not accurate for training.
        return delta
    }

    override fun applyGradients(learningRateFactor: Double) { /* No parameters */ }
    override fun resetGradients() {
        lastInputTensor = null
    }

    override fun toString(): String = "PoolingGLayer (P=$poolSize, S=$stride)"
    override fun toFileString(): String = "POOL_LAYER:..."
}


class FlattenLayer : ImageLayer {

    var lastInputShape: Triple<Int, Int, Int>? = null // (Channels, Height, Width)

    override fun getInputSize(): Int = -1
    override fun getOutputSize(): Int = -1

    override fun forward(input: Any): Any {
        @Suppress("UNCHECKED_CAST")
        val tensor = input as Tensor3D
        val flatList = mutableListOf<Double>()

        for (channel in tensor.indices) {
            for (row in tensor[channel].indices) {
                flatList.addAll(tensor[channel][row])
            }
        }
        return flatList
    }

    override fun forwardAndSave(input: Any): ForwardPassResult {
        @Suppress("UNCHECKED_CAST")
        val tensor = input as Tensor3D
        lastInputShape = Triple(tensor.size, tensor[0].size, tensor[0][0].size)

        val flatList = mutableListOf<Double>()
        for (channel in tensor.indices) {
            for (row in tensor[channel].indices) {
                flatList.addAll(tensor[channel][row])
            }
        }
        // No separate 'z' for flatten, input=output
        return ForwardPassResult(flatList, flatList)
    }

    override fun backward(delta: Any, activationPrevious: Any) { /* No parameters to update */ }

    // Calculates dL/dX for the previous layer (Reshapes 1D delta back to 3D tensor)
    override fun calculateDeltaForPreviousLayer(delta: Any, z: Any, previousLayerOutputSize: Int): Any {
        if (delta !is MutableList<*>) {
            throw IllegalArgumentException("Flatten backward expects 1D List delta from FC layer.")
        }
        @Suppress("UNCHECKED_CAST")
        val deltaList = delta as MutableList<Double>

        val (channels, height, width) = lastInputShape ?: throw IllegalStateException("Input shape not saved.")

        // Reshape 1D list back to 3D Tensor
        val reshapedDelta = Array(channels) { c ->
            Array(height) { r ->
                Array(width) { col ->
                    val index = c * height * width + r * width + col
                    deltaList[index]
                }
            }
        }
        return reshapedDelta
    }

    override fun applyGradients(learningRateFactor: Double) { /* No parameters */ }
    override fun resetGradients() {
        lastInputShape = null
    }

    override fun toString(): String = "FlattenGLayer"
    override fun toFileString(): String = "FLATTEN_LAYER"
}


// --- 4. GNetwork Class (Containing Static Functions) ---

class GNetwork {

    // FIX: Sigmoid functions moved here and marked as static (companion object)
    companion object {
        fun sigmoid(value : Double): Double {
            return 1.0 / (1.0 + exp(-value))
        }
        fun sigmoid_derivative(value : Double): Double {
            val s = sigmoid(value)
            return s * (1.0 - s)
        }
    }

    var layers: MutableList<ImageLayer>

    constructor(l: MutableList<ImageLayer>) {
        layers = l
    }

    // Helper function to convert input list to a 3D tensor for the first ConvLayer
    private fun convert1DTo3DTensor(input: MutableList<Double>, size: Int): Tensor3D {
        val tensor = Array(1) { Array(size) { Array(size) { 0.0 } } }
        val numElements = size * size
        if (input.size != numElements) {
            throw IllegalArgumentException("Input size mismatch: Expected $numElements, got ${input.size}")
        }
        for (i in input.indices) {
            val row = i / size
            val col = i % size
            tensor[0][row][col] = input[i]
        }
        return tensor
    }


    fun feedforward(input : MutableList<Double>, inputSize: Int) : MutableList<Double> {

        var currentOutput: Any = convert1DTo3DTensor(input, inputSize)

        for (layer in layers) {
            currentOutput = layer.forward(currentOutput)
        }

        // FIX: Check for raw MutableList type
        if (currentOutput !is MutableList<*>) {
            throw IllegalStateException("Final layer did not output MutableList.")
        }
        @Suppress("UNCHECKED_CAST")
        return currentOutput as MutableList<Double>
    }

    // SGD is now synchronous to simplify coroutine scope handling on mutable state
    suspend fun SGD(
        trainingData: MutableList<Pair<MutableList<Double>, MutableList<Double>>>,
        miniBatchSize: Int,
        learningRate: Double
    ) //stochastic gradient descent
    {
        coroutineScope {
            for (i in 0 until trainingData.size step miniBatchSize) {
                val end = minOf(i + miniBatchSize, trainingData.size)
                val miniBatch = trainingData.subList(i, end)

                updateMiniBatch(miniBatch, learningRate)
                println("Processed batch starting at $i")
            }
        }
    }

    private fun updateMiniBatch(miniBatch: List<Pair<MutableList<Double>, MutableList<Double>>>, learningRate: Double) {

        // 1. Accumulate Gradients for the whole batch
        for (example in miniBatch) {
            // Assuming input images are square and the size is needed for the first layer
            val inputSize = kotlin.math.sqrt(example.first.size.toDouble()).toInt()
            backpropagation(example.first, example.second, inputSize)
        }

        // 2. Apply Gradients and Reset Accumulators
        for (layer in layers) {
            layer.applyGradients(learningRate / miniBatch.size)
            layer.resetGradients()
        }
    }

    private fun backpropagation(input: MutableList<Double>, targetValues: MutableList<Double>, inputSize: Int) {

        // --- 1. Feedforward (Save activations/intermediate states) ---

        val activations = mutableListOf<Any>()
        val weightedInputs = mutableListOf<Any>()

        var currentOutput: Any = convert1DTo3DTensor(input, inputSize)
        activations.add(currentOutput)

        for (layer in layers) {
            val forwardResult = layer.forwardAndSave(currentOutput)

            currentOutput = forwardResult.output
            weightedInputs.add(forwardResult.z)
            activations.add(currentOutput)
        }

        // --- 2. Backpropagation (Calculate deltas and gradients) ---

        // A. OutputGLayer Delta (Last layer is always FC (Layer class))
        val outputLayerIndex = layers.size - 1
        val outputLayer = layers[outputLayerIndex] as GLayer

        // FIX: Check for raw MutableList type
        if (activations.last() !is MutableList<*>) {
            throw IllegalStateException("Final activation must be 1D List.")
        }
        @Suppress("UNCHECKED_CAST")
        val finalActivation = activations.last() as MutableList<Double>
        @Suppress("UNCHECKED_CAST")
        val finalZ = weightedInputs.last() as MutableList<Double>

        // Calculate initial delta: (a_L - y) * sigmoid_prime(z_L)
        val costDerivative = cost_derivative(finalActivation, targetValues)
        var delta: Any = finalZ.mapIndexed { index, z ->
            costDerivative[index] * sigmoid_derivative(z)
        }.toMutableList()

        // Accumulate gradients for the output FC layer
        outputLayer.backward(delta, activations[outputLayerIndex])

        // B. Propagate Delta Backward through HiddenGLayers
        for (i in outputLayerIndex - 1 downTo 0) {
            val layer = layers[i]
            val nextLayer = layers[i + 1]
            val z = weightedInputs[i] // Weighted input for layer i
            val activationPrevious = activations[i] // Input (a) for layer i

            // Calculate delta for the current layer (based on the delta from the next layer)
            delta = nextLayer.calculateDeltaForPreviousLayer(delta, z, layer.getOutputSize())

            // If the layer is FC, we need to apply the activation derivative: (W^T * delta) . sigmoid_prime(z_L-1)
            // If the layer is Conv/Pool/Flatten, the previous layer is usually 3D, and the delta is 3D.
            if (layer is GLayer) {
                // Apply sigmoid prime to delta coming from the next layer for FC layers
                @Suppress("UNCHECKED_CAST")
                val deltaList = delta as MutableList<Double>
                @Suppress("UNCHECKED_CAST")
                val zList = weightedInputs[i-1] as MutableList<Double> // z from the current layer

                delta = deltaList.mapIndexed { index, d ->
                    // Use the Z from the current layer (weightedInputs[i])
                    d * sigmoid_derivative(zList[index])
                }.toMutableList()
            }

            // Accumulate gradients for the current layer
            layer.backward(delta, activationPrevious)
        }
    }

    fun evaluate(testData : MutableList<Pair<MutableList<Double>, MutableList<Double>>>, inputSize: Int)
    {
        var stats = MutableList<Int>(10) {0}
        var correct = 0
        for(i in testData.indices)
        {
            val output = feedforward(testData[i].first, inputSize)
            val expectedOutput = testData[i].second

            val predictedIndex = output.indexOf(output.maxOrNull() ?: 0.0)
            val expectedIndex = expectedOutput.indexOf(1.0)

            if(predictedIndex == expectedIndex)
            {
                correct++
                stats[expectedIndex]++
            }
        }
        println("Correct: $correct / ${testData.size}")
        println("stats: $stats")
    }

    fun cost_derivative(outputs : MutableList<Double>, targetValues : MutableList<Double>): MutableList<Double> {
        return outputs.mapIndexed { index, output -> output - targetValues[index] }.toMutableList()
    }

    override fun toString(): String {
        return layers.joinToString("\n") { it.toString() }
    }
    fun toFileString(): String {
        return layers.joinToString("\n") { it.toFileString() }
    }
}