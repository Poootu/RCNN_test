package org.example.ConvolutionalNetwork

import kotlinx.coroutines.coroutineScope
import kotlin.math.exp
import kotlin.math.max
import kotlin.random.Random


// NOTE: In a real application, learningRate should not be a global constant.
val learningRate = 0.01 // Reduced learning rate for stability

data class Tensor(
    val data: DoubleArray,
    val h: Int,
    val w: Int,
    val c: Int
) {
    fun copy() = Tensor(data.copyOf(), h, w, c)
}




class Network(val layers: List<Layer>) {

    fun forward(input: Tensor, applySoftmax: Boolean = false): Tensor { // MODIFIED: Added applySoftmax parameter
        var x = input
        for (layer in layers) {
            x = layer.forward(x)
        }
        // Apply softmax if requested (for inference/prediction)
        return if (applySoftmax) softmax(x) else x
    }

    fun backward(finalDelta: Tensor, inputs: List<Tensor>) {
        var delta = finalDelta
        // Iterate backward from the final layer (size - 1) down to the second layer (index 1),
        // as layers[i] needs inputs[i] which is the output of the layer before it (layers[i-1]).
        // Note: The inputs list contains [input, layer1_output, layer2_output, ...]
        for (i in layers.size - 1 downTo 0) {
            // inputs[i] is the input to layer 'i' (which is the output of layer 'i-1')
            delta = layers[i].backward(delta, inputs[i])
        }
    }

    fun train(input: Tensor, target: Tensor) {
        val inputs = ArrayList<Tensor>()
        inputs.add(input) // input[0] is the network's input

        var x = input
        for (layer in layers) {
            x = layer.forward(x)
            inputs.add(x) // inputs[i+1] is the output of layer 'i'
        }

        val probs = softmax(inputs.last())
        val grad = crossEntropyDerivative(probs, target)

        // inputs list contains: [Initial Input, Output_L1, Output_L2, ..., Output_LN-1, Output_LN (pre-softmax logits)]
        // The backward pass requires the input to the layer being backpropagated.
        // E.g., layers[size-1] needs inputs[size-1] (output of layer N-2 or pre-activation of N-1)
        backward(grad, inputs)
    }

    // -----------------------------------------------------
    // Softmax
    // -----------------------------------------------------
    private fun softmax(logits: Tensor): Tensor {
        val input = logits.data
        val max = input.maxOrNull() ?: 0.0 // Handle empty case, although unlikely
        val exp = DoubleArray(input.size) { i ->
            kotlin.math.exp(input[i] - max)
        }
        val sum = exp.sum()
        val out = DoubleArray(input.size) { i -> exp[i] / sum }

        return Tensor(out, 1, 1, input.size)
    }

    // -----------------------------------------------------
    // Cross-entropy derivative: (pred - target)
    // -----------------------------------------------------
    private fun crossEntropyDerivative(pred: Tensor, target: Tensor): Tensor {
        val data = pred.data
        val t = target.data
        val out = DoubleArray(data.size)

        for (i in data.indices) {
            out[i] = data[i] - t[i]   // Gradient of Cross-Entropy w.r.t. logits
        }

        return Tensor(out, 1, 1, data.size)
    }
}

interface Layer {
    fun forward(input: Tensor): Tensor
    fun backward(delta: Tensor, input: Tensor): Tensor
}


class DenseLayer(
    val inputSize: Int,
    val outputSize: Int,
    var weights: Array<DoubleArray>,   // [output][input]
    var biases: DoubleArray            // [output]
) : Layer {

    private lateinit var lastInput: Tensor // Stored for use in backward pass
    private lateinit var lastZ: Tensor // Stored for use in backward pass (though not strictly needed here)

    override fun forward(input: Tensor): Tensor {
        lastInput = input // Input is flattened, 1x1xinputSize

        val output = DoubleArray(outputSize)
        val x = input.data

        for (i in 0 until outputSize) {
            var sum = biases[i]
            val w = weights[i]
            for (j in 0 until inputSize) {
                sum += w[j] * x[j]
            }
            output[i] = sum
        }

        lastZ = Tensor(output, 1, 1, outputSize)
        return lastZ
    }

    override fun backward(delta: Tensor, input: Tensor): Tensor {
        val d = delta.data
        val x = lastInput.data

        // Gradients for weights + bias
        val gradW = Array(outputSize) { DoubleArray(inputSize) }
        val gradB = DoubleArray(outputSize)
        val gradInput = DoubleArray(inputSize) // Delta to be passed to previous layer

        for (i in 0 until outputSize) { // output dimension
            // Bias gradient is just the upstream delta
            gradB[i] = d[i]

            for (j in 0 until inputSize) { // input dimension
                // Weight gradient: delta * input_value
                gradW[i][j] = d[i] * x[j]

                // Input gradient (delta to previous layer): sum of (delta * weight)
                // This accumulates the error contribution from all output neurons 'i'
                // that connect to the input neuron 'j'.
                gradInput[j] += d[i] * weights[i][j]
            }
        }

        // Update weights (SGD)
        for (i in 0 until outputSize) {
            for (j in 0 until inputSize) {
                weights[i][j] -= learningRate * gradW[i][j]
            }
            biases[i] -= learningRate * gradB[i]
        }

        return Tensor(gradInput, 1, 1, inputSize) // This delta is still flattened
    }
}

class ConvLayer(
    val kernelSize: Int,
    val inChannels: Int,
    val outChannels: Int,
    val weights: Array<Array<Array<DoubleArray>>>, // [out][in][kh][kw]
    val biases: DoubleArray,
    val stride: Int = 1,
    val padding: Int = 0
) : Layer {

    private lateinit var lastInput: Tensor // Store input for backprop

    override fun forward(input: Tensor): Tensor {
        lastInput = input // Store input for use in backward pass

        // Calculate output dimensions
        val outH = (input.h + 2*padding - kernelSize) / stride + 1
        val outW = (input.w + 2*padding - kernelSize) / stride + 1

        val output = Tensor(DoubleArray(outH * outW * outChannels),outH, outW, outChannels )

        for (oc in 0 until outChannels) {
            for (oh in 0 until outH) {
                for (ow in 0 until outW) {

                    var sum = biases[oc]

                    for (ic in 0 until inChannels) {
                        for (kh in 0 until kernelSize) {
                            for (kw in 0 until kernelSize) {

                                // Calculate input position (ih, iw) from output position (oh, ow)
                                val ih = oh * stride + kh - padding
                                val iw = ow * stride + kw - padding

                                // Check for boundary conditions (handling padding virtually)
                                if (ih in 0 until input.h && iw in 0 until input.w) {
                                    val inputIndex = ic * input.h * input.w + ih * input.w + iw
                                    sum += input.data[inputIndex] * weights[oc][ic][kh][kw]
                                }
                                // If outside bounds, the input is zero due to padding, so we skip
                            }
                        }
                    }

                    val outIndex = oc * outH * outW + oh * outW + ow
                    output.data[outIndex] = sum
                }
            }
        }

        return output
    }

    override fun backward(delta: Tensor, input: Tensor): Tensor {
        val inH = input.h
        val inW = input.w
        val outH = delta.h
        val outW = delta.w

        // Use the stored input from the forward pass
        val prevInput = lastInput

        // 1. Gradients for Weights and Biases (dW, dB)
        val gradW = Array(outChannels) { Array(inChannels) { Array(kernelSize) { DoubleArray(kernelSize) } } }
        val gradB = DoubleArray(outChannels)

        for (oc in 0 until outChannels) {
            // Gradient w.r.t. Bias is sum of all deltas in that channel
            for (i in 0 until outH * outW) {
                gradB[oc] += delta.data[oc * outH * outW + i]
            }

            // Gradient w.r.t. Weights (Convolution of input with delta)
            for (ic in 0 until inChannels) {
                for (kh in 0 until kernelSize) {
                    for (kw in 0 until kernelSize) {
                        for (oh in 0 until outH) {
                            for (ow in 0 until outW) {
                                val ih = oh * stride + kh - padding
                                val iw = ow * stride + kw - padding

                                if (ih in 0 until inH && iw in 0 until inW) {
                                    val inputIndex = ic * inH * inW + ih * inW + iw
                                    val deltaIndex = oc * outH * outW + oh * outW + ow
                                    gradW[oc][ic][kh][kw] += delta.data[deltaIndex] * prevInput.data[inputIndex]
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Gradient w.r.t. Input (Propagate to previous layer) - CORRECTED LOGIC
        // This is equivalent to convolving the zero-padded delta with the spatially-flipped kernel.
        val gradInput = DoubleArray(inH * inW * inChannels)
        val newDelta = Tensor(gradInput, inH, inW, inChannels) // This is the delta passed back

        // Iterate over the output delta and "splat" the error back to the input
        for (oc in 0 until outChannels) {
            for (ic in 0 until inChannels) {
                for (oh in 0 until outH) {
                    for (ow in 0 until outW) {
                        val deltaIndex = oc * outH * outW + oh * outW + ow
                        val deltaVal = delta.data[deltaIndex]

                        for (kh in 0 until kernelSize) {
                            for (kw in 0 until kernelSize) {
                                // Calculate the corresponding input position (ih, iw)
                                // This index is where the error from (oh, ow) through kernel (kh, kw) originates
                                val ih = oh * stride + kh - padding
                                val iw = ow * stride + kw - padding

                                if (ih in 0 until inH && iw in 0 until inW) {
                                    val inputIndex = ic * inH * inW + ih * inW + iw
                                    // The backpropagated error is weighted by the original kernel weight
                                    // The error accumulates from all relevant output channels (oc)
                                    newDelta.data[inputIndex] += deltaVal * weights[oc][ic][kh][kw]
                                }
                            }
                        }
                    }
                }
            }
        }


        // 3. Update Weights (SGD)
        for (oc in 0 until outChannels) {
            for (ic in 0 until inChannels) {
                for (kh in 0 until kernelSize) {
                    for (kw in 0 until kernelSize) {
                        weights[oc][ic][kh][kw] -= learningRate * gradW[oc][ic][kh][kw]
                    }
                }
            }
            biases[oc] -= learningRate * gradB[oc]
        }
        return newDelta
    }
}

class ReLULayer : Layer {

    private lateinit var lastInput: Tensor // Stored for the derivative check

    override fun forward(input: Tensor): Tensor {
        lastInput = input
        val out = input.copy()
        for (i in out.data.indices)
            out.data[i] = max(0.0, out.data[i])
        return out
    }

    override fun backward(delta: Tensor, input: Tensor): Tensor {
        val grad = delta.copy()
        // Derivative of ReLU is 1 if input > 0, and 0 otherwise
        for (i in grad.data.indices)
            grad.data[i] = if (lastInput.data[i] > 0) delta.data[i] else 0.0
        return grad
    }
}

class FlattenLayer : Layer {
    private lateinit var lastShape: Tensor // Store the 3D shape for unflattening in backward

    override fun forward(input: Tensor): Tensor {
        lastShape = input
        return Tensor(
            data = input.data.copyOf(),
            h = 1, w = 1, c = input.data.size
        )
    }

    override fun backward(delta: Tensor, input: Tensor): Tensor {
        // Unflatten the delta back to the original 3D shape
        return Tensor(
            data = delta.data.copyOf(),
            h = lastShape.h,
            w = lastShape.w,
            c = lastShape.c
        )
    }
}

class MaxPoolLayer(val poolSize: Int = 2) : Layer {

    private lateinit var lastInput: Tensor
    private lateinit var maxIndices: IntArray  // store indices of the maxima in the original input tensor

    override fun forward(input: Tensor): Tensor {
        lastInput = input

        val outH = input.h / poolSize
        val outW = input.w / poolSize
        val out = DoubleArray(outH * outW * input.c)
        maxIndices = IntArray(out.size)

        var outIndex = 0

        for (c in 0 until input.c) {
            for (y in 0 until outH) {
                for (x in 0 until outW) {

                    var maxVal = Double.NEGATIVE_INFINITY
                    var maxIdx = 0

                    for (py in 0 until poolSize) {
                        for (px in 0 until poolSize) {
                            val iy = y * poolSize + py
                            val ix = x * poolSize + px
                            // Calculate index in the flattened input array
                            val idx = c * input.h * input.w + iy * input.w + ix
                            val v = input.data[idx]

                            if (v > maxVal) {
                                maxVal = v
                                maxIdx = idx
                            }
                        }
                    }

                    out[outIndex] = maxVal
                    maxIndices[outIndex] = maxIdx // Store the index of the winner
                    outIndex++
                }
            }
        }

        return Tensor(out, outH, outW, input.c)
    }

    override fun backward(delta: Tensor, input: Tensor): Tensor {
        val grad = DoubleArray(input.data.size)

        // The gradient is only passed through the winning (max) element
        for (i in delta.data.indices) {
            grad[maxIndices[i]] = delta.data[i]
        }

        return Tensor(grad, input.h, input.w, input.c)
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


fun createCnn(
    inputW: Int,
    inputH: Int,
    inputC: Int,
    numClasses: Int
): Network {

    // Helper to initialize weights for a ConvLayer (using Xavier/Glorot would be better)
    fun initConvWeights(oc: Int, ic: Int, k: Int) = Array(oc) {
        Array(ic) {
            Array(k) { DoubleArray(k) { Random.nextDouble(-0.1, 0.1) } }
        }
    }

    // Helper to initialize biases
    fun initBiases(size: Int) = DoubleArray(size) { Random.nextDouble(-0.1, 0.1) }

    // --- First Conv Block: 32x32x3 -> 30x30x16 -> 15x15x16 ---
    val conv1OutC = 16
    val conv1Kernel = 3
    val conv1Stride = 1
    val conv1Padding = 0
    val pool1Size = 2

    val conv1 = ConvLayer(
        kernelSize = conv1Kernel,
        inChannels = inputC,
        outChannels = conv1OutC,
        weights = initConvWeights(conv1OutC, inputC, conv1Kernel),
        biases = initBiases(conv1OutC),
        stride = conv1Stride,
        padding = conv1Padding
    )

    // Dimensions after Conv1: H' = (32 + 2*0 - 3)/1 + 1 = 30. W' = 30. C' = 16. (30x30x16)
    // Dimensions after MaxPool1: H' = 30/2 = 15. W' = 30/2 = 15. C' = 16. (15x15x16)

    // --- Second Conv Block: 15x15x16 -> 13x13x32 -> 6x6x32 (rounding down) ---
    val conv2OutC = 32
    val conv2Kernel = 3
    val conv2Stride = 1
    val conv2Padding = 0
    val pool2Size = 2

    val conv2 = ConvLayer(
        kernelSize = conv2Kernel,
        inChannels = conv1OutC,
        outChannels = conv2OutC,
        weights = initConvWeights(conv2OutC, conv1OutC, conv2Kernel),
        biases = initBiases(conv2OutC),
        stride = conv2Stride,
        padding = conv2Padding
    )

    // Dimensions after Conv2: H' = (15 + 2*0 - 3)/1 + 1 = 13. W' = 13. C' = 32. (13x13x32)
    // Dimensions after MaxPool2: H' = 13/2 = 6 (integer division). W' = 6. C' = 32. (6x6x32)

    // --- Fully Connected/Dense Block ---
    // Input size for Dense layer: 6 * 6 * 32 = 1152
    val denseInSize = 6 * 6 * conv2OutC
    val dense1OutSize = 128

    // Helper to initialize weights for a DenseLayer
    fun initDenseWeights(out: Int, `in`: Int) = Array(out) { DoubleArray(`in`) { Random.nextDouble(-0.1, 0.1) } }

    val dense1 = DenseLayer(
        inputSize = denseInSize,
        outputSize = dense1OutSize,
        weights = initDenseWeights(dense1OutSize, denseInSize),
        biases = initBiases(dense1OutSize)
    )

    val outputLayer = DenseLayer(
        inputSize = dense1OutSize,
        outputSize = numClasses, // Output logits for Softmax
        weights = initDenseWeights(numClasses, dense1OutSize),
        biases = initBiases(numClasses)
    )

    return Network(
        layers = listOf(
            conv1, ReLULayer(), MaxPoolLayer(pool1Size),
            conv2, ReLULayer(), MaxPoolLayer(pool2Size),
            FlattenLayer(),
            dense1, ReLULayer(),
            outputLayer
        )
    )
}