package org.example.ConvolutionalNetwork

import kotlin.random.Random

fun trainingTest()
{
    val REGION_H = 32
    val REGION_W = 32
    val REGION_C = 3
    val NUM_CLASSES = 10
    val EPOCHS = 10 // Start low, typically hundreds are needed

    // 1. Initialize the CNN (using the function provided previously)
    val cnn = createCnn(
        inputW = REGION_W,
        inputH = REGION_H,
        inputC = REGION_C,
        numClasses = NUM_CLASSES
    )

    // 2. Load the prepared and labeled data
    val data = exampleTrainingSetup() // Replace with your actual data loading logic

    println("Starting training with ${data.size} samples...")

    // 3. Start Training
    trainCnn(
        network = cnn,
        trainingData = data,
        regionH = REGION_H,
        regionW = REGION_W,
        regionC = REGION_C,
        numClasses = NUM_CLASSES,
        epochs = EPOCHS
    )
}

fun trainCnn(
    network: Network,
    trainingData: List<Pair<DoubleArray, DoubleArray>>,
    regionH: Int,
    regionW: Int,
    regionC: Int,
    numClasses: Int,
    epochs: Int
) {
    val numSamples = trainingData.size

    // Simple loss tracker (for monitoring progress)
    fun calculateLoss(probs: Tensor, target: Tensor): Double {
        var loss = 0.0
        for (i in target.data.indices) {
            // Using a safe log implementation to avoid issues with zero probability
            val p = probs.data[i].coerceAtLeast(1e-12)
            if (target.data[i] == 1.0) {
                loss -= kotlin.math.ln(p)
            }
        }
        return loss
    }

    for (epoch in 1..epochs) {
        var totalLoss = 0.0

        // Shuffle the data for Stochastic Gradient Descent (SGD)
        val shuffledData = trainingData.shuffled()

        for ((regionData, targetLabel) in shuffledData) {
            // 1. Prepare Input and Target Tensors
            val inputTensor = Tensor(regionData, regionH, regionW, regionC)
            val targetTensor = Tensor(targetLabel, 1, 1, numClasses)

            // 2. Perform Training Step (Forward, Loss, Backward, Weight Update)
            network.train(inputTensor, targetTensor)

            // 3. Track Loss (Forward pass again to get final probabilities for loss calculation)
            val output = network.forward(inputTensor)
            val probs = softmax(output) // Note: Softmax is defined in the Network class scope
            totalLoss += calculateLoss(probs, targetTensor)
        }

        val avgLoss = totalLoss / numSamples
        println("Epoch $epoch/$epochs, Average Loss: $avgLoss")
    }
}

// Re-implement the softmax function outside of Network for loss calculation, or access it from Network.
// Given your current class structure, we'll assume it's available or re-implemented here:
fun softmax(logits: Tensor): Tensor {
    val input = logits.data
    val max = input.max()
    val exp = DoubleArray(input.size) { i ->
        kotlin.math.exp(input[i] - max)
    }
    val sum = exp.sum()
    val out = DoubleArray(input.size) { i -> exp[i] / sum }

    return Tensor(out, 1, 1, input.size)
}

fun exampleTrainingSetup(): List<Pair<DoubleArray, DoubleArray>> {
    // These constants must match your CNN architecture
    val REGION_SIZE = 32 * 32 * 3 // 32x32 RGB
    val NUM_CLASSES = 14          // Background, 0-9, +, -, =

    // --- Create a list of training examples ---
    val trainingData = mutableListOf<Pair<DoubleArray, DoubleArray>>()

    // Example 1: Region containing the digit '5' (Class Index 6)
    val region5Pixels = DoubleArray(REGION_SIZE) { Random.nextDouble(0.0, 1.0) } // Placeholder data
    val target5 = DoubleArray(NUM_CLASSES).apply { this[6] = 1.0 } // One-hot encoding for class 6
    trainingData.add(Pair(region5Pixels, target5))

    // Example 2: Region containing "Background" (Class Index 0)
    val regionBackgroundPixels = DoubleArray(REGION_SIZE) { Random.nextDouble(0.0, 1.0) } // Placeholder data
    val targetBackground = DoubleArray(NUM_CLASSES).apply { this[0] = 1.0 } // One-hot encoding for class 0
    trainingData.add(Pair(regionBackgroundPixels, targetBackground))

    // ... add hundreds or thousands more labeled regions ...

    return trainingData
}





fun testNetwork(region : DoubleArray)
{
    val inputSize = 32
    val channels = 1
    val classes = 10

    val cnn = createCnn(inputW = inputSize, inputH = inputSize, inputC = channels, numClasses = classes)

    println("CNN created with ${cnn.layers.size} layers.")

    // Example: Create a dummy input region (32x32x3)
    val inputData = DoubleArray(inputSize * inputSize * channels) { Random.nextDouble() }
    val dummyInput = Tensor(inputData, inputSize, inputSize, channels)

    // Example: Dummy target (one-hot encoding for class 5)
    val targetData = DoubleArray(classes) { if (it == 5) 1.0 else 0.0 }
    val dummyTarget = Tensor(targetData, 1, 1, classes)

    // Training a single step
    cnn.train(dummyInput, dummyTarget)
    println("Single training step completed (forward and backward pass).")
}
