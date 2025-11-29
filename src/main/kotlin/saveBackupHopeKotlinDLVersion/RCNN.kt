package org.example.saveBackupHopeKotlinDLVersion

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset

import java.io.File

/**
 * Training Script for the R-CNN Classifier.
 * This generates the "math_solver_model" used by RCNNDigitRecognizer.
 */
object Train {
    private const val MODEL_DIR = "math_solver_model"

    @JvmStatic
    fun main(args: Array<String>) {
        // 1. Define the CNN Architecture
        // FIX: Used intArrayOf instead of longArrayOf for kernelSize and strides
        val model = Sequential.of(
            Input(28, 28, 1),

            Conv2D(
                filters = 32,
                kernelSize = intArrayOf(3, 3), // Fixed: IntArray
                strides = intArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = HeNormal(),
                biasInitializer = Zeros(),
                padding = ConvPadding.SAME
            ),

            MaxPool2D(
                poolSize = intArrayOf(1, 2, 2, 1), // Fixed: IntArray
                strides = intArrayOf(1, 2, 2, 1),
                padding = ConvPadding.VALID
            ),

            Conv2D(
                filters = 64,
                kernelSize = intArrayOf(3, 3), // Fixed: IntArray
                strides = intArrayOf(1, 1, 1, 1),
                activation = Activations.Relu,
                kernelInitializer = HeNormal(),
                biasInitializer = Zeros(),
                padding = ConvPadding.SAME
            ),

            MaxPool2D(
                poolSize = intArrayOf(1, 2, 2, 1), // Fixed: IntArray
                strides = intArrayOf(1, 2, 2, 1),
                padding = ConvPadding.VALID
            ),

            Flatten(),

            Dense(
                outputSize = 128,
                activation = Activations.Relu,
                kernelInitializer = HeNormal(),
                biasInitializer = Zeros()
            ),

            // 14 Classes: 0-9 (10), +, -, = (3), background/noise (1)
            Dense(
                outputSize = 14,
                activation = Activations.Softmax,
                kernelInitializer = HeNormal(),
                biasInitializer = Zeros()
            )
        )

        // 2. Load Your Data
        // NOTE: You must provide your own FloatArray data here.
        // Images should be flattened (28*28 = 784 floats), normalized 0.0-1.0
        val (trainData, testData) = generateDummyData(100) // Replaced with dummy data for compilation

        // 3. Compile & Train
        model.use {
            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            println("Starting training...")
            it.fit(
                dataset = trainData,
                epochs = 5,
                batchSize = 32
            )

            val accuracy = it.evaluate(dataset = testData, batchSize = 32).metrics[Metrics.ACCURACY]
            println("Test Accuracy: $accuracy")

            // 4. Save the Model
            // This creates the directory structure compatible with SavedModelInferenceModel.load()
            val modelFile = File(MODEL_DIR)
            it.save(modelFile, writingMode = WritingMode.OVERRIDE)
            println("Model saved to: ${modelFile.absolutePath}")
        }
    }

    /**
     * Helper to generate dummy data so this file compiles and runs.
     * Replace this with real logic to load images from disk.
     */
    private fun generateDummyData(size: Int): Pair<OnHeapDataset, OnHeapDataset> {
        val x = Array(size) { FloatArray(28 * 28 * 1) { Math.random().toFloat() } }
        val y = FloatArray(size) { (Math.random() * 14).toInt().toFloat() } // 14 classes
        val dataset = OnHeapDataset.create(x, y)
        return dataset.split(0.8)
    }
}