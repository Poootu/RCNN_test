package org.example

import java.io.File
import kotlin.random.Random
import com.sksamuel.scrimage.ImmutableImage
import com.sksamuel.scrimage.color.RGBColor
import com.sksamuel.scrimage.filter.GrayscaleFilter
import com.sksamuel.scrimage.nio.PngWriter
import org.example.ConvolutionalNetwork.Tensor


// ---------------------------------------------------------
// MAIN FUNCTION – Generates 1000 background tensors
// ---------------------------------------------------------
fun generateBackgroundTensorsFromFolder(
    sourceFolder: String,
    outputCount: Int = 100,
    cropSize: Int = 32,
    saveDebugPng: Boolean = false,
    debugFolder: String = "src/main/resources/backgroundPhotosDebug"
): List<Tensor> {

    val imgs = File(sourceFolder)
        .walkTopDown()
        .filter { it.isFile && (it.extension.lowercase() in listOf("jpg", "jpeg", "png")) }
        .map { ImmutableImage.loader().fromFile(it).scale(.3) }
        .toList()

    require(imgs.isNotEmpty()) {
        "No images found in folder: $sourceFolder"
    }
    val result = mutableListOf<Tensor>()

    if (saveDebugPng) {
        File(debugFolder).mkdirs()
    }

    repeat(outputCount) { idx ->
        val img = imgs.random()

        val crop = randomGrayscaleCrop(img, cropSize)

        val tensor = imageToTensor(crop)

        if (saveDebugPng) {
            val outFile = File("$debugFolder/bg_${idx}.png")
            crop.output(PngWriter.NoCompression, outFile)
        }

        result.add(tensor)
    }

    return result
}



// ---------------------------------------------------------
// Take a random 32×32 region from a larger image
// ---------------------------------------------------------
fun randomGrayscaleCrop(
    img: ImmutableImage,
    cropSize: Int
): ImmutableImage {

    val w = img.width
    val h = img.height

    if (w < cropSize || h < cropSize) {
        // resize up (rare)
        val resized = img.scaleTo(cropSize * 2, cropSize * 2)
        return randomGrayscaleCrop(resized, cropSize)
    }

    val x = Random.nextInt(0, w - cropSize)
    val y = Random.nextInt(0, h - cropSize)

    val crop = img.subimage(x, y, cropSize, cropSize)

    // convert to grayscale
    return crop.filter { GrayscaleFilter() }
/*    return crop.map { px ->
        val gray = ((px.red() + px.green() + px.blue()) / 3).coerceIn(0, 255)
        RGBColor(gray, gray, gray).toPixel()
    }*/
}



// ---------------------------------------------------------
// Convert a Scrimage image to Tensor(1, 32, 32) normalized 0..1
// ---------------------------------------------------------
fun imageToTensor(img: ImmutableImage): Tensor {
    val w = img.width
    val h = img.height
    val arr = DoubleArray(w * h)

    for (y in 0 until h) {
        for (x in 0 until w) {
            val px = img.pixel(x, y)
            arr[y * w + x] = (px.red() / 255.0)
        }
    }

    return Tensor(arr, h, w, 1)
}
