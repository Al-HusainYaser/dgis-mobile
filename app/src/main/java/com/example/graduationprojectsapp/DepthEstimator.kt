package com.example.graduationprojectsapp

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import androidx.core.graphics.get
import java.nio.ByteBuffer

/**
 * Synchronous depth estimator using the Depth Anything V2 ONNX model.
 * Adapted from the original coroutine-based DepthAnything class for direct
 * use from background worker threads in the camera pipeline.
 *
 * Returns both the raw ALPHA_8 depth bitmap (for hazard analysis)
 * and a colorized ARGB_8888 bitmap (for display).
 */
class DepthEstimator(context: Context) {

    private val modelName = "fused_model_uint8_256.onnx"

    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession =
        ortEnvironment.createSession(context.assets.open(modelName).readBytes())
    private val inputName = ortSession.inputNames.iterator().next()

    // Model-specific dimensions for the 256-variant
    private val inputDim = 256
    private val outputDim = 252

    // The model output is rotated 90° relative to the input — rotate back
    private val rotateTransform = Matrix().apply { postRotate(90f) }

    /**
     * Run depth estimation on the given bitmap.
     * This is a blocking call — must be called from a background thread.
     *
     * @param inputBitmap Camera frame (any size; will be internally scaled to 256×256)
     * @return Pair of:
     *   - rawDepth: ALPHA_8 bitmap where alpha value = depth (255 = close, 0 = far)
     *   - colorizedDepth: ARGB_8888 Inferno-colormap bitmap for display
     */
    fun estimate(inputBitmap: Bitmap): Pair<Bitmap, Bitmap> {
        // Scale input to model's required size
        val resizedImage = Bitmap.createScaledBitmap(inputBitmap, inputDim, inputDim, true)
        val imagePixels = convertToByteBuffer(resizedImage)

        // Build ONNX input tensor: shape [1, H, W, 3], UINT8
        val inputTensor = OnnxTensor.createTensor(
            ortEnvironment,
            imagePixels,
            longArrayOf(1, inputDim.toLong(), inputDim.toLong(), 3),
            OnnxJavaType.UINT8
        )

        // Run inference
        val outputs = ortSession.run(mapOf(inputName to inputTensor))
        val outputTensor = outputs[0] as OnnxTensor

        // Decode raw depth output into an ALPHA_8 bitmap
        var rawDepth = Bitmap.createBitmap(outputDim, outputDim, Bitmap.Config.ALPHA_8)
        rawDepth.copyPixelsFromBuffer(outputTensor.byteBuffer)

        // Correct orientation (model output is transposed 90°)
        rawDepth = Bitmap.createBitmap(rawDepth, 0, 0, outputDim, outputDim, rotateTransform, false)

        // Scale depth map back to input image dimensions
        rawDepth = Bitmap.createScaledBitmap(rawDepth, inputBitmap.width, inputBitmap.height, true)

        // Apply Inferno colormap for visual display
        val colorizedDepth = colormapInferno(rawDepth)

        // Clean up tensors
        inputTensor.close()
        outputs.close()

        return Pair(rawDepth, colorizedDepth)
    }

    /**
     * Converts a Bitmap to a ByteBuffer suitable for ONNX input.
     * Pixel format expected by the model: [R, B, G] (note: blue and green swapped).
     */
    private fun convertToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val imgData = ByteBuffer.allocate(1 * bitmap.width * bitmap.height * 3)
        imgData.rewind()
        for (i in 0 until bitmap.width) {
            for (j in 0 until bitmap.height) {
                val pixel = bitmap[i, j]
                imgData.put(Color.red(pixel).toByte())
                imgData.put(Color.blue(pixel).toByte())
                imgData.put(Color.green(pixel).toByte())
            }
        }
        imgData.rewind()
        return imgData
    }

    fun close() {
        ortSession.close()
        ortEnvironment.close()
    }
}
