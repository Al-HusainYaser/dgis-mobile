package com.example.graduationprojectsapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

data class DetectionResult(val box: RectF, val confidence: Float, val className: String)

class YOLOv11Detector(
    private val context: Context,
    modelName: String,
    private val labels: List<String>
) {
    private var interpreter: Interpreter? = null
    var inputImageWidth = 0
    var inputImageHeight = 0
    private var inputDataType: DataType = DataType.FLOAT32
    private var outputDataType: DataType = DataType.FLOAT32
    private var outputShape: IntArray = intArrayOf()

    init {
        try {
            val modelFile = loadModelFile(modelName)
            val options = Interpreter.Options()
            interpreter = Interpreter(modelFile, options)

            val inputTensor = interpreter!!.getInputTensor(0)
            val outputTensor = interpreter!!.getOutputTensor(0)

            inputImageWidth = inputTensor.shape()[1]
            inputImageHeight = inputTensor.shape()[2]
            inputDataType = inputTensor.dataType()
            outputDataType = outputTensor.dataType()
            outputShape = outputTensor.shape()

            Log.d("YOLO", "Model: $inputImageWidth x $inputImageHeight | Input: $inputDataType | Output: $outputDataType")
        } catch (e: Exception) {
            Log.e("YOLO", "Error loading model", e)
        }
    }

    fun detect(bitmap: Bitmap, originalImageWidth: Int, originalImageHeight: Int): List<DetectionResult> {
        val interpreter = interpreter ?: return emptyList()

        // 1. PREPROCESS: Handle Quantization (Int8 vs Float32)
        val imageProcessorBuilder = ImageProcessor.Builder()

        // IMPORTANT: INT8 models expect 0-255. Float32 models expect 0.0-1.0.
        if (inputDataType == DataType.FLOAT32) {
            imageProcessorBuilder.add(NormalizeOp(0f, 255f)) // Normalize to 0-1
        } else {
            imageProcessorBuilder.add(CastOp(DataType.UINT8)) // Keep as 0-255 Integer
        }

        val imageProcessor = imageProcessorBuilder.build()
        var tensorImage = TensorImage(inputDataType)
        tensorImage.load(bitmap)
        tensorImage = imageProcessor.process(tensorImage)

        // 2. OUTPUT BUFFER
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, outputDataType)

        // 3. RUN INFERENCE
        interpreter.run(tensorImage.buffer, outputBuffer.buffer.rewind())

        // 4. DEQUANTIZE & PARSE
        // If output is Int8, we MUST dequantize it to get real coordinates.
        val outputArray: FloatArray
        if (outputDataType == DataType.UINT8 || outputDataType == DataType.INT8) {
            // Get Quantization Parameters
            val tensor = interpreter.getOutputTensor(0)
            val scale = tensor.quantizationParams().scale
            val zeroPoint = tensor.quantizationParams().zeroPoint.toFloat()

            val rawFloats = outputBuffer.floatArray // This just casts int to float, doesn't dequantize
            outputArray = FloatArray(rawFloats.size)
            for (i in rawFloats.indices) {
                outputArray[i] = (rawFloats[i] - zeroPoint) * scale
            }
        } else {
            outputArray = outputBuffer.floatArray
        }

        // 5. PARSE YOLO OUPUT
        val rows = outputShape[1] // 4 coords + classes
        val cols = outputShape[2] // Anchors

        val detections = ArrayList<DetectionResult>()
        val scaleX = originalImageWidth.toFloat() / inputImageWidth
        val scaleY = originalImageHeight.toFloat() / inputImageHeight

        for (c in 0 until cols) {
            val cx = outputArray[0 * cols + c]
            val cy = outputArray[1 * cols + c]
            val w = outputArray[2 * cols + c]
            val h = outputArray[3 * cols + c]

            var maxScore = 0f
            var maxClassIndex = -1

            for (r in 4 until rows) {
                val score = outputArray[r * cols + c]
                if (score > maxScore) {
                    maxScore = score
                    maxClassIndex = r - 4
                }
            }

            if (maxScore > 0.50f) {
                val x1 = (cx - w / 2) * inputImageWidth
                val y1 = (cy - h / 2) * inputImageHeight
                val x2 = (cx + w / 2) * inputImageWidth
                val y2 = (cy + h / 2) * inputImageHeight

                // Map back to original image size
                val finalX1 = x1 * scaleX
                val finalY1 = y1 * scaleY
                val finalX2 = x2 * scaleX
                val finalY2 = y2 * scaleY

                if (maxClassIndex in labels.indices) {
                    detections.add(DetectionResult(
                        RectF(finalX1, finalY1, finalX2, finalY2),
                        maxScore,
                        labels[maxClassIndex]
                    ))
                }
            }
        }
        return nms(detections)
    }

    private fun nms(detections: List<DetectionResult>): List<DetectionResult> {
        val sorted = detections.sortedByDescending { it.confidence }
        val selected = ArrayList<DetectionResult>()

        for (det in sorted) {
            var overlap = false
            for (sel in selected) {
                val iou = calculateIoU(det.box, sel.box)
                if (iou > 0.45f) { // NMS Threshold
                    overlap = true
                    break
                }
            }
            if (!overlap) selected.add(det)
        }
        return selected
    }

    private fun calculateIoU(a: RectF, b: RectF): Float {
        val left = maxOf(a.left, b.left)
        val top = maxOf(a.top, b.top)
        val right = minOf(a.right, b.right)
        val bottom = minOf(a.bottom, b.bottom)
        if (right <= left || bottom <= top) return 0f
        val intersection = (right - left) * (bottom - top)
        val union = (a.width() * a.height()) + (b.width() * b.height()) - intersection
        return intersection / union
    }

    private fun loadModelFile(path: String): java.nio.MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(path)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }
}