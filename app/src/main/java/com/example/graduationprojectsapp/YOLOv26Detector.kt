package com.example.graduationprojectsapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * YOLOv26n TFLite detector — ported from the Ultralytics Flutter Android implementation.
 *
 * Correctly handles both output tensor layouts that Ultralytics TFLite exports produce:
 *
 *   TRANSPOSED  →  [1, 4+classes, anchors]   e.g. [1, 84, 8400]
 *   STANDARD    →  [1, anchors, 4+classes]   e.g. [1, 8400, 84]
 *
 * The layout is auto-detected at init time by comparing the two output dimensions:
 * the smaller dimension is always features (4+classes), the larger is always anchors.
 *
 * Preprocessing uses Ultralytics-style letterboxing (aspect-ratio-preserving resize
 * + black padding), and output coordinates are un-letterboxed back to original image space.
 */
class YOLOv26Detector(
    private val context: Context,
    val modelName: String,
    private val labels: List<String>,
    useGpu: Boolean = false,
    private val confidenceThreshold: Float = 0.25f,
    private val nmsThreshold: Float = 0.45f
) {
    var inputImageWidth  = 0; private set
    var inputImageHeight = 0; private set

    private var interpreter: Interpreter? = null

    // ── Reusable per-frame buffers ────────────────────────────────────────────
    private lateinit var scaledBitmap: Bitmap
    private lateinit var intValues:    IntArray
    private lateinit var inputBuffer:  ByteBuffer
    private lateinit var rawOutput:    Array<Array<FloatArray>>  // always [1][dim1][dim2]

    // Detected at init: how to index rawOutput during parsing
    private var numAnchors  = 0   // total anchor candidates
    private var numClasses  = 0   // number of class scores per anchor
    private var isTransposed = false  // true → [1, features, anchors]; false → [1, anchors, features]
    private var isPostProcessed = false // true → [box(4), score(1), class(1)] format

    // Detected at init: whether model outputs normalized [0,1] or pixel-space coords.
    // Sampled across several anchors at init, NOT re-evaluated per anchor per frame.
    private var outputIsNormalized = false

    private val blackPaint = Paint().apply { color = Color.BLACK }

    init {
        try {
            val modelBuffer = loadModelFile(modelName)

            val options = Interpreter.Options().apply {
                setNumThreads(Runtime.getRuntime().availableProcessors())
                if (useGpu) {
                    try   { addDelegate(GpuDelegate()); Log.i(TAG, "GPU delegate enabled") }
                    catch (e: Exception) { Log.w(TAG, "GPU unavailable, using CPU: ${e.message}") }
                }
            }

            val interp = Interpreter(modelBuffer, options)
            interp.allocateTensors()
            interpreter = interp

            // ── Input shape: [1, H, W, 3] ─────────────────────────────────────
            val inShape = interp.getInputTensor(0).shape()
            require(inShape.size == 4 && inShape[3] == 3) {
                "Expected input [1,H,W,3], got ${inShape.toList()}"
            }
            inputImageHeight = inShape[1]
            inputImageWidth  = inShape[2]

            // ── Output shape: [1, dim1, dim2] ─────────────────────────────────
            val outShape = interp.getOutputTensor(0).shape()
            require(outShape.size == 3) {
                "Expected 3D output [1, dim1, dim2], got ${outShape.toList()}"
            }
            val dim1 = outShape[1]
            val dim2 = outShape[2]

            // AUTO-DETECT layout:
            //   Features (4+classes) are always a small number (e.g. 17 for 13-class, 84 for 80-class).
            //   Anchors are always large (2100 for 320px, 8400 for 640px).
            //   → whichever dimension is smaller = features; larger = anchors.
            //
            //   Transposed  [1, features, anchors]: dim1 < dim2  →  rawOutput[0][featureRow][anchor]
            //   Standard    [1, anchors, features]: dim1 > dim2  →  rawOutput[0][anchor][featureCol]
            isTransposed = dim1 < dim2
            val featureDim = if (isTransposed) dim1 else dim2
            val anchorDim  = if (isTransposed) dim2 else dim1

            // Detection models (Post-processed) often have shape [1, 100/300, 6] 
            // where 6 = [x1, y1, x2, y2, score, class_index]
            isPostProcessed = (featureDim == 6 && anchorDim <= 1000)

            if (isPostProcessed) {
                numClasses = labels.size // we will use the class index from the model
                numAnchors = anchorDim
            } else {
                numClasses  = featureDim - 4
                numAnchors  = anchorDim
            }

            require(numClasses > 0) {
                "Feature dimension too small ($featureDim). Got output shape ${outShape.toList()}"
            }

            // Allocate rawOutput to match ACTUAL tensor shape (dim1 × dim2, not feature × anchor)
            rawOutput = Array(1) { Array(dim1) { FloatArray(dim2) } }

            // Allocate preprocessing buffers
            scaledBitmap = Bitmap.createBitmap(inputImageWidth, inputImageHeight, Bitmap.Config.ARGB_8888)
            intValues    = IntArray(inputImageWidth * inputImageHeight)
            inputBuffer  = ByteBuffer
                .allocateDirect(inputImageWidth * inputImageHeight * 3 * Float.SIZE_BYTES)
                .apply { order(ByteOrder.nativeOrder()) }

            // ── Detect coordinate space (normalized vs pixel) ONCE at init ────
            // Run a dummy inference on a blank bitmap to sample output values.
            // If all cx/cy/w/h values across 50 sampled anchors are ≤ 2.0 → normalized.
            // This is done ONCE here rather than re-checked per anchor per frame.
            inputBuffer.clear()
            repeat(inputImageWidth * inputImageHeight * 3) { inputBuffer.putFloat(0f) }
            inputBuffer.rewind()
            interp.run(inputBuffer, rawOutput)

            var maxCoord = 0f
            val sampleStep = maxOf(1, numAnchors / 50)
            for (a in 0 until numAnchors step sampleStep) {
                val cx: Float; val cy: Float; val bw: Float; val bh: Float
                if (isTransposed) {
                    cx = rawOutput[0][0][a]; cy = rawOutput[0][1][a]
                    bw = rawOutput[0][2][a]; bh = rawOutput[0][3][a]
                } else {
                    cx = rawOutput[0][a][0]; cy = rawOutput[0][a][1]
                    bw = rawOutput[0][a][2]; bh = rawOutput[0][a][3]
                }
                maxCoord = maxOf(maxCoord, cx, cy, bw, bh)
            }
            outputIsNormalized = maxCoord <= 2f

            Log.i(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            Log.i(TAG, "YOLOv26Detector  ▸  $modelName")
            Log.i(TAG, "  Input    : ${inputImageWidth}×${inputImageHeight}")
            Log.i(TAG, "  Output   : [1, $dim1, $dim2]  →  ${if (isTransposed) "TRANSPOSED [features×anchors]" else "STANDARD [anchors×features]"}")
            Log.i(TAG, "  Anchors  : $numAnchors  |  Classes: $numClasses  |  Labels: ${labels.size}")
            Log.i(TAG, "  Coord    : ${if (outputIsNormalized) "normalized [0,1]" else "pixel-space"}  (maxSampled=$maxCoord)")
            if (numClasses != labels.size) {
                Log.w(TAG, "  ⚠ MODEL has $numClasses classes but labels.txt has ${labels.size} entries.")
                Log.w(TAG, "    Update yolo_labels.txt to match your model's training classes.")
            }
            Log.i(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        } catch (e: Exception) {
            Log.e(TAG, "Init failed for $modelName: ${e.message}", e)
        }
    }

    // ── Public inference ──────────────────────────────────────────────────────

    fun detect(bitmap: Bitmap, origWidth: Int, origHeight: Int): List<DetectionResult> {
        val interp = interpreter ?: return emptyList()

        // ── Step 1: Letterbox ─────────────────────────────────────────────────
        val gain     = min(inputImageWidth.toFloat() / origWidth, inputImageHeight.toFloat() / origHeight)
        val resizedW = (origWidth  * gain).roundToInt()
        val resizedH = (origHeight * gain).roundToInt()
        val padX     = ((inputImageWidth  - resizedW) / 2f - 0.1f).roundToInt().toFloat()
        val padY     = ((inputImageHeight - resizedH) / 2f - 0.1f).roundToInt().toFloat()

        Canvas(scaledBitmap).apply {
            drawRect(0f, 0f, inputImageWidth.toFloat(), inputImageHeight.toFloat(), blackPaint)
            drawBitmap(bitmap, Rect(0, 0, origWidth, origHeight),
                RectF(padX, padY, padX + resizedW, padY + resizedH), null)
        }

        // ── Step 2: Fill input ByteBuffer (float32 RGB, normalized 0–1) ──────
        scaledBitmap.getPixels(intValues, 0, inputImageWidth, 0, 0, inputImageWidth, inputImageHeight)
        inputBuffer.clear()
        for (pixel in intValues) {
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
            inputBuffer.putFloat(((pixel shr 8)  and 0xFF) / 255f)
            inputBuffer.putFloat(( pixel         and 0xFF) / 255f)
        }
        inputBuffer.rewind()

        // ── Step 3: Inference ─────────────────────────────────────────────────
        interp.run(inputBuffer, rawOutput)

        // ── Step 4: Parse output — handle BOTH tensor layouts ─────────────────
        val detections = ArrayList<DetectionResult>(64)

        for (anchor in 0 until numAnchors) {

            // Read cx/cy/w/h and class scores according to detected layout
            var cx: Float; var cy: Float; var bw: Float; var bh: Float
            var bestScore: Float
            var bestClass: Int

            if (isPostProcessed) {
                // Format: [x1, y1, x2, y2, score, class_index]
                if (isTransposed) {
                    cx = rawOutput[0][0][anchor]; cy = rawOutput[0][1][anchor]
                    bw = rawOutput[0][2][anchor]; bh = rawOutput[0][3][anchor]
                    bestScore = rawOutput[0][4][anchor]
                    bestClass = rawOutput[0][5][anchor].roundToInt()
                } else {
                    cx = rawOutput[0][anchor][0]; cy = rawOutput[0][anchor][1]
                    bw = rawOutput[0][anchor][2]; bh = rawOutput[0][anchor][3]
                    bestScore = rawOutput[0][anchor][4]
                    bestClass = rawOutput[0][anchor][5].roundToInt()
                }
                
                // Handle 0-100 score normalization
                if (bestScore > 1.0f) bestScore /= 100f
                
                if (bestScore < confidenceThreshold) continue
            } else {
                if (isTransposed) {
                    cx = rawOutput[0][0][anchor]; cy = rawOutput[0][1][anchor]
                    bw = rawOutput[0][2][anchor]; bh = rawOutput[0][3][anchor]
                } else {
                    cx = rawOutput[0][anchor][0]; cy = rawOutput[0][anchor][1]
                    bw = rawOutput[0][anchor][2]; bh = rawOutput[0][anchor][3]
                }

                // Find best class score — short-circuit below threshold
                bestScore = confidenceThreshold
                bestClass = -1
                for (cls in 0 until numClasses) {
                    val score = if (isTransposed) rawOutput[0][4 + cls][anchor]
                                else             rawOutput[0][anchor][4 + cls]
                    if (score > bestScore) { bestScore = score; bestClass = cls }
                }
            }
            
            if (bestClass < 0 || bestClass !in labels.indices) continue

            // ── Step 5: Convert to model-pixel space ──────────────────────────
            // outputIsNormalized is determined ONCE at init, not per-anchor.
            val modelV1 = if (outputIsNormalized) cx * inputImageWidth  else cx
            val modelV2 = if (outputIsNormalized) cy * inputImageHeight else cy
            val modelV3 = if (outputIsNormalized) bw * inputImageWidth  else bw
            val modelV4 = if (outputIsNormalized) bh * inputImageHeight else bh

            val modelX1: Float; val modelY1: Float; val modelX2: Float; val modelY2: Float
            
            if (isPostProcessed) {
                // Coordinates are already x1, y1, x2, y2
                modelX1 = modelV1; modelY1 = modelV2; modelX2 = modelV3; modelY2 = modelV4
            } else {
                // Coordinates are cx, cy, w, h
                modelX1 = modelV1 - modelV3 / 2f
                modelY1 = modelV2 - modelV4 / 2f
                modelX2 = modelV1 + modelV3 / 2f
                modelY2 = modelV2 + modelV4 / 2f
            }

            // ── Step 6: Un-letterbox → original image coordinates ─────────────
            val x1 = ((modelX1 - padX) / gain).coerceIn(0f, origWidth.toFloat())
            val y1 = ((modelY1 - padY) / gain).coerceIn(0f, origHeight.toFloat())
            val x2 = ((modelX2 - padX) / gain).coerceIn(0f, origWidth.toFloat())
            val y2 = ((modelY2 - padY) / gain).coerceIn(0f, origHeight.toFloat())

            if (x2 > x1 && y2 > y1) {
                detections.add(DetectionResult(RectF(x1, y1, x2, y2), bestScore, labels[bestClass]))
            }
        }

        return applyNms(detections)
    }

    // ── NMS ───────────────────────────────────────────────────────────────────

    private fun applyNms(detections: List<DetectionResult>): List<DetectionResult> {
        val sorted   = detections.sortedByDescending { it.confidence }
        val selected = ArrayList<DetectionResult>(sorted.size)
        for (det in sorted) {
            val suppressed = selected.any { sel -> iou(det.box, sel.box) > nmsThreshold }
            if (!suppressed) selected.add(det)
        }
        return selected
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left   = maxOf(a.left, b.left);  val top    = maxOf(a.top, b.top)
        val right  = minOf(a.right, b.right); val bottom = minOf(a.bottom, b.bottom)
        if (right <= left || bottom <= top) return 0f
        val inter = (right - left) * (bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun loadModelFile(path: String): java.nio.MappedByteBuffer {
        val fd = context.assets.openFd(path)
        return FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    fun close() { interpreter?.close(); interpreter = null }

    companion object { private const val TAG = "YOLOv26Detector" }
}
