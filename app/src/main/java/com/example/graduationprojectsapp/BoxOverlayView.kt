package com.example.graduationprojectsapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class BoxOverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    private var results: List<DetectionResult> = emptyList()
    private var imageWidth = 640
    private var imageHeight = 640

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 50f
        style = Paint.Style.FILL
        isAntiAlias = true
        setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }

    // Call this from MainActivity to update detections
    fun setResults(detectionResults: List<DetectionResult>, imgW: Int, imgH: Int) {
        results = detectionResults
        imageWidth = imgW
        imageHeight = imgH
        invalidate() // Trigger redraw
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // --- THE FIX: Calculate "Center Crop" Scaling ---
        // This matches the default behavior of PreviewView.ScaleType.FILL_CENTER
        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()

        val scaleX = viewWidth / imageWidth
        val scaleY = viewHeight / imageHeight

        // Scale to fill the larger dimension (zoom in)
        val scale = maxOf(scaleX, scaleY)

        // Center the image
        val offsetX = (viewWidth - imageWidth * scale) / 2
        val offsetY = (viewHeight - imageHeight * scale) / 2

        results.forEach { result ->
            val box = result.box

            // Apply scale and offset to map "Image Coordinates" to "Screen Coordinates"
            val left = box.left * scale + offsetX
            val top = box.top * scale + offsetY
            val right = box.right * scale + offsetX
            val bottom = box.bottom * scale + offsetY

            val screenBox = RectF(left, top, right, bottom)

            // Draw
            canvas.drawRect(screenBox, boxPaint)
            canvas.drawText(
                "${result.className} ${(result.confidence * 100).toInt()}%",
                left,
                top - 20f,
                textPaint
            )
        }
    }
}