package com.example.graduationprojectsapp

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.usb.UsbManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.graduationprojectsapp.databinding.ActivityMainBinding
import com.hoho.android.usbserial.driver.UsbSerialPort
import com.hoho.android.usbserial.driver.UsbSerialProber
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var binding: ActivityMainBinding

    // ── Camera ────────────────────────────────────────────────────────────────
    private val cameraExecutor = Executors.newSingleThreadExecutor()

    // ── YOLO detectors ────────────────────────────────────────────────────────
    // Labels are shared by both detectors (same training classes)
    private val labelFilename = "yolo_labels.txt"

    // YOLOv26Detector: letterboxing + direct ByteBuffer + GPU-capable
    private val v26ModelName = "Venom_v7YOLOv26n_FP16.tflite"

    // YOLOv11Detector: stretch-resize preprocessing, for Venom custom models
    private val v11ModelName = "Venom_v5YOLOv11n_FP16.tflite"

    // ── Detector switch ────────────────────────────────────────────────────────
    // USE_V26 = true  → YOLOv26Detector (letterboxing, handles any Ultralytics TFLite layout)
    //                   Requires yolov26n.tflite in assets/. Check Logcat tag "YOLOv26Detector"
    //                   for a startup report showing detected shape, layout, and coord space.
    // USE_V26 = false → YOLOv11Detector (stretch-resize, for the existing Venom models)
    //
    // IMPORTANT: The Venom models (Venom_v*.tflite) were trained with stretch-resize, NOT
    // letterboxing. Running them through YOLOv26Detector (USE_V26=true) will produce shifted
    // boxes because the preprocessing doesn't match the training. Use USE_V26=false for Venom.
    private val USE_V26 = false    // ← set true only when yolov26n.tflite is in assets/

    private lateinit var v26Detector: YOLOv26Detector
    private lateinit var v11Detector: YOLOv11Detector

    // ── Depth ─────────────────────────────────────────────────────────────────
    private lateinit var depthEstimator: DepthEstimator
    private val depthExecutor = Executors.newSingleThreadExecutor()
    private val isDepthRunning = AtomicBoolean(false)
    private val DEPTH_FRAME_INTERVAL = 3
    private var frameCount = 0

    // ── USB / ESP32 ───────────────────────────────────────────────────────────
    private var usbSerialPort: UsbSerialPort? = null
    private val BAUD_RATE = 115200
    private var lastYoloSignalTime = 0L
    private var lastHazardSignalTime = 0L
    private val YOLO_DEBOUNCE_MS  = 1_000L
    private val HAZARD_DEBOUNCE_MS = 2_000L

    // ── Sensors ───────────────────────────────────────────────────────────────
    private lateinit var sensorManager: SensorManager
    private var rotationSensor: Sensor? = null
    private var lastGyroSignalTime = 0L
    private val GYRO_DEBOUNCE_MS = 200L

    // ─────────────────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val labels = loadLabels(labelFilename)

        // Always init v11 (for Venom models, no asset required)
        v11Detector = YOLOv11Detector(this, v11ModelName, labels)

        // Init v26 — requires yolov26n.tflite in assets/
        // If the model file is missing, v26 will log an error and return empty detections.
        v26Detector = YOLOv26Detector(
            context           = this,
            modelName         = v26ModelName,
            labels            = labels,
            useGpu            = false,        // set true to enable GPU delegate
            confidenceThreshold = 0.25f,
            nmsThreshold      = 0.45f
        )

        depthEstimator = DepthEstimator(this)

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        connectToUsbDevice()

        // Init Sensors
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    }

    override fun onResume() {
        super.onResume()
        rotationSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI)
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    // ── Camera setup ──────────────────────────────────────────────────────────

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()

            imageAnalyzer.setAnalyzer(cameraExecutor) { imageProxy ->
                processFrame(imageProxy)
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
                Toast.makeText(this, "Camera failed to start", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ── Per-frame processing ──────────────────────────────────────────────────

    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val bitmap = imageProxyToBitmap(imageProxy) ?: return
            val originalWidth  = bitmap.width
            val originalHeight = bitmap.height

            // ── YOLO detection ────────────────────────────────────────────────
            val detections: List<DetectionResult>

            if (USE_V26) {
                // YOLOv26Detector handles letterboxing internally — pass full bitmap
                detections = v26Detector.detect(bitmap, originalWidth, originalHeight)
            } else {
                // YOLOv11Detector expects a pre-scaled bitmap (stretch resize)
                val scaledForYolo = Bitmap.createScaledBitmap(bitmap, v11Detector.inputImageWidth, v11Detector.inputImageHeight, true)
                detections = v11Detector.detect(scaledForYolo, originalWidth, originalHeight)
            }

            // ── Depth every N frames (non-blocking) ───────────────────────────
            frameCount++
            if (frameCount % DEPTH_FRAME_INTERVAL == 0 && isDepthRunning.compareAndSet(false, true)) {
                val depthInput = Bitmap.createScaledBitmap(bitmap, 480, 480, true)
                depthExecutor.execute { runDepthPipeline(depthInput) }
            }

            // ── UI: YOLO boxes ────────────────────────────────────────────────
            runOnUiThread {
                binding.overlayView.setResults(detections, originalWidth, originalHeight)
            }

            // ── USB: YOLO detections ──────────────────────────────────────────
            handleYoloDetections(detections)

        } finally {
            imageProxy.close()
        }
    }

    // ── Depth pipeline ───────────────────────────────────────────────

    private fun runDepthPipeline(inputBitmap: Bitmap) {
        try {
            val (rawDepth, colorizedDepth) = depthEstimator.estimate(inputBitmap)

            runOnUiThread {
                binding.depthOverlayImage.setImageBitmap(colorizedDepth)
                binding.depthOverlayImage.visibility = View.VISIBLE
            }

        } catch (e: Exception) {
            Log.e(TAG, "Depth pipeline error", e)
        } finally {
            isDepthRunning.set(false)
        }
    }

    // ── YOLO → ESP32 ──────────────────────────────────────────────────────────

    private fun handleYoloDetections(results: List<DetectionResult>) {
        val now = System.currentTimeMillis()
        if (now - lastYoloSignalTime < YOLO_DEBOUNCE_MS) return

        val best = results.maxByOrNull { it.confidence } ?: return

        // Format: "YOLO:label,confidence"  e.g.  "YOLO:person,0.91"
        val message = "YOLO:${best.className.lowercase()},${String.format("%.2f", best.confidence)}"
        sendToEsp32(message)
        lastYoloSignalTime = now
        Log.d(TAG, "YOLO → ESP32: $message")
    }

    // ── USB serial ────────────────────────────────────────────────────────────

    private fun sendToEsp32(message: String) {
        usbSerialPort?.let { port ->
            if (port.isOpen) {
                try {
                    port.write((message + "\n").toByteArray(), 1000)
                } catch (e: IOException) {
                    Log.e(TAG, "USB write failed: ${e.message}")
                }
            }
        }
    }

    // ── Sensors implementation ───────────────────────────────────────────────

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_ROTATION_VECTOR) {
            handleGyroData(event.values)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun handleGyroData(rotationVector: FloatArray) {
        val now = System.currentTimeMillis()
        if (now - lastGyroSignalTime < GYRO_DEBOUNCE_MS) return

        val rotationMatrix = FloatArray(9)
        SensorManager.getRotationMatrixFromVector(rotationMatrix, rotationVector)

        val orientation = FloatArray(3)
        SensorManager.getOrientation(rotationMatrix, orientation)

        // orientation[0] = azimuth (yaw), orientation[1] = pitch, orientation[2] = roll
        // Convert to degrees
        val yaw   = Math.toDegrees(orientation[0].toDouble())
        val pitch = Math.toDegrees(orientation[1].toDouble())
        val roll  = Math.toDegrees(orientation[2].toDouble())

        val message = "GYRO:${String.format("%.2f", yaw)},${String.format("%.2f", pitch)},${String.format("%.2f", roll)}"
        sendToEsp32(message)
        lastGyroSignalTime = now
    }

    private fun connectToUsbDevice() {
        val manager = getSystemService(Context.USB_SERVICE) as UsbManager
        val drivers = UsbSerialProber.getDefaultProber().findAllDrivers(manager)
        if (drivers.isEmpty()) return

        val driver = drivers[0]
        val connection = manager.openDevice(driver.device) ?: return

        usbSerialPort = driver.ports[0]
        try {
            usbSerialPort?.open(connection)
            usbSerialPort?.setParameters(BAUD_RATE, 8, UsbSerialPort.STOPBITS_1, UsbSerialPort.PARITY_NONE)
            runOnUiThread {
                binding.usbStatusText.text = "\u25CF USB"
                binding.usbStatusText.setTextColor(
                    ContextCompat.getColor(this, android.R.color.holo_green_light)
                )
                Toast.makeText(this, "ESP32 Connected", Toast.LENGTH_SHORT).show()
            }
        } catch (e: IOException) {
            Log.e(TAG, "USB open failed", e)
            usbSerialPort = null
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun loadLabels(fileName: String): List<String> {
        return try {
            assets.open(fileName).bufferedReader().readLines().filter { it.isNotBlank() }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load labels: ${e.message}")
            emptyList()
        }
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        return try {
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()
            val nv21  = ByteArray(ySize + uSize + vSize)

            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
            val imageBytes = out.toByteArray()
            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

            val rotation = image.imageInfo.rotationDegrees
            if (rotation != 0) {
                val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            } else {
                bitmap
            }
        } catch (e: Exception) {
            Log.e(TAG, "imageProxyToBitmap failed", e)
            null
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) startCamera()
            else { Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show(); finish() }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        depthExecutor.shutdown()
        v26Detector.close()
        depthEstimator.close()
        try { usbSerialPort?.close() } catch (e: IOException) { /* ignored */ }
    }

    companion object {
        private const val TAG = "FusionApp"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
