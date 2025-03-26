package com.phoned.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

class PhoneDetectionAnalyzer(
    private val context: Context,
    private val onFrameProcessed: (Canvas) -> Unit
) : ImageAnalysis.Analyzer {
    private val holisticDetector = HolisticDetector(context)
    private val objectDetector = ObjectDetector(context)

    @androidx.camera.core.ExperimentalGetImage
    override fun analyze(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        if (mediaImage != null) {
            try {
                val bitmap = mediaImage.toBitmap()
                // Process with holistic detector
                val holisticResult = holisticDetector.detect(bitmap)

                // Process with object detector
                val objectResult = objectDetector.detect(bitmap)

                // Process results as needed
                processResults(holisticResult, objectResult)

            } catch (e: Exception) {
                println("Analysis error: ${e.message}")
            }
        }
        imageProxy.close()
    }

    private fun android.media.Image.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun processResults(
        holisticResult: com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult?,
        objectResult: com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult?
    ) {
        // Process holistic detection results
        holisticResult?.let { result ->
            result.landmarks().forEach { poseLandmarks ->
                poseLandmarks.forEach { landmark ->
                    val x = landmark.x()
                    val y = landmark.y()
                    val z = landmark.z()
                    val visibility = landmark.visibility()
                    // Use landmark data as needed
                }
            }
        }

        // Process object detection results
        objectResult?.let { result ->
            result.detections().forEach { detection ->
                val boundingBox = detection.boundingBox()
                detection.categories().forEach { category ->
                    val label = category.categoryName()
                    val score = category.score()
                    // Use detection data as needed
                }
            }
        }
    }

    fun close() {
        holisticDetector.close()
        objectDetector.close()
    }
}