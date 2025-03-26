package com.phoned.detection

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage

class ObjectDetector(private val context: Context) {
    private var detector: ObjectDetector? = null

    init {
        setupDetector()
    }

    private fun setupDetector() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("efficientdet_lite0.tflite")
            .setDelegate(Delegate.CPU)
            .build()

        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE)
            .setMaxResults(5)
            .setScoreThreshold(0.5f)
            .build()

        try {
            detector = ObjectDetector.createFromOptions(context, options)
        } catch (e: Exception) {
            println("Object detector initialization error: ${e.message}")
        }
    }

    fun detect(bitmap: Bitmap): ObjectDetectorResult? {
        return try {
            // Convert Bitmap to MPImage
            val mpImage = BitmapImageBuilder(bitmap).build()

            val result = detector?.detect(mpImage)

            // Clean up
            mpImage.close()

            result
        } catch (e: Exception) {
            println("Detection error: ${e.message}")
            null
        }
    }

    fun close() {
        try {
            detector?.close()
        } catch (e: Exception) {
            println("Error closing detector: ${e.message}")
        }
    }
}