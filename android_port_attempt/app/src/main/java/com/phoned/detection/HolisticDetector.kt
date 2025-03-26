package com.phoned.detection

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage

class HolisticDetector(private val context: Context) {
    private var poseLandmarker: PoseLandmarker? = null

    init {
        setupPoseLandmarker()
    }

    private fun setupPoseLandmarker() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("pose_landmarker_lite.task")
            .setDelegate(Delegate.CPU)
            .build()

        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE)
            .setNumPoses(1)
            .setMinPoseDetectionConfidence(0.5f)
            .setMinPosePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .build()

        try {
            poseLandmarker = PoseLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            println("Pose landmarker initialization error: ${e.message}")
        }
    }

    fun detect(bitmap: Bitmap): PoseLandmarkerResult? {
        return try {
            // Convert Bitmap to MPImage
            val mpImage = BitmapImageBuilder(bitmap).build()

            val result = poseLandmarker?.detect(mpImage)

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
            poseLandmarker?.close()
        } catch (e: Exception) {
            println("Error closing pose landmarker: ${e.message}")
        }
    }
}