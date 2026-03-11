package com.lbwma.cnn.network

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream

object ThumbnailCache {
    private lateinit var baseDir: File
    private val client = OkHttpClient()
    private const val THUMB_SIZE = 300
    private const val THUMB_QUALITY = 70

    fun init(context: Context) {
        baseDir = File(context.filesDir, "thumbnails")
    }

    private fun dir(conversor: String): File {
        val d = File(baseDir, conversor)
        d.mkdirs()
        return d
    }

    fun getFile(conversor: String, foto: String): File = File(dir(conversor), foto)

    fun exists(conversor: String, foto: String): Boolean = getFile(conversor, foto).exists()

    suspend fun generate(conversor: String, foto: String): Boolean = withContext(Dispatchers.IO) {
        val file = getFile(conversor, foto)
        if (file.exists()) return@withContext true

        try {
            val url = ApiClient.getFotoUrl(conversor, foto)
            val request = Request.Builder()
                .url(url)
                .header("Authorization", ApiClient.getAuthHeader())
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return@withContext false
                val bytes = response.body?.bytes() ?: return@withContext false

                // Decodifica com downsampling pra economizar memória
                val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)

                val longerSide = maxOf(opts.outWidth, opts.outHeight)
                var sampleSize = 1
                while (longerSide / sampleSize > THUMB_SIZE * 2) sampleSize *= 2

                val decodeOpts = BitmapFactory.Options().apply { inSampleSize = sampleSize }
                val sampled = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, decodeOpts)
                    ?: return@withContext false

                // Redimensiona pro tamanho final
                val scale = THUMB_SIZE.toFloat() / maxOf(sampled.width, sampled.height)
                val w = (sampled.width * scale).toInt()
                val h = (sampled.height * scale).toInt()
                val thumb = Bitmap.createScaledBitmap(sampled, w, h, true)
                sampled.recycle()

                // Salva
                FileOutputStream(file).use { out ->
                    thumb.compress(Bitmap.CompressFormat.JPEG, THUMB_QUALITY, out)
                }
                thumb.recycle()
                true
            }
        } catch (_: Exception) {
            false
        }
    }

    fun clearConversor(conversor: String) {
        dir(conversor).deleteRecursively()
    }
}
