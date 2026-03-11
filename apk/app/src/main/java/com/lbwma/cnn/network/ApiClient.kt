package com.lbwma.cnn.network

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.Credentials
import okhttp3.Interceptor
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.net.URLEncoder
import java.util.concurrent.TimeUnit

data class Foto(val nome: String, val tamanhoKb: Double)

object ApiClient {
    var baseUrl: String = ""
        private set
    private var credentials: String = ""
    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .build()

    fun configure(url: String, user: String, pass: String) {
        baseUrl = url.trimEnd('/')
        credentials = Credentials.basic(user, pass)
    }

    private fun authRequest(url: String): Request.Builder =
        Request.Builder().url(url).header("Authorization", credentials)

    private fun encode(value: String): String = URLEncoder.encode(value, "UTF-8")

    suspend fun testConnection(): Boolean = withContext(Dispatchers.IO) {
        try {
            val request = authRequest("$baseUrl/conversores").get().build()
            client.newCall(request).execute().use { it.isSuccessful }
        } catch (_: Exception) {
            false
        }
    }

    suspend fun getConversores(): Result<List<String>> = withContext(Dispatchers.IO) {
        try {
            val request = authRequest("$baseUrl/conversores").get().build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return@withContext Result.failure(Exception("Erro ${response.code}"))
                val body = response.body?.string() ?: "[]"
                val json = try {
                    JSONArray(body)
                } catch (_: Exception) {
                    JSONObject(body).getJSONArray("conversores")
                }
                val list = (0 until json.length()).map { i ->
                    val item = json.get(i)
                    if (item is JSONObject) item.getString("nome") else item.toString()
                }
                Result.success(list)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun createConversor(nome: String): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val request = authRequest("$baseUrl/conversores?nome=${encode(nome)}")
                .post("".toRequestBody(null))
                .build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return@withContext Result.failure(Exception("Erro ${response.code}"))
                Result.success(Unit)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getFotos(nome: String): Result<List<Foto>> = withContext(Dispatchers.IO) {
        try {
            val request = authRequest("$baseUrl/conversores/${encode(nome)}/fotos").get().build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return@withContext Result.failure(Exception("Erro ${response.code}"))
                val body = response.body?.string() ?: "[]"
                val fotosArray = try {
                    JSONArray(body)
                } catch (_: Exception) {
                    JSONObject(body).getJSONArray("fotos")
                }
                val list = (0 until fotosArray.length()).map { i ->
                    val item = fotosArray.get(i)
                    if (item is JSONObject) {
                        Foto(
                            nome = item.getString("nome"),
                            tamanhoKb = item.optDouble("tamanho_kb", 0.0)
                        )
                    } else {
                        Foto(nome = item.toString(), tamanhoKb = 0.0)
                    }
                }
                Result.success(list)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun uploadFoto(nome: String, fileName: String, bytes: ByteArray): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val body = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("fotos", fileName, bytes.toRequestBody("image/*".toMediaType()))
                .build()
            val request = authRequest("$baseUrl/conversores/${encode(nome)}/fotos")
                .post(body)
                .build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return@withContext Result.failure(Exception("Erro ${response.code}"))
                Result.success(Unit)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun deleteFoto(nome: String, arquivo: String): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val request = authRequest("$baseUrl/conversores/${encode(nome)}/fotos/${encode(arquivo)}")
                .delete()
                .build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return@withContext Result.failure(Exception("Erro ${response.code}"))
                Result.success(Unit)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    fun getFotoUrl(nome: String, arquivo: String): String =
        "$baseUrl/conversores/${encode(nome)}/fotos/${encode(arquivo)}/download"

    fun getAuthHeader(): String = credentials

    fun getAuthInterceptor(): Interceptor = Interceptor { chain ->
        val request = chain.request().newBuilder()
            .header("Authorization", credentials)
            .build()
        chain.proceed(request)
    }
}
