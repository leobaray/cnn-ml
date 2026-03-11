package com.lbwma.cnn.screen

import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import android.util.Size
import androidx.activity.compose.BackHandler
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.content.getSystemService
import androidx.lifecycle.compose.LocalLifecycleOwner
import kotlinx.coroutines.delay
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

@Composable
fun CameraScreen(
    onPhotosTaken: (List<File>) -> Unit,
    onCancel: () -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var photoCount by remember { mutableIntStateOf(0) }
    val capturedFiles = remember { mutableListOf<File>() }
    val executor = remember { Executors.newSingleThreadExecutor() }
    val mainHandler = remember { Handler(Looper.getMainLooper()) }
    var showFlash by remember { mutableStateOf(false) }
    var showDiscardDialog by remember { mutableStateOf(false) }
    val vibrator = remember { context.getSystemService<Vibrator>() }

    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    fun discardAndCancel() {
        capturedFiles.forEach { it.delete() }
        capturedFiles.clear()
        onCancel()
    }

    fun tryCancel() {
        if (photoCount > 0) showDiscardDialog = true else discardAndCancel()
    }

    // Botão voltar do sistema
    BackHandler { tryCancel() }

    LaunchedEffect(showFlash) {
        if (showFlash) {
            delay(80)
            showFlash = false
        }
    }

    DisposableEffect(Unit) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Size(1920, 1080),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .build()

            val preview = Preview.Builder()
                .setResolutionSelector(resolutionSelector)
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview
                )
            } catch (e: Exception) {
                Log.e("CameraScreen", "Erro ao abrir câmera", e)
            }
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            try { cameraProviderFuture.get().unbindAll() } catch (_: Exception) {}
            executor.shutdown()
        }
    }

    Box(Modifier.fillMaxSize()) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

        // Flash
        AnimatedVisibility(
            visible = showFlash,
            enter = fadeIn(),
            exit = fadeOut(),
            modifier = Modifier.fillMaxSize()
        ) {
            Box(Modifier.fillMaxSize().background(Color.White.copy(alpha = 0.6f)))
        }

        // Contador
        if (photoCount > 0) {
            Text(
                "$photoCount",
                color = Color.White,
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 52.dp)
                    .background(
                        MaterialTheme.colorScheme.primary.copy(alpha = 0.85f),
                        CircleShape
                    )
                    .padding(horizontal = 20.dp, vertical = 8.dp)
            )
        }

        // Barra inferior
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(
                    Color.Black.copy(alpha = 0.5f),
                    RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp)
                )
                .navigationBarsPadding()
                .padding(horizontal = 24.dp, vertical = 20.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedButton(
                onClick = { tryCancel() },
                colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.White)
            ) {
                Text("Cancelar")
            }

            // Captura
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .border(3.dp, Color.White, CircleShape)
                    .padding(5.dp)
            ) {
                Button(
                    onClick = {
                        val original = previewView.bitmap ?: return@Button
                        val bitmap = if (original.config == Bitmap.Config.HARDWARE) {
                            original.copy(Bitmap.Config.ARGB_8888, false).also { original.recycle() }
                        } else {
                            original
                        }

                        showFlash = true
                        try {
                            vibrator?.vibrate(
                                VibrationEffect.createOneShot(30, VibrationEffect.DEFAULT_AMPLITUDE)
                            )
                        } catch (_: Exception) {}

                        executor.execute {
                            try {
                                val file = File(context.cacheDir, "batch_${System.currentTimeMillis()}.jpg")
                                FileOutputStream(file).use { out ->
                                    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
                                }
                                bitmap.recycle()
                                capturedFiles.add(file)
                                mainHandler.post { photoCount = capturedFiles.size }
                            } catch (e: Exception) {
                                Log.e("CameraScreen", "Erro ao salvar foto", e)
                                bitmap.recycle()
                            }
                        }
                    },
                    modifier = Modifier.fillMaxSize().clip(CircleShape),
                    shape = CircleShape,
                    colors = ButtonDefaults.buttonColors(containerColor = Color.White)
                ) {}
            }

            FilledTonalButton(
                onClick = { onPhotosTaken(capturedFiles.toList()) },
                enabled = photoCount > 0
            ) {
                Text(
                    if (photoCount > 0) "Enviar ($photoCount)" else "Enviar",
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }

    // Dialog de confirmação ao cancelar
    if (showDiscardDialog) {
        AlertDialog(
            onDismissRequest = { showDiscardDialog = false },
            title = { Text("Descartar fotos?") },
            text = {
                Text("Você tirou $photoCount foto(s) que ainda não foram enviadas. Deseja descartar?")
            },
            confirmButton = {
                TextButton(onClick = {
                    showDiscardDialog = false
                    discardAndCancel()
                }) {
                    Text("Descartar", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDiscardDialog = false }) {
                    Text("Continuar tirando")
                }
            }
        )
    }
}
