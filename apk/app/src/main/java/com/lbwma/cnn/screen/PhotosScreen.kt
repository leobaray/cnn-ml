package com.lbwma.cnn.screen

import android.Manifest
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import coil.ImageLoader
import coil.compose.AsyncImage
import com.lbwma.cnn.network.ApiClient
import com.lbwma.cnn.network.Foto
import com.lbwma.cnn.network.ThumbnailCache
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhotosScreen(
    conversorName: String,
    imageLoader: ImageLoader,
    filesToUpload: List<File>,
    onFilesConsumed: () -> Unit,
    onOpenCamera: () -> Unit,
    onViewPhoto: (String) -> Unit,
    onBack: () -> Unit
) {
    var fotos by remember { mutableStateOf<List<Foto>>(emptyList()) }
    var loading by remember { mutableStateOf(true) }
    var refreshing by remember { mutableStateOf(false) }
    var pendingUploads by remember { mutableIntStateOf(0) }
    var showMenu by remember { mutableStateOf(false) }
    var deleteTarget by remember { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()
    val context = LocalContext.current
    val snackbar = remember { SnackbarHostState() }
    val scrollBehavior = TopAppBarDefaults.pinnedScrollBehavior()

    fun loadFotos(isRefresh: Boolean = false) {
        if (isRefresh) refreshing = true else loading = true
        scope.launch {
            ApiClient.getFotos(conversorName)
                .onSuccess { fotos = it; loading = false; refreshing = false }
                .onFailure {
                    loading = false; refreshing = false
                    snackbar.showSnackbar("Erro: ${it.message}")
                }
        }
    }

    // Gera thumbnail local a partir dos bytes antes de fazer upload
    suspend fun generateThumbnailFromBytes(fileName: String, bytes: ByteArray) = withContext(Dispatchers.IO) {
        try {
            val thumbFile = ThumbnailCache.getFile(conversorName, fileName)
            if (thumbFile.exists()) return@withContext

            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)

            val longerSide = maxOf(opts.outWidth, opts.outHeight)
            var sampleSize = 1
            while (longerSide / sampleSize > 600) sampleSize *= 2

            val decodeOpts = BitmapFactory.Options().apply { inSampleSize = sampleSize }
            val sampled = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, decodeOpts) ?: return@withContext

            val scale = 300f / maxOf(sampled.width, sampled.height)
            val w = (sampled.width * scale).toInt()
            val h = (sampled.height * scale).toInt()
            val thumb = Bitmap.createScaledBitmap(sampled, w, h, true)
            sampled.recycle()

            thumbFile.parentFile?.mkdirs()
            FileOutputStream(thumbFile).use { out ->
                thumb.compress(Bitmap.CompressFormat.JPEG, 70, out)
            }
            thumb.recycle()
        } catch (_: Exception) {}
    }

    fun uploadBytes(fileName: String, bytes: ByteArray) {
        pendingUploads++
        scope.launch {
            // Gera thumbnail local antes do upload
            generateThumbnailFromBytes(fileName, bytes)

            ApiClient.uploadFoto(conversorName, fileName, bytes)
                .onFailure { snackbar.showSnackbar("Erro ao enviar $fileName") }
            pendingUploads--
            if (pendingUploads == 0) {
                loadFotos()
                snackbar.showSnackbar("Upload concluído")
            }
        }
    }

    // Upload de arquivos vindos da câmera
    LaunchedEffect(filesToUpload) {
        if (filesToUpload.isNotEmpty()) {
            snackbar.showSnackbar("Enviando ${filesToUpload.size} foto(s)...")
            filesToUpload.forEach { file ->
                val bytes = file.readBytes()
                uploadBytes(file.name, bytes)
                file.delete()
            }
            onFilesConsumed()
        }
    }

    val galleryLauncher = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let {
            scope.launch {
                try {
                    val bytes = context.contentResolver.openInputStream(it)?.readBytes()
                    if (bytes != null) {
                        val fileName = "foto_${System.currentTimeMillis()}.jpg"
                        uploadBytes(fileName, bytes)
                    }
                } catch (_: Exception) {}
            }
        }
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) onOpenCamera()
        else scope.launch { snackbar.showSnackbar("Permissão de câmera negada") }
    }

    LaunchedEffect(Unit) { loadFotos() }

    Scaffold(
        modifier = Modifier.nestedScroll(scrollBehavior.nestedScrollConnection),
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(conversorName, fontWeight = FontWeight.Bold)
                        val subtitle = when {
                            pendingUploads > 0 -> "Enviando $pendingUploads foto(s)..."
                            fotos.isNotEmpty() -> "${fotos.size} foto(s)"
                            else -> null
                        }
                        if (subtitle != null) {
                            Text(
                                subtitle,
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, "Voltar")
                    }
                },
                actions = {
                    IconButton(onClick = { loadFotos(isRefresh = true) }) {
                        Icon(Icons.Default.Refresh, "Atualizar")
                    }
                },
                scrollBehavior = scrollBehavior
            )
        },
        floatingActionButton = {
            Box {
                FloatingActionButton(onClick = { showMenu = true }) {
                    Icon(Icons.Default.Add, "Adicionar foto")
                }
                DropdownMenu(expanded = showMenu, onDismissRequest = { showMenu = false }) {
                    DropdownMenuItem(
                        text = { Text("Tirar fotos") },
                        onClick = {
                            showMenu = false
                            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Escolher da galeria") },
                        onClick = {
                            showMenu = false
                            galleryLauncher.launch("image/*")
                        }
                    )
                }
            }
        },
        snackbarHost = { SnackbarHost(snackbar) }
    ) { padding ->
        Box(Modifier.fillMaxSize().padding(padding)) {
            if (pendingUploads > 0) {
                LinearProgressIndicator(
                    Modifier.fillMaxWidth().align(Alignment.TopCenter)
                )
            }
            when {
                loading -> CircularProgressIndicator(Modifier.align(Alignment.Center))
                fotos.isEmpty() -> {
                    Text(
                        "Nenhuma foto ainda.\nToque em + para adicionar.",
                        modifier = Modifier.align(Alignment.Center).padding(32.dp),
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center
                    )
                }
                else -> PullToRefreshBox(
                    isRefreshing = refreshing,
                    onRefresh = { loadFotos(isRefresh = true) },
                    modifier = Modifier.fillMaxSize()
                ) {
                    LazyVerticalGrid(
                        columns = GridCells.Adaptive(110.dp),
                        contentPadding = PaddingValues(8.dp),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp),
                        modifier = Modifier.fillMaxSize()
                    ) {
                        items(fotos, key = { it.nome }) { foto ->
                            val thumbFile = remember(foto.nome) {
                                ThumbnailCache.getFile(conversorName, foto.nome)
                            }
                            var thumbReady by rememberSaveable(foto.nome) {
                                mutableStateOf(thumbFile.exists())
                            }

                            LaunchedEffect(foto.nome) {
                                if (!thumbReady) {
                                    if (ThumbnailCache.generate(conversorName, foto.nome)) {
                                        thumbReady = true
                                    }
                                }
                            }

                            Card(
                                modifier = Modifier
                                    .aspectRatio(1f)
                                    .clickable { onViewPhoto(foto.nome) },
                                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
                            ) {
                                Box(Modifier.fillMaxSize()) {
                                    if (thumbReady) {
                                        AsyncImage(
                                            model = thumbFile,
                                            contentDescription = foto.nome,
                                            contentScale = ContentScale.Crop,
                                            modifier = Modifier.fillMaxSize()
                                        )
                                    } else {
                                        CircularProgressIndicator(
                                            modifier = Modifier.align(Alignment.Center),
                                            strokeWidth = 2.dp
                                        )
                                    }
                                    IconButton(
                                        onClick = { deleteTarget = foto.nome },
                                        modifier = Modifier.align(Alignment.TopEnd),
                                        colors = IconButtonDefaults.iconButtonColors(
                                            containerColor = Color.Black.copy(alpha = 0.4f)
                                        )
                                    ) {
                                        Icon(
                                            Icons.Default.Delete,
                                            "Deletar",
                                            tint = Color.White
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    deleteTarget?.let { arquivo ->
        AlertDialog(
            onDismissRequest = { deleteTarget = null },
            title = { Text("Deletar foto") },
            text = { Text("Tem certeza que deseja deletar $arquivo?") },
            confirmButton = {
                TextButton(onClick = {
                    val name = arquivo
                    deleteTarget = null
                    scope.launch {
                        ApiClient.deleteFoto(conversorName, name)
                            .onSuccess {
                                ThumbnailCache.getFile(conversorName, name).delete()
                                loadFotos()
                                snackbar.showSnackbar("\"$name\" deletada")
                            }
                            .onFailure { snackbar.showSnackbar("Erro ao deletar") }
                    }
                }) { Text("Deletar", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = {
                TextButton(onClick = { deleteTarget = null }) { Text("Cancelar") }
            }
        )
    }
}
