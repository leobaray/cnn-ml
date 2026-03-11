package com.lbwma.cnn.screen

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.lbwma.cnn.network.ApiClient
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConvertersScreen(onConversorClick: (String) -> Unit) {
    var conversores by remember { mutableStateOf<List<String>>(emptyList()) }
    var loading by remember { mutableStateOf(true) }
    var refreshing by remember { mutableStateOf(false) }
    var showDialog by remember { mutableStateOf(false) }
    var newName by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()
    val snackbar = remember { SnackbarHostState() }
    val scrollBehavior = TopAppBarDefaults.pinnedScrollBehavior()

    fun loadConversores(isRefresh: Boolean = false) {
        if (isRefresh) refreshing = true else loading = true
        scope.launch {
            ApiClient.getConversores()
                .onSuccess {
                    conversores = it
                    loading = false
                    refreshing = false
                }
                .onFailure {
                    loading = false
                    refreshing = false
                    snackbar.showSnackbar("Erro: ${it.message}")
                }
        }
    }

    LaunchedEffect(Unit) { loadConversores() }

    Scaffold(
        modifier = Modifier.nestedScroll(scrollBehavior.nestedScrollConnection),
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("Conversores", fontWeight = FontWeight.Bold)
                        if (conversores.isNotEmpty()) {
                            Text(
                                "${conversores.size} cadastrado(s)",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                },
                actions = {
                    IconButton(onClick = { loadConversores(isRefresh = true) }) {
                        Icon(Icons.Default.Refresh, "Atualizar")
                    }
                },
                scrollBehavior = scrollBehavior
            )
        },
        floatingActionButton = {
            FloatingActionButton(onClick = { showDialog = true }) {
                Icon(Icons.Default.Add, "Novo conversor")
            }
        },
        snackbarHost = { SnackbarHost(snackbar) }
    ) { padding ->
        Box(Modifier.fillMaxSize().padding(padding)) {
            when {
                loading -> CircularProgressIndicator(Modifier.align(Alignment.Center))
                conversores.isEmpty() -> {
                    Text(
                        "Nenhum conversor cadastrado.\nToque em + para criar.",
                        modifier = Modifier.align(Alignment.Center).padding(32.dp),
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = androidx.compose.ui.text.style.TextAlign.Center
                    )
                }
                else -> PullToRefreshBox(
                    isRefreshing = refreshing,
                    onRefresh = { loadConversores(isRefresh = true) },
                    modifier = Modifier.fillMaxSize()
                ) {
                    LazyColumn(Modifier.fillMaxSize()) {
                        items(conversores) { nome ->
                            ListItem(
                                headlineContent = {
                                    Text(nome, fontWeight = FontWeight.Medium)
                                },
                                modifier = Modifier.clickable { onConversorClick(nome) }
                            )
                            HorizontalDivider()
                        }
                    }
                }
            }
        }
    }

    if (showDialog) {
        AlertDialog(
            onDismissRequest = { showDialog = false; newName = "" },
            title = { Text("Novo Conversor") },
            text = {
                OutlinedTextField(
                    value = newName,
                    onValueChange = { newName = it },
                    label = { Text("Nome do conversor") },
                    singleLine = true
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        val name = newName.trim()
                        if (name.isNotEmpty()) {
                            showDialog = false
                            newName = ""
                            scope.launch {
                                ApiClient.createConversor(name)
                                    .onSuccess {
                                        loadConversores()
                                        snackbar.showSnackbar("\"$name\" criado")
                                    }
                                    .onFailure { snackbar.showSnackbar("Erro ao criar: ${it.message}") }
                            }
                        }
                    },
                    enabled = newName.isNotBlank()
                ) { Text("Criar") }
            },
            dismissButton = {
                TextButton(onClick = { showDialog = false; newName = "" }) { Text("Cancelar") }
            }
        )
    }
}
