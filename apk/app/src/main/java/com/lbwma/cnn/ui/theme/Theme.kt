package com.lbwma.cnn.ui.theme

import android.app.Activity
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val CnnColorScheme = darkColorScheme(
    primary = Cyan40,
    onPrimary = Color.Black,
    primaryContainer = CyanDark,
    onPrimaryContainer = Cyan80,
    secondary = Success,
    onSecondary = Color.Black,
    background = Dark00,
    onBackground = TextPrimary,
    surface = Dark05,
    onSurface = TextPrimary,
    surfaceVariant = Dark15,
    onSurfaceVariant = TextSecondary,
    surfaceContainerLowest = Dark00,
    surfaceContainerLow = Dark05,
    surfaceContainer = Dark10,
    surfaceContainerHigh = Dark15,
    surfaceContainerHighest = Dark20,
    outline = Dark30,
    outlineVariant = Dark20,
    error = Error,
    onError = Color.White,
    errorContainer = ErrorDark,
    onErrorContainer = Color(0xFFFFDAD6)
)

@Composable
fun CnnTheme(content: @Composable () -> Unit) {
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = Dark00.toArgb()
            window.navigationBarColor = Dark00.toArgb()
            WindowCompat.getInsetsController(window, view).apply {
                isAppearanceLightStatusBars = false
                isAppearanceLightNavigationBars = false
            }
        }
    }

    MaterialTheme(
        colorScheme = CnnColorScheme,
        typography = Typography,
        content = content
    )
}
