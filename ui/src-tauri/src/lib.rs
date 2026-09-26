use tauri::{
    image::Image,
    menu::{CheckMenuItem, Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent},
    AppHandle, Emitter, Manager, Runtime, WindowEvent, Wry,
};

/// Handles kept so the tray can be updated from commands later.
struct TrayHandles {
    tray: TrayIcon<Wry>,
    gaming: CheckMenuItem<Wry>,
}

/// A 32x32 filled circle in the colour for `state`, built at runtime so the
/// UI ships no per-state image assets.
fn state_icon(state: &str) -> Image<'static> {
    let (r, g, b) = match state {
        "listening" => (34u8, 197u8, 94u8),
        "thinking" => (245u8, 158u8, 11u8),
        "error" => (239u8, 68u8, 68u8),
        _ => (148u8, 163u8, 184u8), // idle / unknown
    };

    let size = 32u32;
    let mut rgba = vec![0u8; (size * size * 4) as usize];
    let center = (size as f32 - 1.0) / 2.0;
    let radius = size as f32 * 0.42;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let distance = (dx * dx + dy * dy).sqrt();
            let index = ((y * size + x) * 4) as usize;
            if distance <= radius {
                rgba[index] = r;
                rgba[index + 1] = g;
                rgba[index + 2] = b;
                rgba[index + 3] = 255;
            } else if distance <= radius + 1.0 {
                // One-pixel feather so the dot does not look jagged.
                let alpha = ((radius + 1.0 - distance) * 255.0) as u8;
                rgba[index] = r;
                rgba[index + 1] = g;
                rgba[index + 2] = b;
                rgba[index + 3] = alpha;
            }
        }
    }
    Image::new_owned(rgba, size, size)
}

/// Bring the main window back from the tray (or from minimised).
fn show_main_window<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

#[tauri::command]
fn set_atlas_state(app: AppHandle, state: String) -> Result<(), String> {
    let handles = app.state::<TrayHandles>();
    handles
        .tray
        .set_icon(Some(state_icon(&state)))
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn set_gaming_mode(app: AppHandle, enabled: bool) -> Result<(), String> {
    let handles = app.state::<TrayHandles>();
    handles
        .gaming
        .set_checked(enabled)
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn quit_app(app: AppHandle) {
    app.exit(0);
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // --- system tray -------------------------------------------------
            let open_item = MenuItem::with_id(app, "open", "Open Atlas", true, None::<&str>)?;
            let gaming_item =
                CheckMenuItem::with_id(app, "gaming", "Gaming Mode", true, false, None::<&str>)?;
            let shutdown_item =
                MenuItem::with_id(app, "shutdown", "Shutdown Atlas", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open_item, &gaming_item, &shutdown_item])?;

            let tray = TrayIconBuilder::new()
                .icon(state_icon("idle"))
                .tooltip("Atlas")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "open" => show_main_window(app),
                    // The frontend owns the API call; the tray only asks.
                    "gaming" => {
                        let _ = app.emit("tray://toggle-gaming", ());
                    }
                    "shutdown" => {
                        let _ = app.emit("tray://shutdown", ());
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        show_main_window(tray.app_handle());
                    }
                })
                .build(app)?;

            // Keep the handles so commands can recolour the icon and tick the
            // gaming item as the frontend reports state changes.
            app.manage(TrayHandles {
                tray,
                gaming: gaming_item,
            });

            Ok(())
        })
        // Closing the window hides Atlas to the tray instead of quitting.
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .invoke_handler(tauri::generate_handler![
            set_atlas_state,
            set_gaming_mode,
            quit_app
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
