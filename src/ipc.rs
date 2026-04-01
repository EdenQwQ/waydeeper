use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcRequest {
    pub command: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcResponse {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub type CommandHandler =
    Arc<dyn Fn(&str, &serde_json::Value) -> Result<serde_json::Value> + Send + Sync>;

pub struct DaemonSocket {
    monitor: String,
    socket_path: PathBuf,
    running: Arc<AtomicBool>,
}

impl DaemonSocket {
    pub fn new(monitor: &str) -> Result<Self> {
        let runtime_dir = config::runtime_dir()?;
        let socket_path = runtime_dir.join(format!("{}.sock", monitor));

        Ok(Self {
            monitor: monitor.to_string(),
            socket_path,
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn start(&mut self, handler: CommandHandler) -> Result<()> {
        let runtime_dir = config::runtime_dir()?;
        std::fs::create_dir_all(&runtime_dir)?;

        if self.socket_path.exists() {
            match UnixStream::connect(&self.socket_path) {
                Ok(_) => {
                    return Err(anyhow!(
                        "Another daemon already running for monitor {}",
                        self.monitor
                    ));
                }
                Err(_) => {
                    let _ = std::fs::remove_file(&self.socket_path);
                }
            }
        }

        let listener = UnixListener::bind(&self.socket_path)
            .with_context(|| format!("Failed to bind socket at {:?}", self.socket_path))?;

        listener
            .set_nonblocking(true)
            .context("Failed to set socket non-blocking")?;

        self.running.store(true, Ordering::SeqCst);
        let running = self.running.clone();

        thread::spawn(move || {
            while running.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let handler = handler.clone();
                        thread::spawn(move || {
                            let _ = Self::handle_connection(&mut stream, &handler);
                        });
                    }
                    Err(ref error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(50));
                        continue;
                    }
                    Err(_) => break,
                }
            }
        });

        log::info!("IPC socket listening at {:?}", self.socket_path);
        Ok(())
    }

    fn handle_connection(
        stream: &mut UnixStream,
        handler: &CommandHandler,
    ) -> Result<()> {
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;

        let mut data = Vec::new();
        let mut buffer = [0u8; 4096];

        loop {
            match stream.read(&mut buffer) {
                Ok(0) => break,
                Ok(bytes_read) => {
                    data.extend_from_slice(&buffer[..bytes_read]);
                    if data.contains(&b'\n') {
                        break;
                    }
                }
                Err(ref error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if !data.is_empty() && data.contains(&b'\n') {
                        break;
                    }
                    thread::sleep(Duration::from_millis(10));
                    continue;
                }
                Err(error) => return Err(error.into()),
            }
        }

        if data.is_empty() {
            return Ok(());
        }

        let response = match serde_json::from_slice::<IpcRequest>(&data) {
            Ok(request) => match handler(&request.command, &request.params) {
                Ok(result) => IpcResponse {
                    success: true,
                    result: Some(result),
                    error: None,
                },
                Err(error) => IpcResponse {
                    success: false,
                    result: None,
                    error: Some(error.to_string()),
                },
            },
            Err(error) => IpcResponse {
                success: false,
                result: None,
                error: Some(format!("Invalid JSON: {}", error)),
            },
        };

        let response_data = serde_json::to_vec(&response)?;
        stream.write_all(&response_data)?;
        stream.write_all(b"\n")?;
        stream.flush()?;
        Ok(())
    }

    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if self.socket_path.exists() {
            let _ = std::fs::remove_file(&self.socket_path);
        }
        log::info!("IPC socket stopped");
    }
}

impl Drop for DaemonSocket {
    fn drop(&mut self) {
        self.stop();
    }
}

pub struct DaemonClient {
    monitor: String,
    socket_path: PathBuf,
}

impl DaemonClient {
    pub fn new(monitor: &str) -> Result<Self> {
        let runtime_dir = config::runtime_dir()?;
        let socket_path = runtime_dir.join(format!("{}.sock", monitor));

        Ok(Self {
            monitor: monitor.to_string(),
            socket_path,
        })
    }

    pub fn is_running(&self) -> bool {
        if !self.socket_path.exists() {
            return false;
        }
        match self.send_command("PING", serde_json::Value::Null, Duration::from_secs(2)) {
            Ok(response) => response.success,
            Err(_) => false,
        }
    }

    pub fn send_command(
        &self,
        command: &str,
        params: serde_json::Value,
        timeout: Duration,
    ) -> Result<IpcResponse> {
        if !self.socket_path.exists() {
            return Err(anyhow!("Daemon not running for monitor {}", self.monitor));
        }

        let mut stream = UnixStream::connect(&self.socket_path)?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;

        let request = IpcRequest {
            command: command.to_string(),
            params,
        };

        let request_data = serde_json::to_vec(&request)?;
        stream.write_all(&request_data)?;
        stream.write_all(b"\n")?;
        stream.flush()?;

        let mut data = Vec::new();
        let mut buffer = [0u8; 4096];

        loop {
            match stream.read(&mut buffer) {
                Ok(0) => break,
                Ok(bytes_read) => {
                    data.extend_from_slice(&buffer[..bytes_read]);
                    if data.contains(&b'\n') {
                        break;
                    }
                }
                Err(ref error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error.into()),
            }
        }

        let response: IpcResponse = serde_json::from_slice(&data)?;
        Ok(response)
    }

    pub fn stop_daemon(&self) -> Result<bool> {
        let response =
            self.send_command("STOP", serde_json::Value::Null, Duration::from_secs(5))?;
        Ok(response.success)
    }
}

pub fn list_running_daemons() -> Result<Vec<String>> {
    let runtime_dir = match config::runtime_dir() {
        Ok(dir) => dir,
        Err(_) => return Ok(Vec::new()),
    };

    if !runtime_dir.exists() {
        return Ok(Vec::new());
    }

    let mut running = Vec::new();

    for entry in std::fs::read_dir(&runtime_dir)?.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "sock") {
            if let Some(stem) = path.file_stem() {
                let monitor = stem.to_string_lossy().to_string();
                if let Ok(client) = DaemonClient::new(&monitor) {
                    if client.is_running() {
                        running.push(monitor);
                    } else {
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }
        }
    }

    Ok(running)
}

pub fn stop_daemon(monitor: &str, timeout: Duration) -> Result<bool> {
    let client = DaemonClient::new(monitor)?;
    if !client.is_running() {
        return Ok(false);
    }

    if client.stop_daemon()? {
        thread::sleep(Duration::from_millis(500));
        let remaining = timeout.saturating_sub(Duration::from_millis(500));
        let steps = (remaining.as_millis() / 100) as u32;
        for _ in 0..steps {
            if !client.is_running() {
                return Ok(true);
            }
            thread::sleep(Duration::from_millis(100));
        }
    }

    Ok(false)
}

pub fn stop_all_daemons(
    timeout: Duration,
) -> Result<std::collections::HashMap<String, bool>> {
    let mut results = std::collections::HashMap::new();
    for monitor in list_running_daemons()? {
        results.insert(
            monitor.clone(),
            stop_daemon(&monitor, timeout).unwrap_or(false),
        );
    }
    Ok(results)
}
