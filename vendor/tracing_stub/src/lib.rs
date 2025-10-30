use std::sync::{Mutex, OnceLock};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogEntry {
    pub level: &'static str,
    pub message: String,
}

static LOGS: OnceLock<Mutex<Vec<LogEntry>>> = OnceLock::new();

fn log_store() -> &'static Mutex<Vec<LogEntry>> {
    LOGS.get_or_init(|| Mutex::new(Vec::new()))
}

pub fn take_logs() -> Vec<LogEntry> {
    log_store().lock().expect("log mutex poisoned").drain(..).collect()
}

pub fn __record(level: &'static str, message: String) {
    eprintln!("{message}");
    log_store()
        .lock()
        .expect("log mutex poisoned")
        .push(LogEntry { level, message });
}

#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {{
        $crate::__record("warn", format!($($arg)*));
    }};
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {{
        $crate::__record("error", format!($($arg)*));
    }};
}
