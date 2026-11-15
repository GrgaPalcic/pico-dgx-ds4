//! `--traffic-file`: one JSON line per request, at its end (server.rs). Counts, times and flags only:
//! never text, token ids, the API key or a client address.
//!
//! One writer thread appends the lines, fed by a channel, so a request never waits on the file. The
//! file is opened `O_APPEND|O_CREAT`, mode 0600; on a write error the writer prints one line to stderr
//! and stops writing.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Sender};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The lines' writer: one thread, one file.
pub struct Traffic {
    tx: Option<Sender<String>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Traffic {
    /// Open `path` (created if missing, mode 0600) and start the writer.
    pub fn new(path: &Path) -> std::io::Result<Traffic> {
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = std::fs::OpenOptions::new().append(true).create(true).mode(0o600).open(path)?;
        let (tx, rx) = channel::<String>();
        let path: PathBuf = path.into();
        let handle = std::thread::Builder::new().name("traffic".into()).spawn(move || {
            let mut ok = true;
            while let Ok(line) = rx.recv() {
                if !ok {
                    continue; // one write error is enough: the writer stops writing
                }
                if let Err(e) = writeln!(file, "{line}") {
                    eprintln!("traffic: {}: {e}", path.display());
                    ok = false;
                }
            }
        })?;
        Ok(Traffic { tx: Some(tx), handle: Some(handle) })
    }

    /// Hand one request's line to the writer; the request never waits on the file.
    pub fn send(&self, line: String) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(line);
        }
    }
}

impl Drop for Traffic {
    fn drop(&mut self) {
        self.tx = None; // the writer drains what is left and returns
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// One request's record. Every field but the arrival is filled in as the request goes; a field that is
/// not knowable on the path the request took is written as null.
pub struct Rec {
    t: f64, // arrival, unix seconds
    t0: Instant,
    pub sess: Option<u64>,
    pub base: Option<u64>,
    pub prompt: Option<usize>,
    pub cached: Option<u32>,
    pub completion: Option<usize>,
    pub queue_ms: Option<f64>,
    pub prefill_ms: Option<f64>,
    pub decode_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub finish: Option<&'static str>,
    pub stream: Option<bool>,
    pub thinking: Option<bool>,
    pub images: Option<usize>,
    pub tools: Option<bool>,
}

impl Rec {
    pub fn new() -> Rec {
        Rec {
            t: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64(),
            t0: Instant::now(),
            sess: None,
            base: None,
            prompt: None,
            cached: None,
            completion: None,
            queue_ms: None,
            prefill_ms: None,
            decode_ms: None,
            ttft_ms: None,
            finish: None,
            stream: None,
            thinking: None,
            images: None,
            tools: None,
        }
    }

    /// The request's end: the line, with the total time measured now.
    pub fn line(&self) -> String {
        let sess = self.sess.map_or_else(|| "null".into(), |v| format!("\"{v:016x}\""));
        let base = self.base.filter(|&v| v != 0).map_or_else(|| "null".into(), |v| format!("\"{v:016x}\""));
        let n = |v: Option<usize>| v.map_or_else(|| "null".into(), |v| v.to_string());
        let c = |v: Option<u32>| v.map_or_else(|| "null".into(), |v| v.to_string());
        let ms = |v: Option<f64>| v.map_or_else(|| "null".into(), |v| format!("{v:.1}"));
        let b = |v: Option<bool>| v.map_or_else(|| "null".into(), |v| v.to_string());
        format!(
            "{{\"t\": {:.3}, \"sess\": {sess}, \"base\": {base}, \"prompt\": {}, \"cached\": {}, \"completion\": {}, \
             \"queue_ms\": {}, \"prefill_ms\": {}, \"decode_ms\": {}, \"ttft_ms\": {}, \"total_ms\": {:.1}, \
             \"finish\": \"{}\", \"stream\": {}, \"thinking\": {}, \"images\": {}, \"tools\": {}}}",
            self.t,
            n(self.prompt),
            c(self.cached),
            n(self.completion),
            ms(self.queue_ms),
            ms(self.prefill_ms),
            ms(self.decode_ms),
            ms(self.ttft_ms),
            self.t0.elapsed().as_secs_f64() * 1e3,
            self.finish.unwrap_or("error"),
            b(self.stream),
            b(self.thinking),
            n(self.images),
            b(self.tools),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn a_line_has_the_keys_in_order_and_no_text() {
        let dir = std::env::temp_dir().join(format!("dsfront-traffic-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("traffic.jsonl");
        let t = Traffic::new(&path).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        let mut r = Rec::new();
        r.sess = Some(0x2a);
        r.base = Some(0);
        r.prompt = Some(11);
        r.cached = Some(7);
        r.completion = Some(3);
        r.queue_ms = Some(0.2);
        r.prefill_ms = Some(12.34);
        r.decode_ms = Some(56.7);
        r.ttft_ms = Some(12.5);
        r.finish = Some("stop");
        r.stream = Some(true);
        r.thinking = Some(false);
        r.images = Some(0);
        r.tools = Some(false);
        t.send(r.line());
        let mut u = Rec::new();
        u.finish = Some("cancelled");
        t.send(u.line());
        drop(t); // the writer drains and joins
        let lines: Vec<String> = std::fs::read_to_string(&path).unwrap().lines().map(str::to_string).collect();
        assert_eq!(lines.len(), 2);
        let v: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(v["sess"].as_str(), Some("000000000000002a"));
        assert!(v["base"].is_null(), "session 0 names no base");
        assert_eq!(v["prompt"].as_u64(), Some(11));
        assert_eq!(v["cached"].as_u64(), Some(7));
        assert_eq!(v["completion"].as_u64(), Some(3));
        assert_eq!(v["queue_ms"].as_f64(), Some(0.2));
        assert_eq!(v["prefill_ms"].as_f64(), Some(12.3));
        assert_eq!(v["decode_ms"].as_f64(), Some(56.7));
        assert_eq!(v["ttft_ms"].as_f64(), Some(12.5));
        assert_eq!(v["finish"].as_str(), Some("stop"));
        assert_eq!(v["stream"].as_bool(), Some(true));
        assert_eq!(v["thinking"].as_bool(), Some(false));
        assert_eq!(v["images"].as_u64(), Some(0));
        assert_eq!(v["tools"].as_bool(), Some(false));
        assert!(v["t"].as_f64().unwrap() > 1.7e9 && v["total_ms"].as_f64().unwrap() >= 0.0);
        // the keys, in the order of the task's line
        let order = [
            "\"t\"", "\"sess\"", "\"base\"", "\"prompt\"", "\"cached\"", "\"completion\"", "\"queue_ms\"",
            "\"prefill_ms\"", "\"decode_ms\"", "\"ttft_ms\"", "\"total_ms\"", "\"finish\"", "\"stream\"",
            "\"thinking\"", "\"images\"", "\"tools\"",
        ];
        let mut at = 0;
        for k in order {
            at = lines[0][at..].find(k).expect("a key of the line") + at + k.len();
        }
        // a request whose session and Done never came: those fields are null, the line still there
        let w: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
        assert!(w["sess"].is_null() && w["base"].is_null() && w["prompt"].is_null() && w["cached"].is_null());
        assert!(w["queue_ms"].is_null() && w["ttft_ms"].is_null());
        assert_eq!(w["finish"].as_str(), Some("cancelled"));
        assert!(w["total_ms"].as_f64().unwrap() >= 0.0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
