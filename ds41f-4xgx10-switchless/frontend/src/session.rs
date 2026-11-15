//! Session matching for exact prefix reuse (docs/protocol.md, "Sessions and exact prefix reuse").
//!
//! Clients do not name their conversation, so a request is matched by text: the session whose text
//! (prompt plus generated output) shares the longest prefix with the new prompt. The shared part is
//! sent as that session's own token ids, which is what is in the engine's KV state; the rest is
//! tokenised on its own from a cut next to an added token, where tokenisation restarts anyway.
//!
//! One file per session (`--table-dir`, `<id 16 hex>.t`), written by a thread of its own after every
//! finished turn, so a stop, a watchdog restart or a power cut loses at most the turn in flight
//! (docs/engine.md, "The disk tier"): the engine writes each session's state through to its own copy,
//! and this table names the session that copy belongs to. Each file holds the conversation's text, so
//! the directory is ours alone (mode 0700, its files 0600) and a file is written whole or not at all.

use crate::tokenizer::Tokenizer;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Bytes of shared text below which a match is not worth a session (a BOS and a few tokens).
const MIN_SHARED: usize = 64;
/// A request continues a session in place when the shared text reaches this close to the end of
/// the session's prompt; the encoding may rewrite the tail of the prompt (`<think>` → `</think>`).
const PROMPT_SLACK: usize = 16;

/// Vision (docs/vision.md): one image span of a session's prompt. Its placeholder starts at `byte` in
/// the session's text, its ids start at `at` and run `rows` deep, and all of them are the image id.
/// `hash` is the encoder's: two different images can tokenise to the same ids, so the key carries it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImgKey {
    pub byte: usize,
    pub at: u32,
    pub rows: u32,
    pub hash: u64,
}

struct Session {
    text: String,
    ids: Vec<u32>,
    /// ends[i] = byte length of the text of ids[..=i]; a span's ids all share the placeholder's end.
    /// u32: half the memory per token, and a session's text is bounded by u32::MAX bytes (finish).
    ends: Vec<u32>,
    /// vision: its image spans, in prompt order; empty for a text session
    imgs: Vec<ImgKey>,
    prompt_len: usize,
    busy: bool,
    last_used: u64,
    /// Unix time of its last request: a saved table leaves out what is older than its limit.
    t_used: u64,
}

pub struct Plan {
    pub session_id: u64,
    /// The session a new session may fork from (0 = none).
    pub base_session_id: u64,
    /// Leading ids taken from a session (the rest is tokenised).
    pub reused: usize,
    /// vision: the `at` of every session span the reused ids hold, in prompt order (empty for text).
    pub img_reused: Vec<u32>,
    pub ids: Vec<u32>,
    /// For the log, counts only: the session whose text shares the most with the prompt (0: none
    /// shares any), the bytes it shares, and the ids its history has in common with `ids`.
    pub matched: u64,
    pub shared: usize,
    pub common: usize,
}

impl Plan {
    /// Where the request starts, for the log (no text, no ids): a continuation of a session, a
    /// fork of one (the engine starts from it if it holds it), or a new session that names none.
    pub fn describe(&self, prompt_bytes: usize) -> String {
        let of = format!("{} of {prompt_bytes} bytes, {} ids in common", self.shared, self.common);
        if self.base_session_id != 0 {
            format!("session {}, a fork of {}: it shares {of}", self.session_id, self.base_session_id)
        } else if self.matched == self.session_id {
            format!("session {} in place: it shares {of}", self.session_id)
        } else if self.matched != 0 {
            format!("session {}, new: the best match, session {}, shares {} bytes (under {MIN_SHARED})", self.session_id, self.matched, self.shared)
        } else {
            format!("session {}, new: no session shares a byte", self.session_id)
        }
    }
}

pub struct SessionTable {
    sessions: HashMap<u64, Session>,
    next_id: u64,
    clock: u64,
    max_sessions: usize,
    max_tokens: usize,
    /// Where each session's file goes (None: nothing is written or read).
    dir: Option<PathBuf>,
    /// The writer's queue, and its thread: one write per finished turn, one removal per session that
    /// goes. Dropping the sender lets the writer finish what is queued and stop.
    tx: Option<mpsc::Sender<Job>>,
    th: Option<std::thread::JoinHandle<()>>,
}

enum Job {
    Write(u64, Vec<u8>),
    Remove(u64),
    Drain(mpsc::Sender<()>),
}

/// The session files' writer: one thread, in order, so that a removal queued before a write of the same
/// session cannot land after it. A write goes under a temporary name, is synced, then renamed into
/// place; a barrier answers once every job before it is done.
fn writer(dir: PathBuf, rx: mpsc::Receiver<Job>) {
    for job in rx {
        match job {
            Job::Write(id, bytes) => {
                if let Err(e) = write_file(&dir, id, &bytes) {
                    eprintln!("{e}");
                }
            }
            Job::Remove(id) => {
                let _ = std::fs::remove_file(session_path(&dir, id));
            }
            Job::Drain(ack) => {
                let _ = ack.send(());
            }
        }
    }
}

fn write_file(dir: &Path, id: u64, bytes: &[u8]) -> Result<(), String> {
    private_dir(dir)?;
    let path = session_path(dir, id);
    let tmp = PathBuf::from(format!("{}.tmp", path.display()));
    let _ = std::fs::remove_file(&tmp);
    let write = || -> std::io::Result<()> {
        let mut o = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&tmp)?;
        o.write_all(bytes)?;
        o.sync_all()?;
        drop(o);
        std::fs::rename(&tmp, &path)?;
        if let Ok(d) = std::fs::File::open(dir) {
            d.sync_all()?;
        }
        Ok(())
    };
    write().map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", path.display())
    })
}

/// One session's file: the magic, its written time, its body's length and hash, then its body (the id,
/// its clocks, its text, its ids and its image keys, as the whole-table file held them).
fn session_bytes(id: u64, s: &Session) -> Vec<u8> {
    let mut body = Vec::new();
    for v in [id, s.last_used, s.t_used, s.prompt_len as u64, s.text.len() as u64] {
        body.extend(v.to_le_bytes());
    }
    body.extend(s.text.as_bytes());
    body.extend((s.ids.len() as u64).to_le_bytes());
    for &x in &s.ids {
        body.extend(x.to_le_bytes());
    }
    body.extend((s.imgs.len() as u64).to_le_bytes());
    for i in &s.imgs {
        for v in [i.byte as u64, i.at as u64, i.rows as u64, i.hash] {
            body.extend(v.to_le_bytes());
        }
    }
    let mut f = TABLE_MAGIC.to_vec();
    for v in [unix_now(), body.len() as u64, fnv(&body)] {
        f.extend(v.to_le_bytes());
    }
    f.extend(body);
    f
}

impl SessionTable {
    /// Holds up to `max_sessions` sessions and `max_tokens` ids in total, each written to `dir` (made
    /// 0700; None: nothing is written). Ids start from the clock, so that a table begun afresh never
    /// reuses an earlier one's ids, whose sessions the engine may still hold.
    pub fn new(max_sessions: usize, max_tokens: usize, dir: Option<PathBuf>) -> Self {
        let (tx, th) = match &dir {
            None => (None, None),
            Some(d) => {
                let (tx, rx) = mpsc::channel::<Job>();
                let d = d.clone();
                (Some(tx), Some(std::thread::spawn(move || writer(d, rx))))
            }
        };
        SessionTable { sessions: HashMap::new(), next_id: (unix_now() << 20).max(1), clock: 0, max_sessions, max_tokens, dir, tx, th }
    }

    /// Until every queued write and removal is done: a shutdown, not a stop (each turn is already
    /// written when it ends).
    pub fn drain(&self) {
        if let Some(tx) = &self.tx {
            let (ack, rx) = mpsc::channel();
            if tx.send(Job::Drain(ack)).is_ok() {
                let _ = rx.recv();
            }
        }
    }

    fn forget(&self, id: u64) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Job::Remove(id));
        }
    }
}

impl Drop for SessionTable {
    fn drop(&mut self) {
        self.tx = None;  // the writer takes what is queued and returns
        if let Some(th) = self.th.take() {
            let _ = th.join();
        }
    }
}

fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + 64 <= n && a[i..i + 64] == b[i..i + 64] {
        i += 64;
    }
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Vision: the shared byte count clipped to whole image spans. The session's spans and the request's
/// are compared in prompt order; a prefix that would pair two different images (a byte, a row count or
/// a hash apart), or that holds an image on one side only, is cut back to the earlier placeholder.
fn clip_images(a: &[ImgKey], b: &[ImgKey], n: usize) -> usize {
    let mut k = 0;
    loop {
        let x = a.get(k).filter(|i| i.byte < n);
        let y = b.get(k).filter(|i| i.byte < n);
        match (x, y) {
            (None, None) => return n,
            (Some(x), Some(y)) if (x.byte, x.rows, x.hash) == (y.byte, y.rows, y.hash) => k += 1,
            (Some(x), y) => return x.byte.min(y.map_or(usize::MAX, |y| y.byte)),
            (None, Some(y)) => return y.byte,
        }
    }
}

/// Vision: the ends of a session's ids. Every image span's rows share the placeholder's end, so a cut
/// lands on the span's edge, never inside it; every other id adds its own bytes.
fn ends_of(ids: &[u32], imgs: &[ImgKey], tok: &Tokenizer) -> Vec<u32> {
    let mut ends = Vec::with_capacity(ids.len());
    let mut end = 0u32;
    let mut i = 0usize;
    for im in imgs {
        let at = im.at as usize;
        if at < i || at + im.rows as usize > ids.len() {
            continue; // a record that does not fit its ids: leave them as ordinary ids
        }
        while i < at {
            end += tok.token_bytes(ids[i]).len() as u32;
            ends.push(end);
            i += 1;
        }
        end += tok.token_bytes(crate::vision::IMAGE_TOKEN_ID).len() as u32;
        for _ in 0..im.rows {
            ends.push(end);
        }
        i += im.rows as usize;
    }
    while i < ids.len() {
        end += tok.token_bytes(ids[i]).len() as u32;
        ends.push(end);
        i += 1;
    }
    ends
}

/// Vision: the text a session's ids spell, each image span's rows as its one placeholder. The check
/// that a loaded table belongs to this tokenizer reads it, as the ids of a span do not spell its text.
fn spell(ids: &[u32], imgs: &[ImgKey], tok: &Tokenizer) -> Vec<u8> {
    let mut v = Vec::new();
    let mut i = 0usize;
    for im in imgs {
        let at = im.at as usize;
        if at < i || at + im.rows as usize > ids.len() {
            continue; // a record that does not fit its ids: leave them as ordinary ids
        }
        v.extend_from_slice(&tok.decode_bytes(&ids[i..at]));
        v.extend_from_slice(tok.token_bytes(crate::vision::IMAGE_TOKEN_ID));
        i = at + im.rows as usize;
    }
    v.extend_from_slice(&tok.decode_bytes(&ids[i..]));
    v
}

/// One session's file: the magic, then its written time, its body's length and its hash, then the body
/// (today's per-record bytes: the id, its clocks, its text, its ids and its image keys).
const TABLE_MAGIC: &[u8; 8] = b"DSFT0003";

/// <dir>/<id 16 hex>.t
fn session_path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("{id:016x}.t"))
}

/// Whether `name` is one of ours (a directory scan ignores anything else).
fn session_file(name: &str) -> Option<u64> {
    let stem = name.strip_suffix(".t")?;
    if stem.len() != 16 || !stem.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(stem, 16).ok()
}

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(1469598103934665603u64, |h, &x| (h ^ x as u64).wrapping_mul(1099511628211))
}

/// The directory is ours and nobody else's (made 0700 if it was ours but open to others).
fn private_dir(dir: &Path) -> Result<(), String> {
    let me = std::fs::metadata("/proc/self").map_err(|e| format!("/proc/self: {e}"))?.uid();
    let m = std::fs::symlink_metadata(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if !m.is_dir() || m.uid() != me {
        return Err(format!("{} is not a directory of ours", dir.display()));
    }
    let abs = std::fs::canonicalize(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if abs.ancestors().any(|a| a.join(".git").exists()) {
        return Err(format!("{} is in a git tree, where no conversation goes", dir.display()));
    }
    if m.mode() & 0o077 != 0 {
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    Ok(())
}

struct Bytes<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Bytes<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.b.len() - self.at < n {
            return Err("cut short".into());
        }
        self.at += n;
        Ok(&self.b[self.at - n..self.at])
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

/// One session's file read and checked: its magic, its age, its body's hash, its text (UTF-8, the ids
/// spelling it with this tokenizer) and its image keys. Err holds why it is not taken (the file goes).
fn read_session_file(path: &Path, id: u64, tok: &Tokenizer, max_age_s: u64) -> Result<Session, String> {
    let me = std::fs::metadata("/proc/self").map_err(|e| e.to_string())?.uid();
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let m = file.metadata().map_err(|e| e.to_string())?;
    if !m.is_file() || m.uid() != me || m.mode() & 0o777 != 0o600 {
        return Err(format!("{} is not a file of ours of mode 0600", path.display()));
    }
    let mut f = Vec::new();
    file.read_to_end(&mut f).map_err(|e| e.to_string())?;
    if f.len() < 32 || &f[..8] != TABLE_MAGIC {
        return Err("not a saved session".into());
    }
    let mut h = Bytes { b: &f[8..32], at: 0 };
    let (written, body_len, hash) = (h.u64()?, h.u64()?, h.u64()?);
    let now = unix_now();
    if now.saturating_sub(written) > max_age_s {
        return Err(format!("saved {} s ago", now.saturating_sub(written)));
    }
    let body = &f[32..];
    if body.len() as u64 != body_len || fnv(body) != hash {
        return Err("incomplete or damaged".into());
    }
    let mut r = Bytes { b: body, at: 0 };
    let (id_in, last_used, t_used, prompt_len, text_len) =
        (r.u64()?, r.u64()?, r.u64()?, r.u64()? as usize, r.u64()? as usize);
    if id_in != id {
        return Err("named after another session".into());
    }
    let text = r.take(text_len)?;
    let n_ids = r.u64()? as usize;
    let raw = r.take(n_ids.checked_mul(4).ok_or("a bad count")?)?;
    let ids: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let n_imgs = r.u64()? as usize;
    let raw = r.take(n_imgs.checked_mul(32).ok_or("a bad count")?)?;
    let imgs: Vec<ImgKey> = raw
        .chunks_exact(32)
        .map(|c| {
            let v = |k: usize| u64::from_le_bytes(c[k * 8..k * 8 + 8].try_into().unwrap());
            ImgKey { byte: v(0) as usize, at: v(1) as u32, rows: v(2) as u32, hash: v(3) }
        })
        .collect();
    let Ok(text) = std::str::from_utf8(text) else { return Err("its text is not UTF-8".into()) };
    let vocab = tok.vocab_size() as u32;
    let bad_imgs = imgs.iter().any(|i| {
        i.rows == 0 || i.at as usize + i.rows as usize > ids.len() || i.byte > text.len()
    }) || imgs.windows(2).any(|w| w[0].at >= w[1].at || w[0].byte >= w[1].byte);
    if now.saturating_sub(t_used) > max_age_s || prompt_len > text.len() || ids.iter().any(|&x| x >= vocab) || bad_imgs {
        return Err("unused for longer than the limit, or another tokenizer's ids".into());
    }
    if text.len() > u32::MAX as usize {
        return Err("a session this size is not stored".into()); // ends are u32
    }
    let ends = ends_of(&ids, &imgs, tok);
    if ends.last().copied().unwrap_or(0) as usize != text.len() || spell(&ids, &imgs, tok) != text.as_bytes() {
        return Err("another tokenizer's ids".into());
    }
    if r.at != body.len() {
        return Err("a bad body".into());
    }
    Ok(Session { text: text.to_string(), ids, ends, imgs, prompt_len, busy: false, last_used, t_used })
}

impl SessionTable {
    /// The sessions an earlier run left in the directory: every file that passes the checks (whole, of
    /// this tokenizer, young enough) comes back. `*.tmp` goes, and so does every file that fails.
    /// Returns how many sessions came back; Err holds the reason ("" when there is no directory).
    pub fn load(&mut self, tok: &Tokenizer, max_age_s: u64) -> Result<usize, String> {
        let Some(dir) = self.dir.clone() else { return Ok(0) };
        let md = match std::fs::symlink_metadata(&dir) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(format!("{}: {e}", dir.display())),
            Ok(m) => m,
        };
        if !md.is_dir() {
            return Err(format!("{} is not a directory", dir.display()));
        }
        private_dir(&dir)?;
        let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let mut got = HashMap::new();
        let mut dropped = 0usize;
        for e in entries {
            let Ok(e) = e else { continue };
            let name = e.file_name();
            let Some(name) = name.to_str() else { continue };
            let path = e.path();
            if name.ends_with(".tmp") {
                let _ = std::fs::remove_file(&path);
                continue;
            }
            let Some(id) = session_file(name) else { continue };
            match read_session_file(&path, id, tok, max_age_s) {
                Ok(s) => {
                    got.insert(id, s);
                }
                Err(why) => {
                    let _ = std::fs::remove_file(&path);
                    dropped += 1;
                    if !why.is_empty() {
                        eprintln!("session file: {why}");
                    }
                }
            }
        }
        let _ = dropped;
        let k = got.len();
        self.clock = got.values().map(|s| s.last_used).max().unwrap_or(0).max(self.clock);
        self.next_id = self.next_id.max(got.keys().max().map_or(0, |m| m + 1));
        self.sessions = got;
        let _ = self.evict(0);
        Ok(k)
    }

    fn new_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Choose the session and the token ids for a prompt. The chosen session is busy until
    /// `finish` or `abandon`. `imgs` are the request's image spans (empty for a text request): the
    /// shared prefix is clipped to whole spans, so a session whose images differ or sit elsewhere is
    /// never taken past its first image. Returns the plan and the sessions to RELEASE on the engine.
    pub fn plan(&mut self, text: &str, imgs: &[ImgKey], tok: &Tokenizer, threads: usize) -> (Plan, Vec<u64>) {
        self.clock += 1;
        let mut top: Option<(u64, usize)> = None;
        for (&id, s) in &self.sessions {
            let n = clip_images(&s.imgs, imgs, common_prefix(text.as_bytes(), s.text.as_bytes()));
            if n > 0 && top.map_or(true, |(_, bn)| n > bn) {
                top = Some((id, n));
            }
        }
        let best = top.filter(|&(_, n)| n >= MIN_SHARED);
        let (mut plan, cut) = match best {
            None => (Plan { session_id: self.new_id(), base_session_id: 0, reused: 0, img_reused: Vec::new(), ids: Vec::new(), matched: 0, shared: 0, common: 0 }, 0),
            Some((sid, mut shared)) => {
                while !text.is_char_boundary(shared) {
                    shared -= 1;
                }
                let s = &self.sessions[&sid];
                // The last cut point within the shared text that is a token boundary of the session
                // and an edge of an added token in the new text. A span's ids all share one end, so
                // partition_point lands after the whole span and a cut never falls inside it.
                let mut cut = (0, 0); // (byte position, ids before it)
                for (a, e) in tok.added_spans(text) {
                    if a > shared {
                        break;
                    }
                    for p in [a, e] {
                        if p <= shared && p > cut.0 {
                            let k = s.ends.partition_point(|&x| x as usize <= p);
                            if k > 0 && s.ends[k - 1] as usize == p {
                                cut = (p, k);
                            }
                        }
                    }
                }
                let ids = s.ids[..cut.1].to_vec();
                let img_reused: Vec<u32> = s.imgs.iter().filter(|i| i.byte < cut.0).map(|i| i.at).collect();
                let in_place = !s.busy && shared + PROMPT_SLACK >= s.prompt_len;
                let plan = if in_place {
                    Plan { session_id: sid, base_session_id: 0, reused: cut.1, img_reused, ids, matched: 0, shared: 0, common: 0 }
                } else {
                    Plan { session_id: self.new_id(), base_session_id: sid, reused: cut.1, img_reused, ids, matched: 0, shared: 0, common: 0 }
                };
                (plan, cut.0)
            }
        };
        plan.ids.extend(tok.encode_parallel(&text[cut..], threads));
        if let Some((sid, n)) = top {
            plan.matched = sid;
            plan.shared = n;
            let s = &self.sessions[&sid].ids;
            plan.common = s.iter().zip(&plan.ids).take_while(|(a, b)| a == b).count();
        }
        let clock = self.clock;
        let s = self.sessions.entry(plan.session_id).or_insert_with(|| Session {
            text: String::new(),
            ids: Vec::new(),
            ends: Vec::new(),
            imgs: Vec::new(),
            prompt_len: 0,
            busy: false,
            last_used: clock,
            t_used: 0,
        });
        s.busy = true;
        s.last_used = clock;
        s.t_used = unix_now();
        let released = self.evict(plan.session_id);
        (plan, released)
    }

    /// Record what the session holds after a request: the prompt, its image spans and the generated
    /// ids. `imgs` are the request's spans with their places in `prompt_ids` (empty for text). The
    /// session's text must fit u32 (its ends are u32): one longer is not stored, the request unaffected.
    pub fn finish(&mut self, session_id: u64, prompt_text: &str, prompt_ids: &[u32], generated: &[u32], imgs: &[ImgKey], tok: &Tokenizer) {
        if !self.sessions.contains_key(&session_id) {
            return;
        }
        let gen_bytes = tok.decode_bytes(generated);
        let mut text = prompt_text.to_string();
        let mut ids = prompt_ids.to_vec();
        // Generated text that is not valid UTF-8 cannot come back verbatim; keep the prompt only.
        if let Ok(g) = std::str::from_utf8(&gen_bytes) {
            text.push_str(g);
            ids.extend_from_slice(generated);
        }
        if text.len() > u32::MAX as usize {
            self.sessions.remove(&session_id); // its ends are u32: a session this size is not stored
            self.forget(session_id);
            return;
        }
        let ends = ends_of(&ids, imgs, tok);
        debug_assert_eq!(ends.last().copied().unwrap_or(0) as usize, text.len());
        let s = self.sessions.get_mut(&session_id).unwrap();
        s.text = text;
        s.ids = ids;
        s.ends = ends;
        s.imgs = imgs.to_vec();
        s.prompt_len = prompt_text.len();
        s.busy = false;
        let bytes = session_bytes(session_id, s);
        if let Some(tx) = &self.tx {
            let _ = tx.send(Job::Write(session_id, bytes));  // the finished turn, by the writer's thread
        }
    }

    /// The request failed: forget the session (the engine's copy is in an unknown state; the
    /// caller releases it).
    pub fn abandon(&mut self, session_id: u64) {
        self.sessions.remove(&session_id);
        self.forget(session_id);
    }

    /// Drop least recently used idle sessions over the limits; returns their ids.
    fn evict(&mut self, keep: u64) -> Vec<u64> {
        let mut out = Vec::new();
        loop {
            let total: usize = self.sessions.values().map(|s| s.ids.len()).sum();
            if self.sessions.len() <= self.max_sessions && total <= self.max_tokens {
                return out;
            }
            let victim = self.sessions.iter().filter(|(&id, s)| id != keep && !s.busy).min_by_key(|(_, s)| s.last_used).map(|(&id, _)| id);
            match victim {
                Some(id) => {
                    self.sessions.remove(&id);
                    self.forget(id);
                    out.push(id);
                }
                None => return out,
            }
        }
    }

    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Where the sessions' files go, for the log ("(none)": nothing is written).
    pub fn dir(&self) -> String {
        match &self.dir {
            Some(d) => d.display().to_string(),
            None => "(none)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tokenizer the tests need: `DSFRONT_TEST_TOKENIZER`, or the checkpoint's own path.
    fn tokenizer() -> Option<Tokenizer> {
        let path = std::env::var("DSFRONT_TEST_TOKENIZER").unwrap_or_else(|_| "/data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/tokenizer.json".into());
        Tokenizer::from_file(&path).ok()
    }

    #[test]
    fn reuses_the_models_own_ids() {
        let Some(tok) = tokenizer() else { return };
        let mut t = table_at(None);
        let sys = "<｜begin▁of▁sentence｜><｜System｜>You are a careful assistant who answers briefly and well.";
        let p1 = format!("{sys}<｜User｜>hello<｜Assistant｜></think>");
        let (plan, _) = t.plan(&p1, &[], &tok, 1);
        assert_eq!(plan.ids, tok.encode(&p1));
        // the model spelled "Hi there" in single letters, which BPE would never produce
        let mut gen: Vec<u32> = "Hi there".chars().map(|c| tok.encode(&c.to_string())[0]).collect();
        gen.push(tok.token_id("<｜end▁of▁sentence｜>").unwrap());
        t.finish(plan.session_id, &p1, &plan.ids, &gen, &[], &tok);
        let p2 = format!("{p1}Hi there<｜end▁of▁sentence｜><｜User｜>next<｜Assistant｜></think>");
        let (plan2, _) = t.plan(&p2, &[], &tok, 1);
        assert_eq!(plan2.session_id, plan.session_id);
        let mut want = plan.ids.clone();
        want.extend(&gen);
        want.extend(tok.encode("<｜User｜>next<｜Assistant｜></think>"));
        assert_eq!(plan2.ids, want);
        assert_eq!(plan2.reused, plan.ids.len() + gen.len());
        assert_eq!(tok.decode(&plan2.ids), p2);
        // the same system prompt in another conversation forks instead of taking over the session
        let p3 = format!("{sys}<｜User｜>something else entirely<｜Assistant｜></think>");
        let (plan3, _) = t.plan(&p3, &[], &tok, 1);
        assert_ne!(plan3.session_id, plan.session_id);
        assert_eq!(plan3.base_session_id, plan.session_id);
        assert_eq!(tok.decode(&plan3.ids), p3);
    }

    #[test]
    fn the_plan_says_where_it_starts() {
        let Some(tok) = tokenizer() else { return };
        let mut t = table_at(None);
        // pi-like: a long system text that ends with the time, then the tools and the first message
        let body = "You are an expert coding assistant. Read files, run commands and edit code.\n".repeat(40);
        let conv = |time: &str, msg: &str| {
            format!("<｜begin▁of▁sentence｜><｜System｜>{body}Current date and time: Monday, October 12, 2026 at {time}\n\n## Tools\n\nnone<｜User｜>{msg}<｜Assistant｜></think>")
        };
        let a = conv("09:14:08 AM", "fix the build");
        let (pa, _) = t.plan(&a, &[], &tok, 1);
        assert_eq!(pa.describe(a.len()), format!("session {}, new: no session shares a byte", pa.session_id));
        t.finish(pa.session_id, &a, &pa.ids, &tok.encode("Done."), &[], &tok);
        // another conversation, a minute later: a fork of the first, which has every id before the time
        let b = conv("09:15:31 AM", "write the tests");
        let (pb, _) = t.plan(&b, &[], &tok, 1);
        assert_eq!(pb.base_session_id, pa.session_id);
        assert_eq!(pb.ids, tok.encode(&b), "the ids are a cold run's");
        let head = tok.encode(&b[..b.find("09:15").unwrap()]).len();
        assert!(pb.common + 2 >= head && pb.common <= head + 2, "{} ids in common, {head} before the time", pb.common);
        let d = pb.describe(b.len());
        assert!(d.starts_with(&format!("session {}, a fork of {}: it shares ", pb.session_id, pa.session_id)), "{d}");
        assert!(d.ends_with(&format!("of {} bytes, {} ids in common", b.len(), pb.common)), "{d}");
        // the next turn of the first: in place
        let a2 = format!("{a}Done.<｜end▁of▁sentence｜><｜User｜>thanks<｜Assistant｜></think>");
        let (pa2, _) = t.plan(&a2, &[], &tok, 1);
        assert_eq!(pa2.session_id, pa.session_id);
        assert!(pa2.describe(a2.len()).starts_with(&format!("session {} in place: it shares ", pa.session_id)));
        t.finish(pa2.session_id, &a2, &pa2.ids, &tok.encode("Sure."), &[], &tok);
        // a prompt that shares only the BOS and a few bytes: a new session, which names no base
        let c = "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>";
        let (pc, _) = t.plan(c, &[], &tok, 1);
        assert_eq!(pc.base_session_id, 0);
        let n = common_prefix(c.as_bytes(), a.as_bytes());
        assert!(n > 0 && n < MIN_SHARED);
        let d = pc.describe(c.len());
        assert!(d.starts_with(&format!("session {}, new: the best match, session ", pc.session_id)), "{d}");
        assert!(d.ends_with(&format!(", shares {n} bytes (under 64)")), "{d}");
    }

    /// A table that writes its sessions to `dir` (None: nothing is written or read).
    fn table_at(dir: Option<&Path>) -> SessionTable {
        SessionTable::new(8, 1 << 20, dir.map(|d| d.to_path_buf()))
    }

    #[test]
    fn a_finished_turn_leaves_its_file_and_the_next_start_finds_it() {
        let Some(tok) = tokenizer() else { return };
        let dir = std::env::temp_dir().join(format!("dsfront-table-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let mut t = table_at(Some(&dir));
        let body = "You are an expert coding assistant. Read files, run commands and edit code.\n".repeat(3);
        let p1 = format!("You are a careful assistant.\n{body}\n{}{}hello{}", crate::encoding::USER_SP, crate::encoding::USER_SP, crate::encoding::ASSISTANT_SP);
        let (plan, _) = t.plan(&p1, &[], &tok, 1);
        let gen = tok.encode("Hi there.");
        t.finish(plan.session_id, &p1, &plan.ids, &gen, &[], &tok);
        t.drain(); // the writer's queue is empty: the turn is on the disk
        let file = dir.join(format!("{:016x}.t", plan.session_id));
        assert_eq!(std::fs::metadata(&file).unwrap().mode() & 0o777, 0o600);
        assert_eq!(std::fs::metadata(&dir).unwrap().mode() & 0o777, 0o700);
        // a directory in a git tree is refused: nothing is written there
        let repo = dir.join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir(repo.join("run")).unwrap();
        let mut g = table_at(Some(&repo.join("run")));
        let (gp, _) = g.plan(&p1, &[], &tok, 1);
        g.finish(gp.session_id, &p1, &gp.ids, &gen, &[], &tok);
        g.drain();
        assert!(!repo.join("run").join(format!("{:016x}.t", gp.session_id)).exists());
        std::fs::remove_dir_all(&repo).unwrap();
        // the next start takes the session back, and a turn of it goes on from where it left
        let mut u = table_at(Some(&dir));
        assert_eq!(u.load(&tok, 3600), Ok(1));
        assert!(u.next_id > plan.session_id, "ids go on after the files'");
        let p2 = format!("{p1}Hi there.{}{}next{}", crate::encoding::USER_SP, crate::encoding::USER_SP, crate::encoding::ASSISTANT_SP);
        let (plan2, _) = u.plan(&p2, &[], &tok, 1);
        assert_eq!(plan2.session_id, plan.session_id);
        assert_eq!(plan2.reused, plan.ids.len() + gen.len());
        assert_eq!(tok.decode(&plan2.ids), p2);
        // a damaged file: it is not taken, and it goes
        let mut b = std::fs::read(&file).unwrap();
        let k = b.len() - 3;
        b[k] ^= 1;
        std::fs::write(&file, &b).unwrap();
        assert_eq!(table_at(Some(&dir)).load(&tok, 3600), Ok(0));
        assert!(!file.exists(), "a damaged file goes");
        // a file written longer ago than the limit: not taken, and gone
        t.finish(plan.session_id, &p1, &plan.ids, &gen, &[], &tok);
        t.drain();
        let mut b = std::fs::read(&file).unwrap();
        b[8..16].copy_from_slice(&1u64.to_le_bytes()); // `written` long ago
        std::fs::write(&file, &b).unwrap();
        assert_eq!(table_at(Some(&dir)).load(&tok, 3600), Ok(0));
        assert!(!file.exists(), "an old file goes");
        // a session that goes takes its file with it, whatever the queue's order
        let other = "Another conversation entirely, with nothing in common.";
        let (plan3, _) = t.plan(other, &[], &tok, 1);
        t.finish(plan3.session_id, other, &plan3.ids, &tok.encode("ok"), &[], &tok);
        t.drain();
        let f3 = dir.join(format!("{:016x}.t", plan3.session_id));
        assert!(f3.exists());
        t.abandon(plan3.session_id);
        t.drain();
        assert!(!f3.exists(), "the removal queued after its write takes it away");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_table_evicts_at_its_limits() {
        let Some(tok) = tokenizer() else { return };
        let prompt = |msg: &str| format!("<｜System｜>You are a careful assistant who answers briefly and well.<｜User｜>{msg}<｜Assistant｜></think>");
        let turn = |t: &mut SessionTable, msg: &str| {
            let p = prompt(msg);
            let (plan, _) = t.plan(&p, &[], &tok, 1);
            t.finish(plan.session_id, &p, &plan.ids, &tok.encode("ok"), &[], &tok);
            plan.session_id
        };
        // two sessions at most: the third drops the least recently used
        let mut t = SessionTable::new(2, 1 << 20, None);
        let a = turn(&mut t, "one");
        let b = turn(&mut t, "two");
        assert_eq!(t.len(), 2);
        let c = turn(&mut t, "three");
        assert_eq!(t.len(), 2, "the table holds its two sessions");
        assert!(!t.sessions.contains_key(&a), "the least recently used went");
        assert!(t.sessions.contains_key(&b) && t.sessions.contains_key(&c));
        // an id budget: a session over it drops the one before it (eviction is planned with the new
        // session still empty, so the sessions already stored are what the budget measures)
        let held = tok.encode(&prompt("one")).len() + tok.encode("ok").len();
        let mut u = SessionTable::new(8, held - 1, None);
        let a = turn(&mut u, "one");
        assert_eq!(u.len(), 1, "one session of {held} ids is stored");
        let b = turn(&mut u, "two");
        assert_eq!(u.len(), 1, "two sessions of {held} ids do not fit a budget of {}", held - 1);
        assert!(!u.sessions.contains_key(&a) && u.sessions.contains_key(&b));
    }

    /// A prompt part: literal text, or the placeholder of an image span.
    enum Part<'a> {
        Text(&'a str),
        Img { rows: u32, hash: u64 },
    }

    /// A prompt with `Img` parts as placeholders: the text, the ids a cold run sends (each placeholder
    /// expanded, as server.rs does) and the request's keys.
    fn prompt_of(tok: &Tokenizer, parts: &[Part]) -> (String, Vec<u32>, Vec<ImgKey>) {
        let ph = tok.token_bytes(crate::vision::IMAGE_TOKEN_ID).to_vec();
        let (mut text, mut ids, mut imgs) = (String::new(), Vec::new(), Vec::new());
        for p in parts {
            match p {
                Part::Text(s) => {
                    text.push_str(s);
                    ids.extend(tok.encode(s));
                }
                Part::Img { rows, hash } => {
                    let (byte, at) = (text.len(), ids.len() as u32);
                    text.push_str(&String::from_utf8_lossy(&ph));
                    ids.extend(std::iter::repeat(crate::vision::IMAGE_TOKEN_ID).take(*rows as usize));
                    imgs.push(ImgKey { byte, at, rows: *rows, hash: *hash });
                }
            }
        }
        (text, ids, imgs)
    }

    /// The ids and span starts server.rs would send for a plan: the reused part as it is, the tail's
    /// placeholders expanded.
    fn sent(plan: &Plan, rows: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let (tail, tail_starts) = crate::vision::expand(&plan.ids[plan.reused..], &rows[plan.img_reused.len()..]).unwrap();
        let mut ids = plan.ids[..plan.reused].to_vec();
        ids.extend(tail);
        let mut starts = plan.img_reused.clone();
        starts.extend(tail_starts.iter().map(|&s| s + plan.reused as u32));
        (ids, starts)
    }

    const SYS: &str = "<｜System｜>You are a careful assistant who answers briefly and well.";

    #[test]
    fn an_image_turn_reuses_its_span() {
        let Some(tok) = tokenizer() else { return };
        let mut t = table_at(None);
        let (p1, ids1, imgs1) = prompt_of(&tok, &[Part::Text(SYS), Part::Img { rows: 3, hash: 7 }, Part::Text("what is this?")]);
        let (plan, _) = t.plan(&p1, &imgs1, &tok, 1);
        assert_eq!(plan.reused, 0, "a cold run");
        t.finish(plan.session_id, &p1, &ids1, &tok.encode("Hi there."), &imgs1, &tok);
        let (p2, ids2, imgs2) = prompt_of(
            &tok,
            &[Part::Text(SYS), Part::Img { rows: 3, hash: 7 }, Part::Text("what is this?"), Part::Text("Hi there."), Part::Text(crate::encoding::ASSISTANT_SP), Part::Text("next")],
        );
        let (plan2, _) = t.plan(&p2, &imgs2, &tok, 1);
        assert_eq!(plan2.session_id, plan.session_id);
        // the whole span is inside the reused ids, and the plan names it
        assert_eq!(plan2.img_reused, vec![imgs2[0].at]);
        assert!(plan2.reused > (imgs2[0].at + imgs2[0].rows) as usize, "{} ids reused, the span ends at {}", plan2.reused, imgs2[0].at + imgs2[0].rows);
        // what server.rs sends is the whole prompt, its placeholder expanded
        let rows: Vec<u32> = imgs2.iter().map(|i| i.rows).collect();
        assert_eq!(sent(&plan2, &rows), (ids2, vec![imgs2[0].at]));
    }

    #[test]
    fn a_different_image_cuts_before_its_span() {
        let Some(tok) = tokenizer() else { return };
        let mut t = table_at(None);
        let (p1, ids1, imgs1) = prompt_of(&tok, &[Part::Text(SYS), Part::Img { rows: 3, hash: 7 }, Part::Text("what is this?")]);
        let (plan, _) = t.plan(&p1, &imgs1, &tok, 1);
        t.finish(plan.session_id, &p1, &ids1, &tok.encode("Hi there."), &imgs1, &tok);
        // the same grid at the same place, another image: the same ids, a different hash
        let (p2, _, imgs2) = prompt_of(&tok, &[Part::Text(SYS), Part::Img { rows: 3, hash: 9 }, Part::Text("what is this?")]);
        let (plan2, _) = t.plan(&p2, &imgs2, &tok, 1);
        // the cut is before the placeholder, so the turn is a fork of the session, not a continuation
        assert_eq!(plan2.base_session_id, plan.session_id);
        assert_eq!(plan2.img_reused, Vec::<u32>::new());
        assert_eq!(plan2.reused, imgs2[0].at as usize, "the cut is at the placeholder's byte");
    }

    #[test]
    fn a_text_turn_cuts_before_an_image() {
        let Some(tok) = tokenizer() else { return };
        let mut t = table_at(None);
        let (p1, ids1, imgs1) = prompt_of(&tok, &[Part::Text(SYS), Part::Img { rows: 3, hash: 7 }, Part::Text("what is this?")]);
        let (plan, _) = t.plan(&p1, &imgs1, &tok, 1);
        t.finish(plan.session_id, &p1, &ids1, &tok.encode("Hi there."), &imgs1, &tok);
        // a request that names no image, its text holding the placeholder literally: the shared prefix
        // stops at the session's first image
        let (p2, _, _) = prompt_of(&tok, &[Part::Text(SYS), Part::Img { rows: 3, hash: 7 }, Part::Text("what is this?")]);
        let (plan2, _) = t.plan(&p2, &[], &tok, 1);
        assert_eq!(plan2.base_session_id, plan.session_id);
        assert_eq!(plan2.img_reused, Vec::<u32>::new());
        assert_eq!(plan2.reused, imgs1[0].at as usize);
    }

    #[test]
    fn a_cut_never_falls_inside_a_span() {
        let Some(tok) = tokenizer() else { return };
        let mut t = table_at(None);
        let (p1, ids1, imgs1) = prompt_of(&tok, &[Part::Text(SYS), Part::Img { rows: 3, hash: 1 }, Part::Text("hi"), Part::Img { rows: 4, hash: 2 }, Part::Text("there")]);
        let (plan, _) = t.plan(&p1, &imgs1, &tok, 1);
        t.finish(plan.session_id, &p1, &ids1, &tok.encode("Sure."), &imgs1, &tok);
        let full = format!("{p1}Sure.");
        for i in 0..=full.len() {
            if !full.is_char_boundary(i) {
                continue;
            }
            let (p, _) = t.plan(&format!("{}wrap", &full[..i]), &imgs1, &tok, 1);
            for im in &imgs1 {
                assert!(
                    p.reused <= im.at as usize || p.reused >= (im.at + im.rows) as usize,
                    "the cut at byte {i} lands at id {} inside the span {im:?}",
                    p.reused
                );
            }
        }
    }

    #[test]
    fn a_saved_image_session_keeps_its_spans() {
        let Some(tok) = tokenizer() else { return };
        let dir = std::env::temp_dir().join(format!("dsfront-table-img-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let mut t = table_at(Some(&dir));
        let (p1, ids1, imgs1) = prompt_of(&tok, &[Part::Text(SYS), Part::Img { rows: 3, hash: 7 }, Part::Text("hi")]);
        let (plan, _) = t.plan(&p1, &imgs1, &tok, 1);
        t.finish(plan.session_id, &p1, &ids1, &tok.encode("Sure."), &imgs1, &tok);
        t.drain();
        let file = dir.join(format!("{:016x}.t", plan.session_id));
        let mut u = table_at(Some(&dir));
        assert_eq!(u.load(&tok, 3600), Ok(1));
        // the session's spans come back: a turn with the same image reuses past them
        let (p2, _, imgs2) =
            prompt_of(&tok, &[Part::Text(SYS), Part::Img { rows: 3, hash: 7 }, Part::Text("hi"), Part::Text("Sure."), Part::Text(crate::encoding::ASSISTANT_SP), Part::Text("next")]);
        let (plan2, _) = u.plan(&p2, &imgs2, &tok, 1);
        assert_eq!(plan2.session_id, plan.session_id);
        assert_eq!(plan2.img_reused, vec![imgs2[0].at]);
        assert!(plan2.reused > (imgs2[0].at + imgs2[0].rows) as usize);
        // another version's file is refused, and the file goes
        let mut b = std::fs::read(&file).unwrap();
        b[..8].copy_from_slice(b"DSFT0002");
        std::fs::write(&file, &b).unwrap();
        assert_eq!(table_at(Some(&dir)).load(&tok, 3600), Ok(0));
        assert!(!file.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
