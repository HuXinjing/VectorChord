//! Durable logical cache state. Device pointers, leases and in-flight work are never persisted.
use crate::protocol::Descriptor;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_BYTES: u64 = 128 * 1024 * 1024;
pub const MAX_ENTRIES: usize = 1_000_000;
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub descriptor: Descriptor,
    pub tenant: String,
    pub profile: String,
    pub quantization_contract: Option<String>,
    pub query_dtype: u8,
    pub pinned: bool,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    version: u32,
    entries: Vec<Entry>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    version: u32,
    payload: String,
    sha256: String,
}
#[derive(Serialize)]
struct StateRef<'a> {
    version: u32,
    entries: &'a [Entry],
}
struct BoundedWriter<W> {
    inner: W,
    written: u64,
    limit: u64,
}
impl<W: Write> Write for BoundedWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() as u64 > self.limit.saturating_sub(self.written) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "cache recovery byte limit exceeded",
            ));
        }
        let n = self.inner.write(bytes)?;
        self.written += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

pub struct RecoveryFile {
    path: PathBuf,
    generations: Option<Vec<u64>>,
    _lock: fs::File,
}
impl RecoveryFile {
    pub fn open(path: PathBuf) -> Result<Self> {
        let parent = path.parent().context("cache recovery directory required")?;
        if !path.is_absolute() || parent.canonicalize()? != parent {
            bail!("cache recovery requires a canonical absolute directory");
        }
        let m = fs::symlink_metadata(parent)?;
        if !m.is_dir() || m.uid() != unsafe { libc::geteuid() } || m.mode() & 0o022 != 0 {
            bail!("cache recovery directory must be owned and not writable by other users");
        }
        let mut name = path
            .file_name()
            .context("cache recovery filename required")?
            .to_os_string();
        name.push(".lock");
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(parent.join(name))?;
        let m = lock.metadata()?;
        if !m.is_file()
            || m.uid() != unsafe { libc::geteuid() }
            || m.mode() & 0o077 != 0
            || m.nlink() != 1
        {
            bail!("private cache recovery lease required");
        }
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            bail!("cache recovery state is already owned by another daemon");
        }
        Ok(Self {
            path,
            generations: None,
            _lock: lock,
        })
    }
    pub fn load(&self) -> Result<Vec<Entry>> {
        let mut file = match OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&self.path)
        {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };
        let m = file.metadata()?;
        if !m.is_file()
            || m.uid() != unsafe { libc::geteuid() }
            || m.mode() & 0o077 != 0
            || m.nlink() != 1
            || m.len() > MAX_BYTES
        {
            bail!("cache recovery file must be a private bounded regular file");
        }
        let mut bytes = Vec::new();
        Read::by_ref(&mut file)
            .take(MAX_BYTES + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_BYTES {
            bail!("cache recovery file exceeds byte limit");
        }
        let envelope: Envelope =
            serde_json::from_slice(&bytes).context("invalid cache recovery envelope")?;
        if envelope.version != 1
            || envelope.sha256 != hex::encode(Sha256::digest(envelope.payload.as_bytes()))
        {
            bail!("cache recovery checksum or version mismatch");
        }
        let state: State =
            serde_json::from_str(&envelope.payload).context("invalid cache recovery state")?;
        if state.version != 1 || state.entries.len() > MAX_ENTRIES {
            bail!("unsupported or oversized cache recovery state");
        }
        for entry in &state.entries {
            validate(entry)?;
        }
        Ok(state.entries)
    }
    pub fn commit(&mut self, engine: &mut crate::engine::Engine, force: bool) -> Result<()> {
        let generations = engine.recovery_generations();
        if !force && self.generations.as_ref() == Some(&generations) {
            return Ok(());
        }
        let started = std::time::Instant::now();
        let entries = engine.recovery_snapshot()?;
        let count = entries.len();
        self.publish(&entries)?;
        self.generations = Some(generations);
        println!(
            "{}",
            serde_json::json!({"event":"tilemaxsim_cache_checkpoint_committed","entries":count,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0})
        );
        Ok(())
    }
    fn publish(&self, entries: &[Entry]) -> Result<()> {
        if entries.len() > MAX_ENTRIES {
            bail!("cache recovery entry limit exceeded");
        }
        let mut payload = BoundedWriter {
            inner: Vec::new(),
            written: 0,
            limit: MAX_BYTES / 2,
        };
        serde_json::to_writer(
            &mut payload,
            &StateRef {
                version: 1,
                entries,
            },
        )?;
        let payload = String::from_utf8(payload.inner)?;
        let envelope = Envelope {
            version: 1,
            sha256: hex::encode(Sha256::digest(payload.as_bytes())),
            payload,
        };
        let parent = self.path.parent().unwrap();
        // Recheck the destination before atomic replacement, including on later commits.
        match self.path.symlink_metadata() {
            Ok(_) => {
                self.load()?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        // Container restarts commonly reuse PID 1. An orphan from a crash
        // must not collide with the next process's first temporary filename.
        let mut nonce = [0u8; 16];
        fs::File::open("/dev/urandom")?.read_exact(&mut nonce)?;
        let temp = parent.join(format!(
            ".cache-recovery-{}-{}-{}.tmp",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed),
            hex::encode(nonce),
        ));
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&temp)?;
            let mut writer = BoundedWriter {
                inner: BufWriter::with_capacity(256 * 1024, &mut file),
                written: 0,
                limit: MAX_BYTES,
            };
            serde_json::to_writer(&mut writer, &envelope)?;
            // Flush userspace buffering before fsync and atomic publication.
            writer.flush()?;
            drop(writer);
            file.sync_all()?;
            fs::rename(&temp, &self.path)?;
            fs::File::open(parent)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }
}
pub(crate) fn validate(entry: &Entry) -> Result<()> {
    let d = &entry.descriptor;
    if entry.tenant.is_empty()
        || entry.tenant.len() > 256
        || entry.tenant.chars().any(char::is_control)
        || d.contract.is_empty()
        || d.contract.len() > 512
        || d.contract.chars().any(char::is_control)
        || d.digest.len() != 64
        || !d
            .digest
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        || d.rows == 0
        || d.dimension == 0
        || d.dimension > 60_000
        || ![1, 2, 3].contains(&d.dtype)
        || ![1, 2, 3].contains(&entry.query_dtype)
        || entry
            .quantization_contract
            .as_ref()
            .is_some_and(|s| s.is_empty() || s.len() > 128)
    {
        bail!("invalid logical cache recovery entry");
    }
    let profile = profile(&entry.profile)?;
    if matches!(
        profile,
        crate::protocol::ScoringProfile::Pq | crate::protocol::ScoringProfile::OpqRpq
    ) != entry.quantization_contract.is_some()
        || entry.quantization_contract.as_ref().is_some_and(|id| {
            id.len() != 69
                || !id.starts_with("qtc1-")
                || !id[5..]
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        })
    {
        bail!("invalid cache recovery quantization contract");
    }
    Ok(())
}
pub fn profile(tag: &str) -> Result<crate::protocol::ScoringProfile> {
    use crate::protocol::ScoringProfile::*;
    Ok(match tag {
        "exact-fp16-v1" => ExactFp16,
        "int8-row-scale-v1" => Int8,
        "fp8-e4m3-row-scale-v1" => Fp8E4m3,
        "fp8-e4m3-raw-v1" => RawFp8E4m3,
        "pq-adc-v1" => Pq,
        "opq-rpq-adc-v1" => OpqRpq,
        _ => bail!("unsupported cache recovery scoring profile"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};
    struct Root(PathBuf);
    impl Root {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!(
                "cache-state-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&p).unwrap();
            fs::set_permissions(&p, fs::Permissions::from_mode(0o700)).unwrap();
            Self(p)
        }
        fn file(&self) -> PathBuf {
            self.0.join("cache.json")
        }
    }
    impl Drop for Root {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn entry() -> Entry {
        Entry {
            descriptor: Descriptor {
                candidate_id: 0,
                contract: "embedding@revision-7".into(),
                digest: "ab".repeat(32),
                rows: 3,
                dimension: 8,
                dtype: 2,
                raw_fp8_cache_key: Default::default(),
            },
            tenant: "tenant-a".into(),
            profile: "exact-fp16-v1".into(),
            quantization_contract: None,
            query_dtype: 2,
            pinned: false,
        }
    }
    #[test]
    fn checkpoint_is_atomic_private_and_reopens_after_owner_drops() {
        let root = Root::new();
        let file = RecoveryFile::open(root.file()).unwrap();
        assert!(file.load().unwrap().is_empty());
        file.publish(&[entry()]).unwrap();
        assert_eq!(file.load().unwrap().len(), 1);
        assert_eq!(fs::metadata(root.file()).unwrap().mode() & 0o777, 0o600);
        assert!(RecoveryFile::open(root.file()).is_err());
        drop(file);
        let file = RecoveryFile::open(root.file()).unwrap();
        assert_eq!(file.load().unwrap()[0].tenant, "tenant-a");
        // A crash before publication must leave the preceding committed state readable.
        fs::write(root.0.join(".cache-recovery-incomplete.tmp"), b"{partial").unwrap();
        assert_eq!(file.load().unwrap().len(), 1);
        // Simulate a PID/sequence reused by a new container after a partial write.
        let orphan = root.0.join(format!(
            ".cache-recovery-{}-{}.tmp",
            std::process::id(),
            SEQUENCE.load(Ordering::Relaxed)
        ));
        fs::write(&orphan, b"{partial").unwrap();
        file.publish(&[entry()]).unwrap();
        assert_eq!(file.load().unwrap().len(), 1);
        assert_eq!(fs::read(orphan).unwrap(), b"{partial");
    }
    #[test]
    fn corrupt_version_and_unsafe_entries_are_rejected_without_replacing_them() {
        let root = Root::new();
        let file = RecoveryFile::open(root.file()).unwrap();
        file.publish(&[entry()]).unwrap();
        for bytes in [
            b"{partial".to_vec(),
            br#"{"version":2,"entries":[]}"#.to_vec(),
            br#"{"version":1,"entries":[],"pointer":17}"#.to_vec(),
        ] {
            fs::write(root.file(), &bytes).unwrap();
            assert!(file.load().is_err());
            assert!(file.publish(&[entry()]).is_err());
            assert_eq!(fs::read(root.file()).unwrap(), bytes);
        }
        let mut e = entry();
        e.descriptor.digest = "../tensor".into();
        assert!(validate(&e).is_err());
        e = entry();
        e.query_dtype = 0;
        assert!(validate(&e).is_err());
        e = entry();
        e.profile = "unknown".into();
        assert!(validate(&e).is_err());
    }
    #[test]
    fn large_checkpoint_preserves_every_logical_entry() {
        let root = Root::new();
        let file = RecoveryFile::open(root.file()).unwrap();
        let entries: Vec<_> = (0..32768)
            .map(|i| {
                let mut value = entry();
                value.descriptor.digest = format!("{i:064x}");
                value.tenant = format!("tenant-{}", i % 3);
                value.pinned = i % 5 == 0;
                value
            })
            .collect();
        let started = std::time::Instant::now();
        file.publish(&entries).unwrap();
        println!(
            "large checkpoint publish_ms={}",
            started.elapsed().as_secs_f64() * 1000.0
        );
        let restored = file.load().unwrap();
        assert_eq!(
            serde_json::to_vec(&restored).unwrap(),
            serde_json::to_vec(&entries).unwrap()
        );
    }
    #[test]
    fn checkpoint_text_limits_match_the_scoring_protocol() {
        let mut value = entry();
        value.tenant = "t".repeat(256);
        value.descriptor.contract = "c".repeat(512);
        assert!(validate(&value).is_ok());
        value.tenant.push('t');
        assert!(validate(&value).is_err());
        value.tenant = "tenant-a".into();
        value.descriptor.contract.push('c');
        assert!(validate(&value).is_err());
        value.descriptor.contract = "model@1".into();
        value.tenant = "tenant\u{0085}".into();
        assert!(validate(&value).is_err());
    }
    #[test]
    fn valid_json_payload_tampering_is_rejected() {
        let root = Root::new();
        let file = RecoveryFile::open(root.file()).unwrap();
        file.publish(&[entry()]).unwrap();
        let mut envelope: Envelope =
            serde_json::from_slice(&fs::read(root.file()).unwrap()).unwrap();
        envelope.payload = envelope.payload.replace("tenant-a", "tenant-b");
        fs::write(root.file(), serde_json::to_vec(&envelope).unwrap()).unwrap();
        assert!(file.load().is_err());
    }
    #[test]
    fn symlinks_shared_files_and_shared_parents_are_not_recovery_authorities() {
        let root = Root::new();
        let file = RecoveryFile::open(root.file()).unwrap();
        let target = root.0.join("other");
        fs::write(&target, b"untouched").unwrap();
        symlink(&target, root.file()).unwrap();
        assert!(file.load().is_err());
        assert!(file.publish(&[]).is_err());
        assert_eq!(fs::read(&target).unwrap(), b"untouched");
        fs::remove_file(root.file()).unwrap();
        file.publish(&[]).unwrap();
        fs::set_permissions(root.file(), fs::Permissions::from_mode(0o640)).unwrap();
        assert!(file.load().is_err());
        fs::set_permissions(&root.0, fs::Permissions::from_mode(0o770)).unwrap();
        assert!(RecoveryFile::open(root.file()).is_err());
    }
}
