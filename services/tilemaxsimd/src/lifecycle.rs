// Copyright (c) 2026 HuXinjing

use anyhow::{Context, Result, bail};
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

pub struct ManagedFile {
    path: PathBuf,
    device: u64,
    inode: u64,
    _file: fs::File,
}

impl ManagedFile {
    pub fn create_pid(path: &Path) -> Result<Self> {
        remove_stale_pid_file(path)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .with_context(|| format!("cannot create PID file {}", path.display()))?;
        writeln!(file, "{}", std::process::id())?;
        file.sync_all()?;
        let metadata = file.metadata()?;
        Ok(Self {
            path: path.to_owned(),
            device: metadata.dev(),
            inode: metadata.ino(),
            _file: file,
        })
    }

    pub fn create_ready(path: &Path, contents: &[u8]) -> Result<Self> {
        if fs::symlink_metadata(path).is_ok() {
            bail!("ready file already exists: {}", path.display());
        }
        let parent = path.parent().filter(|value| !value.as_os_str().is_empty());
        let parent = parent.unwrap_or_else(|| Path::new("."));
        let name = path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("ready file path has no file name"))?
            .to_string_lossy();
        let temporary = parent.join(format!(".{name}.{}.tmp", std::process::id()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)
            .with_context(|| {
                format!("cannot create temporary ready file {}", temporary.display())
            })?;
        let result = (|| -> Result<Self> {
            file.write_all(contents)?;
            file.sync_all()?;
            fs::hard_link(&temporary, path)
                .with_context(|| format!("cannot publish ready file {}", path.display()))?;
            let metadata = fs::symlink_metadata(path)?;
            let guard = Self {
                path: path.to_owned(),
                device: metadata.dev(),
                inode: metadata.ino(),
                _file: file,
            };
            fs::File::open(parent)?.sync_all()?;
            Ok(guard)
        })();
        let _ = fs::remove_file(&temporary);
        result
    }

    pub fn remove(mut self) -> Result<()> {
        self.remove_if_owned()
    }

    fn remove_if_owned(&mut self) -> Result<()> {
        let metadata = match fs::symlink_metadata(&self.path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        if metadata.dev() != self.device || metadata.ino() != self.inode {
            bail!("refusing to remove replaced file {}", self.path.display());
        }
        fs::remove_file(&self.path)?;
        Ok(())
    }
}

impl Drop for ManagedFile {
    fn drop(&mut self) {
        let _ = self.remove_if_owned();
    }
}

fn remove_stale_pid_file(path: &Path) -> Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if !metadata.file_type().is_file() || metadata.uid() != unsafe { libc::geteuid() } {
        bail!("refusing to replace unsafe PID file {}", path.display());
    }
    let contents = fs::read_to_string(path)
        .with_context(|| format!("cannot read existing PID file {}", path.display()))?;
    let pid = contents
        .trim()
        .parse::<libc::pid_t>()
        .with_context(|| format!("invalid existing PID file {}", path.display()))?;
    if pid <= 0 {
        bail!("invalid existing PID file {}", path.display());
    }
    // SAFETY: kill(pid, 0) performs existence/permission checking only.
    if unsafe { libc::kill(pid, 0) } == 0 {
        bail!("daemon recorded in {} is still running", path.display());
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() != Some(libc::ESRCH) {
        return Err(error).context("cannot verify existing daemon PID");
    }
    let current = fs::symlink_metadata(path)?;
    if current.dev() != metadata.dev() || current.ino() != metadata.ino() {
        bail!("PID file changed while checking {}", path.display());
    }
    fs::remove_file(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn directory(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "vchord-tilemaxsim-lifecycle-{name}-{}",
            std::process::id()
        ))
    }

    #[test]
    fn managed_ready_file_is_removed() {
        let directory = directory("ready-remove");
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("ready");
        let _ = fs::remove_file(&path);
        let guard = ManagedFile::create_ready(&path, b"ready\n").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"ready\n");
        guard.remove().unwrap();
        assert!(!path.exists());
        fs::remove_dir(&directory).unwrap();
    }

    #[test]
    fn dropping_guard_does_not_remove_replacement_file() {
        let directory = directory("ready-replacement");
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("ready");
        let _ = fs::remove_file(&path);
        let guard = ManagedFile::create_ready(&path, b"original").unwrap();
        fs::remove_file(&path).unwrap();
        fs::write(&path, b"replacement").unwrap();
        drop(guard);
        assert_eq!(fs::read(&path).unwrap(), b"replacement");
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&directory).unwrap();
    }

    #[test]
    fn pid_file_refuses_live_process_and_replaces_stale_process() {
        let directory = directory("pid");
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("daemon.pid");
        fs::write(&path, format!("{}\n", std::process::id())).unwrap();
        let error = ManagedFile::create_pid(&path).err().unwrap();
        assert!(format!("{error:#}").contains("still running"));
        fs::write(&path, format!("{}\n", i32::MAX)).unwrap();
        let guard = ManagedFile::create_pid(&path).unwrap();
        assert_eq!(
            fs::read_to_string(&path).unwrap().trim(),
            std::process::id().to_string()
        );
        guard.remove().unwrap();
        fs::remove_dir(&directory).unwrap();
    }
}
