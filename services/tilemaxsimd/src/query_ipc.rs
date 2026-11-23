// This software is licensed under the repository's dual license model.
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

const FRAME_BYTES: usize = 160;
const MAX_PENDING: usize = 64;
const MAX_QUERY_BYTES: u32 = 32 << 20;
pub const WIRE_DTYPE_FP16_REFERENCE: u8 = 0x82;

pub type QueryId = [u8; 16];

pub struct QueryLease {
    pub rows: u32,
    pub dimension: u32,
    pub bytes: u32,
    pub memory: [u8; 64],
    pub ready: [u8; 64],
    peer: UnixStream,
    expires_at: Instant,
    release: std::sync::atomic::AtomicBool,
}

impl fmt::Debug for QueryLease {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.debug_struct("QueryLease")
            .field("rows", &self.rows)
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

impl QueryLease {
    pub fn pin_after_failed_unmap(&self) {
        self.release
            .store(false, std::sync::atomic::Ordering::Release);
    }
}

impl Drop for QueryLease {
    fn drop(&mut self) {
        if self.release.load(std::sync::atomic::Ordering::Acquire) {
            // The producer only frees after all mappings have been closed.
            let _ = self.peer.write_all(b"R");
        }
    }
}

#[derive(Default)]
pub struct QueryInbox {
    pending: Mutex<HashMap<QueryId, QueryLease>>,
    published: Condvar,
}

impl QueryInbox {
    pub fn receive(&self, mut peer: UnixStream) -> Result<()> {
        peer.set_read_timeout(Some(Duration::from_millis(100)))?;
        peer.set_write_timeout(Some(Duration::from_millis(100)))?;
        let mut frame = [0; FRAME_BYTES];
        let decoded = peer.read_exact(&mut frame).and_then(|()| {
            let invalid =
                || std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid query IPC frame");
            let field = |offset| u32::from_be_bytes(frame[offset..offset + 4].try_into().unwrap());
            let id: QueryId = frame[4..20].try_into().unwrap();
            let (rows, dimension, bytes) = (field(20), field(24), field(28));
            if &frame[..4] != b"TQI1"
                || id == [0; 16]
                || rows == 0
                || dimension == 0
                || bytes > MAX_QUERY_BYTES
                || rows.checked_mul(dimension).and_then(|n| n.checked_mul(2)) != Some(bytes)
            {
                return Err(invalid());
            }
            Ok((id, rows, dimension, bytes))
        });
        let (id, rows, dimension, bytes) = match decoded {
            Ok(value) => value,
            Err(error) => {
                // A rejected frame cannot have been mapped by this process.
                let _ = peer.write_all(b"R");
                return Err(error.into());
            }
        };
        let value = QueryLease {
            rows,
            dimension,
            bytes,
            memory: frame[32..96].try_into().unwrap(),
            ready: frame[96..160].try_into().unwrap(),
            peer,
            expires_at: Instant::now() + Duration::from_secs(5),
            release: std::sync::atomic::AtomicBool::new(true),
        };
        let mut pending = self.pending.lock().unwrap();
        if pending.len() >= MAX_PENDING || pending.contains_key(&id) {
            bail!("query IPC reference capacity");
        }
        pending.insert(id, value);
        self.published.notify_all();
        Ok(())
    }

    pub fn claim(&self, id: QueryId, rows: u32, dimension: u32) -> Result<Arc<QueryLease>> {
        let pending = self.pending.lock().unwrap();
        let (mut pending, _) = self
            .published
            .wait_timeout_while(pending, Duration::from_millis(50), |entries| {
                !entries.contains_key(&id)
            })
            .unwrap();
        let Some(value) = pending.get(&id) else {
            bail!("query IPC reference is missing");
        };
        if value.expires_at <= Instant::now() || rows != value.rows || dimension != value.dimension
        {
            bail!("query IPC reference is expired or has a different shape");
        }
        Ok(Arc::new(pending.remove(&id).unwrap()))
    }

    pub fn prune(&self) {
        let now = Instant::now();
        self.pending
            .lock()
            .unwrap()
            .retain(|_, value| value.expires_at > now);
    }
}

pub fn accept_once(listener: &UnixListener, inbox: &QueryInbox) -> Result<()> {
    match listener.accept() {
        Ok((peer, _)) => inbox.receive(peer),
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(()),
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_pins_until_every_worker_finishes() {
        let inbox = QueryInbox::default();
        let (mut producer, consumer) = UnixStream::pair().unwrap();
        let id = [7; 16];
        producer
            .set_read_timeout(Some(Duration::from_millis(10)))
            .unwrap();
        let mut frame = Vec::from(*b"TQI1");
        frame.extend(id);
        for value in [3_u32, 320, 1920] {
            frame.extend(value.to_be_bytes());
        }
        frame.resize(FRAME_BYTES, 0);
        producer.write_all(&frame).unwrap();
        inbox.receive(consumer).unwrap();
        assert!(inbox.claim(id, 3, 319).is_err());
        let query = inbox.claim(id, 3, 320).unwrap();
        let worker = query.clone();
        drop(query);
        assert!(producer.read(&mut [0]).is_err());
        drop(worker);
        let mut ack = [0];
        producer.read_exact(&mut ack).unwrap();
        assert_eq!(ack, *b"R");
        assert!(inbox.claim(id, 3, 320).is_err());
    }
}
