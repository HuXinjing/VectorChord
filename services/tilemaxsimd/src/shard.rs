// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// Copyright (c) 2026 Hu Xinjing

use crate::cache::TinyLfu;
use crate::protocol::Descriptor;
use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const INDEX_NAME: &str = "tilemaxsim-shards-v1.json";

#[derive(Deserialize)]
struct RawIndex {
    format: String,
    version: u32,
    alignment: usize,
    shards: Vec<RawShard>,
    entries: Vec<RawEntry>,
}

#[derive(Deserialize)]
struct RawShard {
    name: String,
    #[serde(rename = "bytes")]
    size: usize,
    checksum: String,
}

#[derive(Deserialize)]
struct RawEntry {
    digest: String,
    shard: String,
    offset: u64,
    length: usize,
    rows: u32,
    dimension: u32,
    dtype: String,
}

struct ShardFile {
    file: File,
    size: usize,
    checksum: String,
    verified: bool,
}

#[derive(Clone)]
struct Entry {
    shard: String,
    offset: u64,
    length: usize,
    rows: u32,
    dimension: u32,
    dtype: u8,
}

struct ContractStore {
    root: PathBuf,
    shards: HashMap<String, ShardFile>,
    entries: HashMap<String, Entry>,
}

struct HostEntry {
    payload: Arc<[u8]>,
    priority: f64,
    owner_tenant: String,
}

// Minimum density/age remains the eviction rule; equal priorities use a
// deterministic key tie-break instead of HashMap iteration order.
#[derive(Clone, Debug)]
struct HostRank {
    priority: f64,
    key: String,
}
impl PartialEq for HostRank {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for HostRank {}
impl PartialOrd for HostRank {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HostRank {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority
            .total_cmp(&other.priority)
            .then_with(|| self.key.cmp(&other.key))
    }
}

struct HostCache {
    maximum_bytes: usize,
    current_bytes: usize,
    entries: HashMap<String, HostEntry>,
    ranks: BTreeSet<HostRank>,
    tenant_ranks: HashMap<String, BTreeSet<HostRank>>,
    sketch: TinyLfu,
    inflation: f64,
    tenant_bytes: HashMap<String, usize>,
    default_tenant_max_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub admission_rejections: u64,
}

impl HostCache {
    fn new(maximum_bytes: usize, tenant_max_percent: u8) -> Self {
        Self {
            maximum_bytes,
            current_bytes: 0,
            entries: HashMap::new(),
            ranks: BTreeSet::new(),
            tenant_ranks: HashMap::new(),
            sketch: TinyLfu::new(4096),
            inflation: 0.0,
            tenant_bytes: HashMap::new(),
            default_tenant_max_bytes: maximum_bytes * tenant_max_percent as usize / 100,
            hits: 0,
            misses: 0,
            evictions: 0,
            admission_rejections: 0,
        }
    }

    fn insert_rank(&mut self, key: &str, owner: &str, priority: f64) {
        let rank = HostRank {
            priority,
            key: key.to_owned(),
        };
        self.ranks.insert(rank.clone());
        self.tenant_ranks
            .entry(owner.to_owned())
            .or_default()
            .insert(rank);
    }

    fn remove_rank(&mut self, key: &str, owner: &str, priority: f64) {
        let rank = HostRank {
            priority,
            key: key.to_owned(),
        };
        self.ranks.remove(&rank);
        if let Some(ranks) = self.tenant_ranks.get_mut(owner) {
            ranks.remove(&rank);
            if ranks.is_empty() {
                self.tenant_ranks.remove(owner);
            }
        }
    }

    fn get(&mut self, key: &str) -> Option<Arc<[u8]>> {
        let frequency = self.sketch.increment(&key);
        let Some(entry) = self.entries.get_mut(key) else {
            self.misses += 1;
            return None;
        };
        let old = entry.priority;
        let priority = self.inflation + f64::from(frequency) / entry.payload.len() as f64;
        entry.priority = priority;
        let owner = entry.owner_tenant.clone();
        let payload = Arc::clone(&entry.payload);
        if old.total_cmp(&priority).is_ne() {
            self.remove_rank(key, &owner, old);
            self.insert_rank(key, &owner, priority);
        }
        self.hits += 1;
        Some(payload)
    }

    fn remove(&mut self, key: &str) -> Option<HostEntry> {
        let entry = self.entries.remove(key)?;
        self.remove_rank(key, &entry.owner_tenant, entry.priority);
        self.current_bytes = self.current_bytes.saturating_sub(entry.payload.len());
        let remove_tenant = if let Some(bytes) = self.tenant_bytes.get_mut(&entry.owner_tenant) {
            *bytes = bytes.saturating_sub(entry.payload.len());
            *bytes == 0
        } else {
            false
        };
        if remove_tenant {
            self.tenant_bytes.remove(&entry.owner_tenant);
        }
        Some(entry)
    }

    fn put(&mut self, tenant: &str, key: String, payload: Arc<[u8]>) {
        if self.maximum_bytes == 0 || payload.len() > self.maximum_bytes {
            return;
        }
        self.remove(&key);
        let frequency = self.sketch.estimate(&key).max(1);
        let mut priority = self.inflation + f64::from(frequency) / payload.len() as f64;
        while self
            .tenant_bytes
            .get(tenant)
            .copied()
            .unwrap_or(0)
            .saturating_add(payload.len())
            > self.default_tenant_max_bytes
        {
            let Some((victim_key, victim_priority)) = self
                .tenant_ranks
                .get(tenant)
                .and_then(|ranks| ranks.first())
                .map(|rank| (rank.key.clone(), rank.priority))
            else {
                return;
            };
            if priority <= victim_priority {
                self.admission_rejections += 1;
                return;
            }
            let victim = self
                .remove(&victim_key)
                .expect("host tenant victim disappeared");
            self.inflation = self.inflation.max(victim.priority);
            priority = self.inflation + f64::from(frequency) / payload.len() as f64;
            self.evictions += 1;
        }
        while self.current_bytes + payload.len() > self.maximum_bytes {
            let Some((victim_key, victim_priority)) = self
                .ranks
                .first()
                .map(|rank| (rank.key.clone(), rank.priority))
            else {
                return;
            };
            if priority < victim_priority {
                self.admission_rejections += 1;
                return;
            }
            let victim = self.remove(&victim_key).expect("host victim disappeared");
            self.inflation = self.inflation.max(victim.priority);
            priority = self.inflation + f64::from(frequency) / payload.len() as f64;
            self.evictions += 1;
        }
        self.current_bytes += payload.len();
        *self.tenant_bytes.entry(tenant.to_owned()).or_default() += payload.len();
        self.insert_rank(&key, tenant, priority);
        self.entries.insert(
            key,
            HostEntry {
                payload,
                priority,
                owner_tenant: tenant.to_owned(),
            },
        );
    }
}

pub struct ShardStore {
    contracts: HashMap<String, ContractStore>,
    roots: Vec<(String, PathBuf)>,
    host_cache: HostCache,
    pub batch_read_calls: u64,
    pub batch_read_bytes: u64,
    verify_full_shards: bool,
    read_workers: usize,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct HostCacheStatus {
    pub capacity_bytes: usize,
    pub used_bytes: usize,
    pub entries: usize,
    pub tenants: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub admission_rejections: u64,
}

impl ShardStore {
    pub fn open(
        roots: &[(String, PathBuf)],
        host_cache_bytes: usize,
        host_tenant_max_percent: u8,
        verify_full_shards: bool,
    ) -> Result<Self> {
        let mut contracts = HashMap::new();
        for (contract, root) in roots {
            if contract.is_empty() || contracts.contains_key(contract) {
                bail!("invalid or duplicate model contract {contract:?}");
            }
            contracts.insert(contract.clone(), open_contract(root)?);
        }
        if contracts.is_empty() {
            bail!("at least one immutable shard contract is required");
        }
        Ok(Self {
            contracts,
            roots: roots.to_vec(),
            host_cache: HostCache::new(host_cache_bytes, host_tenant_max_percent),
            batch_read_calls: 0,
            batch_read_bytes: 0,
            verify_full_shards,
            read_workers: std::thread::available_parallelism()
                .map_or(1, usize::from)
                .min(8),
        })
    }

    pub fn set_read_workers(&mut self, workers: usize) -> Result<()> {
        if !(1..=256).contains(&workers) {
            bail!("tensor read workers must be between 1 and 256");
        }
        self.read_workers =
            workers.min(std::thread::available_parallelism().map_or(1, usize::from));
        Ok(())
    }

    pub fn read_workers(&self) -> usize {
        self.read_workers
    }

    /// Routing hint only. Callers must still resolve and validate normally.
    /// No policy get/touch is performed, so choosing a path cannot add hits or
    /// change TinyLFU frequency/ownership. A later miss remains safe to resolve.
    pub fn all_cached(&self, descriptors: &[Descriptor]) -> bool {
        descriptors
            .iter()
            .all(|d| self.host_cache.entries.contains_key(&cache_key(d)))
    }

    // Prefetch only immutable bytes. Policy-changing cache get/put operations
    // still run in original descriptor order in resolve_many, including on errors.
    fn prefetch_objects(
        &mut self,
        descriptors: &[Descriptor],
    ) -> Result<HashMap<String, Result<Arc<[u8]>>>> {
        let mut prefetched = HashMap::new();
        let mut seen = HashSet::new();
        let mut jobs = Vec::new();
        for descriptor in descriptors {
            let key = cache_key(descriptor);
            if !seen.insert(key.clone()) {
                continue;
            }
            if self.host_cache.entries.contains_key(&key) {
                continue;
            }
            let Some(contract) = self.contracts.get(&descriptor.contract) else {
                prefetched.insert(
                    key,
                    Err(anyhow!("model contract has no immutable shard root")),
                );
                continue;
            };
            if !contract.entries.contains_key(&descriptor.digest) {
                jobs.push((key, descriptor, contract.root.as_path()));
            }
        }
        if jobs.is_empty() {
            return Ok(prefetched);
        }
        let count = jobs.len().min(self.read_workers);
        let results = std::thread::scope(|scope| -> Result<Vec<_>> {
            let mut workers = Vec::new();
            for batch in jobs.chunks(jobs.len().div_ceil(count)) {
                workers.push(scope.spawn(move || {
                    batch
                        .iter()
                        .map(|(key, descriptor, root)| {
                            (key.clone(), read_object_payload(root, descriptor))
                        })
                        .collect::<Vec<_>>()
                }));
            }
            let mut out = Vec::new();
            for worker in workers {
                out.extend(
                    worker
                        .join()
                        .map_err(|_| anyhow!("object reader worker panicked"))?,
                );
            }
            Ok(out)
        })?;
        for (key, result) in results {
            if let Ok(payload) = &result {
                self.batch_read_calls += 1;
                self.batch_read_bytes += payload.len() as u64;
            }
            prefetched.insert(key, result);
        }
        Ok(prefetched)
    }

    /// Re-open every immutable contract index and publish the new generation
    /// only after all roots validate. Content-addressed host-cache entries stay
    /// valid across the atomic metadata swap.
    pub fn reload(&mut self) -> Result<()> {
        let mut contracts = HashMap::new();
        for (contract, root) in &self.roots {
            contracts.insert(contract.clone(), open_contract(root)?);
        }
        self.contracts = contracts;
        Ok(())
    }

    pub fn resolve_many(
        &mut self,
        descriptors: &[Descriptor],
        tenant: &str,
    ) -> Result<Vec<Arc<[u8]>>> {
        let mut prefetched = if self.read_workers > 1 {
            Some(self.prefetch_objects(descriptors)?)
        } else {
            None
        };
        let mut output = vec![None; descriptors.len()];
        let mut groups = HashMap::<(String, String), Vec<(usize, Entry)>>::new();
        for (index, descriptor) in descriptors.iter().enumerate() {
            let key = cache_key(descriptor);
            if let Some(payload) = self.host_cache.get(&key) {
                output[index] = Some(payload);
                continue;
            }
            let contract = self
                .contracts
                .get(&descriptor.contract)
                .ok_or_else(|| anyhow!("model contract has no immutable shard root"))?;
            let Some(entry) = contract.entries.get(&descriptor.digest).cloned() else {
                let payload = match prefetched.as_mut() {
                    Some(prefetched) if prefetched.contains_key(&key) => {
                        match prefetched.get(&key).unwrap() {
                            Ok(payload) => Arc::clone(payload),
                            Err(_) => return Err(prefetched.remove(&key).unwrap().unwrap_err()),
                        }
                    }
                    _ => {
                        // A prior admission can evict an item that was cached at
                        // prefetch time. Resolve it normally rather than retaining
                        // an evicted cache entry or bypassing its validation.
                        let payload = read_object_payload(&contract.root, descriptor)?;
                        self.batch_read_calls += 1;
                        self.batch_read_bytes += payload.len() as u64;
                        payload
                    }
                };
                output[index] = Some(Arc::clone(&payload));
                self.host_cache.put(tenant, key, payload);
                continue;
            };
            if entry.rows != descriptor.rows
                || entry.dimension != descriptor.dimension
                || entry.dtype != descriptor.dtype
                || entry.length
                    != tensor_bytes(descriptor.rows, descriptor.dimension, descriptor.dtype)?
            {
                bail!("tensor descriptor disagrees with the immutable shard index");
            }
            groups
                .entry((descriptor.contract.clone(), entry.shard.clone()))
                .or_default()
                .push((index, entry));
        }

        for ((contract_name, shard_name), mut entries) in groups {
            let contract = self.contracts.get_mut(&contract_name).unwrap();
            let shard = contract.shards.get_mut(&shard_name).unwrap();
            if self.verify_full_shards {
                verify_shard(shard)?;
            }
            entries.sort_by_key(|(_, entry)| entry.offset);
            let mut ranges = Vec::<(u64, u64, Vec<(usize, Entry)>)>::new();
            let mut cursor = 0;
            while cursor < entries.len() {
                let start = entries[cursor].1.offset;
                let mut end = start + entries[cursor].1.length as u64;
                let mut limit = cursor + 1;
                while limit < entries.len() {
                    let candidate = &entries[limit].1;
                    let candidate_end = candidate.offset + candidate.length as u64;
                    if candidate.offset.saturating_sub(end) > 64 * 1024
                        || candidate_end.saturating_sub(start) > 8 * 1024 * 1024
                    {
                        break;
                    }
                    end = end.max(candidate_end);
                    limit += 1;
                }
                ranges.push((start, end, entries[cursor..limit].to_vec()));
                cursor = limit;
            }
            let worker_count = ranges.len().min(self.read_workers);
            let ranges_per_worker = ranges.len().div_ceil(worker_count);
            let resolved = std::thread::scope(|scope| -> Result<Vec<_>> {
                let mut workers = Vec::new();
                for ranges in ranges.chunks(ranges_per_worker) {
                    let file = shard.file.try_clone()?;
                    workers.push(scope.spawn(move || -> Result<Vec<(usize, Arc<[u8]>)>> {
                        let mut resolved = Vec::new();
                        for (start, end, entries) in ranges {
                            let mut range = vec![0_u8; (end - start) as usize];
                            file.read_exact_at(&mut range, *start)?;
                            for (output_index, entry) in entries {
                                let local = (entry.offset - start) as usize;
                                let payload =
                                    Arc::<[u8]>::from(&range[local..local + entry.length]);
                                let actual = hex::encode(Sha256::digest(&payload));
                                if actual != descriptors[*output_index].digest {
                                    bail!("tensor checksum mismatch inside immutable shard");
                                }
                                resolved.push((*output_index, payload));
                            }
                        }
                        Ok(resolved)
                    }));
                }
                let mut resolved = Vec::new();
                for worker in workers {
                    resolved.extend(
                        worker
                            .join()
                            .map_err(|_| anyhow!("shard reader worker panicked"))??,
                    );
                }
                Ok(resolved)
            })?;
            self.batch_read_calls += ranges.len() as u64;
            self.batch_read_bytes += ranges
                .iter()
                .map(|(start, end, _)| end - start)
                .sum::<u64>();
            for (output_index, payload) in resolved {
                output[output_index] = Some(Arc::clone(&payload));
                self.host_cache
                    .put(tenant, cache_key(&descriptors[output_index]), payload);
            }
        }
        output
            .into_iter()
            .map(|payload| payload.ok_or_else(|| anyhow!("tensor resolution produced no payload")))
            .collect()
    }

    pub fn host_status(&self) -> HostCacheStatus {
        HostCacheStatus {
            capacity_bytes: self.host_cache.maximum_bytes,
            used_bytes: self.host_cache.current_bytes,
            entries: self.host_cache.entries.len(),
            tenants: self.host_cache.tenant_bytes.len(),
            hits: self.host_cache.hits,
            misses: self.host_cache.misses,
            evictions: self.host_cache.evictions,
            admission_rejections: self.host_cache.admission_rejections,
        }
    }
}

fn open_contract(root: &Path) -> Result<ContractStore> {
    let index_path = root.join(INDEX_NAME);
    let root_metadata = std::fs::symlink_metadata(root)
        .with_context(|| format!("cannot inspect immutable tensor root {}", root.display()))?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        bail!("immutable tensor root must be a real directory");
    }
    if !index_path.try_exists()? {
        return Ok(ContractStore {
            root: root.to_path_buf(),
            shards: HashMap::new(),
            entries: HashMap::new(),
        });
    }
    let mut index_file = nofollow(&index_path)?;
    let mut document = String::new();
    index_file.read_to_string(&mut document)?;
    let raw: RawIndex = serde_json::from_str(&document).context("invalid immutable shard index")?;
    if raw.format != "vectorchord.tilemaxsim.shards" || raw.version != 1 {
        bail!("unsupported immutable shard index");
    }
    if raw.alignment == 0 || !raw.alignment.is_power_of_two() {
        bail!("invalid immutable shard alignment");
    }
    let mut shards = HashMap::new();
    for shard in raw.shards {
        let digest = shard
            .checksum
            .strip_prefix("sha256:")
            .ok_or_else(|| anyhow!("invalid immutable shard checksum"))?;
        if shard.name != format!("shards/sha256-{digest}.vts")
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            bail!("invalid immutable shard name");
        }
        let path = root.join(&shard.name);
        let file = nofollow(&path)?;
        if file.metadata()?.len() != shard.size as u64 {
            bail!("immutable shard size mismatch");
        }
        if shards
            .insert(
                shard.name,
                ShardFile {
                    file,
                    size: shard.size,
                    checksum: digest.to_owned(),
                    verified: false,
                },
            )
            .is_some()
        {
            bail!("duplicate immutable shard");
        }
    }
    let mut entries = HashMap::new();
    for entry in raw.entries {
        let dtype = match entry.dtype.as_str() {
            "float32" => 1,
            "float16" => 2,
            _ => bail!("invalid shard tensor dtype"),
        };
        let shard = shards
            .get(&entry.shard)
            .ok_or_else(|| anyhow!("shard tensor references an unknown file"))?;
        if !(entry.offset as usize).is_multiple_of(raw.alignment)
            || entry.length != tensor_bytes(entry.rows, entry.dimension, dtype)?
            || entry.offset + entry.length as u64 > shard.size as u64
        {
            bail!("invalid shard tensor range");
        }
        if entries
            .insert(
                entry.digest,
                Entry {
                    shard: entry.shard,
                    offset: entry.offset,
                    length: entry.length,
                    rows: entry.rows,
                    dimension: entry.dimension,
                    dtype,
                },
            )
            .is_some()
        {
            bail!("duplicate tensor digest in shard index");
        }
    }
    Ok(ContractStore {
        root: root.to_path_buf(),
        shards,
        entries,
    })
}

fn object_path(root: &Path, digest: &str) -> Result<PathBuf> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        bail!("invalid tensor object digest");
    }
    Ok(root
        .join("objects")
        .join(&digest[..2])
        .join(format!("{digest}.tensor")))
}

fn read_object_payload(root: &Path, descriptor: &Descriptor) -> Result<Arc<[u8]>> {
    let expected = tensor_bytes(descriptor.rows, descriptor.dimension, descriptor.dtype)?;
    let mut file = open_object(root, &descriptor.digest)
        .with_context(|| "tensor is missing from immutable shards and objects")?;
    if file.metadata()?.len() != expected as u64 {
        bail!("immutable tensor object length disagrees with its descriptor");
    }
    // SAFETY: new_zeroed_slice initializes every byte, and every u8 bit pattern
    // is valid. The unique Arc is filled in place before it is ever shared.
    let mut payload = unsafe { Arc::<[u8]>::new_zeroed_slice(expected).assume_init() };
    file.read_exact(Arc::get_mut(&mut payload).expect("new tensor payload is unique"))?;
    let mut trailing = [0u8; 1];
    match file.read_exact(&mut trailing) {
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {}
        Ok(()) => bail!("immutable tensor object length changed while reading"),
        Err(error) => return Err(error.into()),
    }
    if hex::encode(Sha256::digest(&payload)) != descriptor.digest {
        bail!("immutable tensor object checksum mismatch");
    }
    Ok(payload)
}

fn open_object(root: &Path, digest: &str) -> Result<File> {
    let path = object_path(root, digest)?;
    let objects = root.join("objects");
    let prefix = objects.join(&digest[..2]);
    for directory in [&objects, &prefix] {
        let metadata = std::fs::symlink_metadata(directory)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            bail!("immutable tensor object path contains a non-directory component");
        }
    }
    nofollow(&path)
}

fn nofollow(path: &Path) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("cannot open immutable shard path {}", path.display()))
}

fn verify_shard(shard: &mut ShardFile) -> Result<()> {
    if shard.verified {
        return Ok(());
    }
    let mut digest = Sha256::new();
    let mut offset = 0_u64;
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    while offset < shard.size as u64 {
        let count = buffer.len().min(shard.size - offset as usize);
        shard.file.read_exact_at(&mut buffer[..count], offset)?;
        digest.update(&buffer[..count]);
        offset += count as u64;
    }
    if hex::encode(digest.finalize()) != shard.checksum {
        bail!("immutable shard checksum mismatch");
    }
    shard.verified = true;
    Ok(())
}

fn tensor_bytes(rows: u32, dimension: u32, dtype: u8) -> Result<usize> {
    let scalar = match dtype {
        1 => 4,
        2 => 2,
        3 => 1,
        _ => bail!("unsupported tensor dtype"),
    };
    (rows as usize)
        .checked_mul(dimension as usize)
        .and_then(|value| value.checked_mul(scalar))
        .ok_or_else(|| anyhow!("tensor shape overflow"))
}

pub fn cache_key(descriptor: &Descriptor) -> String {
    format!(
        "{}:{}:{}:{}:{}",
        descriptor.contract,
        descriptor.digest,
        descriptor.rows,
        descriptor.dimension,
        descriptor.dtype
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn assert_host_policy(cache: &HostCache) {
        assert!(cache.current_bytes <= cache.maximum_bytes);
        assert_eq!(
            cache.current_bytes,
            cache
                .entries
                .values()
                .map(|e| e.payload.len())
                .sum::<usize>()
        );
        assert_eq!(cache.ranks.len(), cache.entries.len());
        assert_eq!(
            cache
                .tenant_ranks
                .values()
                .map(BTreeSet::len)
                .sum::<usize>(),
            cache.entries.len()
        );
        let mut owners = HashMap::<String, usize>::new();
        for (key, value) in &cache.entries {
            *owners.entry(value.owner_tenant.clone()).or_default() += value.payload.len();
            let rank = HostRank {
                priority: value.priority,
                key: key.clone(),
            };
            assert!(cache.ranks.contains(&rank));
            assert!(cache.tenant_ranks[&value.owner_tenant].contains(&rank));
        }
        assert_eq!(owners, cache.tenant_bytes);
        for (owner, bytes) in &owners {
            assert!(*bytes <= cache.default_tenant_max_bytes);
            let minimum = cache
                .entries
                .values()
                .filter(|e| &e.owner_tenant == owner)
                .map(|e| e.priority)
                .min_by(f64::total_cmp)
                .unwrap();
            assert_eq!(cache.tenant_ranks[owner].first().unwrap().priority, minimum);
        }
        assert_eq!(
            cache.ranks.first().map(|r| r.priority),
            cache
                .entries
                .values()
                .map(|e| e.priority)
                .min_by(f64::total_cmp)
        );
    }

    #[test]
    fn host_eviction_indexes_remain_bounded_through_access_replacement_and_removal() {
        let mut cache = HostCache::new(1024, 60);
        let mut random = 918273_u64;
        for _ in 0..5000 {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            let key = format!("object-{}", (random >> 8) % 97);
            let owner = format!("owner-{}", (random >> 24) % 3);
            match random % 5 {
                0 => {
                    cache.remove(&key);
                }
                1 | 2 => {
                    cache.get(&key);
                }
                _ => {
                    cache.put(
                        &owner,
                        key,
                        Arc::from(vec![7; 1 + ((random >> 32) % 127) as usize]),
                    );
                }
            }
            assert_host_policy(&cache);
        }
    }

    #[test]
    fn host_tenant_pressure_preserves_other_owners_and_borrowed_payloads() {
        let mut cache = HostCache::new(100, 50);
        cache.put("a", "old-a".into(), Arc::from(vec![1; 30]));
        cache.put("b", "old-b".into(), Arc::from(vec![2; 30]));
        let held = cache.get("old-a").unwrap();
        for _ in 0..4 {
            cache.get("new-a");
        }
        cache.put("a", "new-a".into(), Arc::from(vec![3; 40]));
        assert!(!cache.entries.contains_key("old-a"));
        assert_eq!(&*held, &[1; 30]);
        assert_eq!(&*cache.get("old-b").unwrap(), &[2; 30]);
        assert_host_policy(&cache);
        cache.put("c", "new-a".into(), Arc::from(vec![4; 20]));
        assert!(!cache.tenant_ranks.contains_key("a"));
        assert_eq!(cache.entries["new-a"].owner_tenant, "c");
        assert_host_policy(&cache);
    }

    struct Objects {
        root: PathBuf,
        descriptors: Vec<Descriptor>,
    }
    impl Objects {
        fn new(count: usize) -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "tilemaxsim-parallel-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            let descriptors = (0..count)
                .map(|i| {
                    let rows = 1 + (i % 7) as u32;
                    let payload: Vec<_> = (0..rows * 2)
                        .flat_map(|j| ((i as u16 * 113 + j as u16) % 1024).to_le_bytes())
                        .collect();
                    let digest = hex::encode(Sha256::digest(&payload));
                    let parent = root.join("objects").join(&digest[..2]);
                    fs::create_dir_all(&parent).unwrap();
                    fs::write(parent.join(format!("{digest}.tensor")), payload).unwrap();
                    Descriptor {
                        candidate_id: i as u32,
                        contract: "objects@1".into(),
                        digest,
                        rows,
                        dimension: 2,
                        dtype: 2,
                        raw_fp8_cache_key: Default::default(),
                    }
                })
                .collect();
            Self { root, descriptors }
        }
        fn store(&self, workers: usize, bytes: usize) -> ShardStore {
            let mut store =
                ShardStore::open(&[("objects@1".into(), self.root.clone())], bytes, 60, false)
                    .unwrap();
            // Exercise concurrency even on a single-CPU CI runner.
            store.read_workers = workers;
            store
        }
    }
    impl Drop for Objects {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
    #[test]
    fn configured_read_workers_preserve_cpu_bound() {
        let objects = Objects::new(1);
        let mut store = objects.store(1, 1024);
        for workers in [1, 32, 256] {
            store.set_read_workers(workers).unwrap();
            assert_eq!(
                store.read_workers(),
                workers.min(std::thread::available_parallelism().map_or(1, usize::from))
            );
        }
        for workers in [0, 257, usize::MAX] {
            let before = store.read_workers();
            assert!(store.set_read_workers(workers).is_err());
            assert_eq!(store.read_workers(), before);
        }
    }

    fn same_policy(a: &HostCache, b: &HostCache) {
        assert_host_policy(a);
        assert_host_policy(b);
        assert_eq!(
            (
                a.current_bytes,
                a.hits,
                a.misses,
                a.evictions,
                a.admission_rejections
            ),
            (
                b.current_bytes,
                b.hits,
                b.misses,
                b.evictions,
                b.admission_rejections
            )
        );
        assert_eq!(a.tenant_bytes, b.tenant_bytes);
        assert_eq!(a.entries.len(), b.entries.len());
        for (key, value) in &a.entries {
            let other = &b.entries[key];
            assert_eq!(value.owner_tenant, other.owner_tenant);
            assert_eq!(value.priority.to_bits(), other.priority.to_bits());
            assert_eq!(value.payload, other.payload);
        }
    }
    #[test]
    fn parallel_object_reads_preserve_results_and_ordered_tenant_policy() {
        let objects = Objects::new(32);
        let mut serial = objects.store(1, 192);
        let mut parallel = objects.store(4, 192);
        for round in 0..20 {
            let descriptors: Vec<_> = (0..24)
                .map(|i| objects.descriptors[(i * 7 + round * 3) % 32].clone())
                .collect();
            let owner = format!("owner-{}", round % 3);
            let expected = serial.resolve_many(&descriptors, &owner).unwrap();
            let actual = parallel.resolve_many(&descriptors, &owner).unwrap();
            assert_eq!(actual, expected);
            same_policy(&serial.host_cache, &parallel.host_cache);
        }
    }
    #[test]
    fn parallel_object_corruption_preserves_error_and_prior_cache_admissions() {
        for truncate in [false, true] {
            let objects = Objects::new(8);
            let bad = &objects.descriptors[3];
            let file = objects
                .root
                .join("objects")
                .join(&bad.digest[..2])
                .join(format!("{}.tensor", bad.digest));
            if truncate {
                fs::write(file, [0u8]).unwrap();
            } else {
                let bytes = fs::read(&file).unwrap();
                fs::write(file, vec![0xff; bytes.len()]).unwrap();
            }
            let mut serial = objects.store(1, 4096);
            let mut parallel = objects.store(4, 4096);
            let a = serial
                .resolve_many(&objects.descriptors, "owner")
                .unwrap_err();
            let b = parallel
                .resolve_many(&objects.descriptors, "owner")
                .unwrap_err();
            assert_eq!(a.to_string(), b.to_string());
            assert_eq!(serial.host_cache.entries.len(), 3);
            same_policy(&serial.host_cache, &parallel.host_cache);
        }
    }
    #[test]
    fn cached_item_evicted_during_resolution_is_revalidated_if_not_prefetched() {
        let objects = Objects::new(8);
        let future = objects.descriptors[6].clone();
        let mut serial = objects.store(1, 64);
        let mut parallel = objects.store(4, 64);
        serial
            .resolve_many(std::slice::from_ref(&future), "a")
            .unwrap();
        parallel
            .resolve_many(std::slice::from_ref(&future), "a")
            .unwrap();
        fs::remove_file(
            objects
                .root
                .join("objects")
                .join(&future.digest[..2])
                .join(format!("{}.tensor", future.digest)),
        )
        .unwrap();
        let input = [
            objects.descriptors[0].clone(),
            objects.descriptors[1].clone(),
            future,
        ];
        assert_eq!(
            serial.resolve_many(&input, "a").unwrap_err().to_string(),
            parallel.resolve_many(&input, "a").unwrap_err().to_string()
        );
        same_policy(&serial.host_cache, &parallel.host_cache);
    }

    #[test]
    fn parallel_objects_preserve_all_canonical_storage_types() {
        let objects = Objects::new(8);
        for (dtype, dimension) in [(1, 1), (2, 2), (3, 4)] {
            let descriptors: Vec<_> = objects
                .descriptors
                .iter()
                .map(|d| {
                    let mut value = d.clone();
                    value.dtype = dtype;
                    value.dimension = dimension;
                    value
                })
                .collect();
            let expected = objects
                .store(1, 4096)
                .resolve_many(&descriptors, "owner")
                .unwrap();
            let actual = objects
                .store(4, 4096)
                .resolve_many(&descriptors, "owner")
                .unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn mixed_packed_and_loose_objects_keep_payloads_and_corruption_checks() {
        let objects = Objects::new(8);
        let mut packed = Vec::new();
        let mut entries = Vec::new();
        for d in &objects.descriptors[..4] {
            let offset = packed.len().next_multiple_of(16);
            packed.resize(offset, 0);
            let payload = fs::read(
                objects
                    .root
                    .join("objects")
                    .join(&d.digest[..2])
                    .join(format!("{}.tensor", d.digest)),
            )
            .unwrap();
            entries.push(serde_json::json!({"digest":d.digest,"offset":offset,"length":payload.len(),"rows":d.rows,"dimension":d.dimension,"dtype":"float16"}));
            packed.extend(payload);
        }
        let digest = hex::encode(Sha256::digest(&packed));
        let name = format!("shards/sha256-{digest}.vts");
        fs::create_dir(objects.root.join("shards")).unwrap();
        fs::write(objects.root.join(&name), &packed).unwrap();
        for entry in &mut entries {
            entry["shard"] = name.clone().into();
        }
        let index = serde_json::json!({"format":"vectorchord.tilemaxsim.shards","version":1,"alignment":16,
            "shards":[{"name":name,"bytes":packed.len(),"checksum":format!("sha256:{digest}")}],"entries":entries});
        fs::write(
            objects.root.join(INDEX_NAME),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        let mut serial = objects.store(1, 128);
        let mut parallel = objects.store(4, 128);
        let a = serial.resolve_many(&objects.descriptors, "owner").unwrap();
        let b = parallel
            .resolve_many(&objects.descriptors, "owner")
            .unwrap();
        assert_eq!(a, b);
        same_policy(&serial.host_cache, &parallel.host_cache);
        packed[0] ^= 0xff;
        fs::write(objects.root.join(&name), packed).unwrap();
        assert!(
            objects
                .store(4, 128)
                .resolve_many(&objects.descriptors, "owner")
                .unwrap_err()
                .to_string()
                .contains("checksum mismatch")
        );
    }

    #[test]
    fn object_reader_checks_symlinks_length_and_cached_immutable_reuse() {
        let objects = Objects::new(2);
        let descriptor = objects.descriptors[0].clone();
        let path = objects
            .root
            .join("objects")
            .join(&descriptor.digest[..2])
            .join(format!("{}.tensor", descriptor.digest));
        let mut store = objects.store(4, 4096);
        let expected = store
            .resolve_many(std::slice::from_ref(&descriptor), "a")
            .unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(
            store
                .resolve_many(std::slice::from_ref(&descriptor), "b")
                .unwrap(),
            expected
        );
        let target = objects.root.join("other");
        fs::write(&target, &*expected[0]).unwrap();
        std::os::unix::fs::symlink(target, &path).unwrap();
        let mut empty = objects.store(4, 4096);
        assert!(empty.resolve_many(&[descriptor], "a").is_err());
    }

    #[test]
    fn tensor_length_matches_all_protocol_storage_types() {
        assert_eq!(tensor_bytes(3, 5, 1).unwrap(), 60);
        assert_eq!(tensor_bytes(3, 5, 2).unwrap(), 30);
        assert_eq!(tensor_bytes(3, 5, 3).unwrap(), 15);
        assert!(tensor_bytes(3, 5, 4).is_err());
    }

    #[test]
    fn raw_fp8_objects_resolve_and_reject_incorrect_shape() {
        let root = std::env::temp_dir().join(format!(
            "tilemaxsim-raw-fp8-object-store-{}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let payload = [0x38_u8, 0x00, 0xb8, 0x30];
        let digest = hex::encode(Sha256::digest(payload));
        let directory = root.join("objects").join(&digest[..2]);
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join(format!("{digest}.tensor")), payload).unwrap();
        let mut store =
            ShardStore::open(&[("model@1".to_owned(), root.clone())], 1024, 100, false).unwrap();
        let descriptor = || Descriptor {
            candidate_id: 1,
            contract: "model@1".to_owned(),
            digest: digest.clone(),
            rows: 2,
            dimension: 2,
            dtype: 3,
            raw_fp8_cache_key: std::sync::OnceLock::new(),
        };
        let mut invalid = descriptor();
        invalid.rows = 3;
        assert!(store.resolve_many(&[invalid], "tenant-a").is_err());
        let resolved = store.resolve_many(&[descriptor()], "tenant-a").unwrap();
        assert_eq!(&*resolved[0], &payload);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn content_addressed_objects_are_visible_without_index_reload() {
        let root =
            std::env::temp_dir().join(format!("tilemaxsim-object-store-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mut store =
            ShardStore::open(&[("model@1".to_owned(), root.clone())], 1024, 100, false).unwrap();
        let payload = [0_u8, 60, 0, 0];
        let digest = hex::encode(Sha256::digest(payload));
        let directory = root.join("objects").join(&digest[..2]);
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join(format!("{digest}.tensor")), payload).unwrap();
        let descriptor = Descriptor {
            candidate_id: 1,
            contract: "model@1".to_owned(),
            digest,
            rows: 1,
            dimension: 2,
            dtype: 2,
            raw_fp8_cache_key: std::sync::OnceLock::new(),
        };

        let resolved = store.resolve_many(&[descriptor], "tenant-a").unwrap();
        assert_eq!(&*resolved[0], &payload);
        fs::remove_dir_all(root).unwrap();
    }
}
