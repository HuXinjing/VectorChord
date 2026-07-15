use crate::cache::TinyLfu;
use crate::protocol::Descriptor;
use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
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
    shards: HashMap<String, ShardFile>,
    entries: HashMap<String, Entry>,
}

struct HostEntry {
    payload: Arc<[u8]>,
    priority: f64,
    owner_tenant: String,
}

struct HostCache {
    maximum_bytes: usize,
    current_bytes: usize,
    entries: HashMap<String, HostEntry>,
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

    fn get(&mut self, key: &str) -> Option<Arc<[u8]>> {
        let frequency = self.sketch.increment(&key);
        let entry = self.entries.get_mut(key);
        if let Some(entry) = entry {
            entry.priority = self.inflation + f64::from(frequency) / entry.payload.len() as f64;
            self.hits += 1;
            Some(entry.payload.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    fn remove(&mut self, key: &str) -> Option<HostEntry> {
        let entry = self.entries.remove(key)?;
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
                .entries
                .iter()
                .filter(|(_, entry)| entry.owner_tenant == tenant)
                .min_by(|left, right| left.1.priority.total_cmp(&right.1.priority))
                .map(|(key, entry)| (key.clone(), entry.priority))
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
                .entries
                .iter()
                .min_by(|left, right| left.1.priority.total_cmp(&right.1.priority))
                .map(|(key, entry)| (key.clone(), entry.priority))
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
        })
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
            let entry = contract
                .entries
                .get(&descriptor.digest)
                .ok_or_else(|| anyhow!("tensor is missing from the immutable shard index"))?
                .clone();
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
            let worker_count = ranges.len().min(8);
            let ranges_per_worker = ranges.len().div_ceil(worker_count);
            let resolved = std::thread::scope(|scope| -> Result<Vec<_>> {
                let mut workers = Vec::new();
                for ranges in ranges.chunks(ranges_per_worker) {
                    let file = shard.file.try_clone()?;
                    workers.push(scope.spawn(move || -> Result<Vec<(usize, Vec<u8>)>> {
                        let mut resolved = Vec::new();
                        for (start, end, entries) in ranges {
                            let mut range = vec![0_u8; (end - start) as usize];
                            file.read_exact_at(&mut range, *start)?;
                            for (output_index, entry) in entries {
                                let local = (entry.offset - start) as usize;
                                let payload = range[local..local + entry.length].to_vec();
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
                let payload: Arc<[u8]> = payload.into();
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

    pub fn host_status(&self) -> (u64, u64, u64, u64) {
        (
            self.host_cache.hits,
            self.host_cache.misses,
            self.host_cache.evictions,
            self.host_cache.admission_rejections,
        )
    }
}

fn open_contract(root: &Path) -> Result<ContractStore> {
    let index_path = root.join(INDEX_NAME);
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
    Ok(ContractStore { shards, entries })
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
