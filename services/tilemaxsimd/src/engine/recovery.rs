use super::*;
use crate::protocol;
use crate::recovery::Entry;
use anyhow::Context;

impl Engine {
    pub(super) fn record_prepared_recovery(&mut self, descriptors: &[Descriptor]) {
        let Some(records) = self.recovery_entries.as_mut() else {
            return;
        };
        for descriptor in descriptors {
            let key = gpu_cache_key(descriptor, ScoringProfile::ExactFp16, None);
            if let Some(entry) = self
                .devices
                .iter()
                .find_map(|d| d.cache.entry(&key).filter(|e| e.ready))
            {
                if records.len() >= crate::recovery::MAX_ENTRIES && !records.contains_key(&key) {
                    // A later checkpoint fails closed if complete coverage exceeds this bound.
                    continue;
                }
                records.entry(key).or_insert_with(|| Entry {
                    descriptor: {
                        let mut d = descriptor.clone();
                        d.candidate_id = 0;
                        d
                    },
                    tenant: entry.owner_tenant.clone(),
                    profile: ScoringProfile::ExactFp16.cache_tag().to_owned(),
                    quantization_contract: None,
                    query_dtype: 2,
                    pinned: entry.pinned,
                });
            }
        }
    }

    pub fn enable_recovery(&mut self) {
        self.recovery_entries = Some(HashMap::new());
    }
    pub fn recovery_generations(&self) -> Vec<u64> {
        self.devices
            .iter()
            .map(|d| d.cache.durability_generation())
            .collect()
    }
    pub(super) fn record_recovery(&mut self, request: &Request) {
        let Some(records) = self.recovery_entries.as_mut() else {
            return;
        };
        let Some(candidates) = request
            .candidates
            .get(request.candidate_start..request.candidate_end)
        else {
            return;
        };
        for descriptor in candidates {
            let key = gpu_cache_key(
                descriptor,
                request.scoring_profile,
                request.quantization_contract.as_deref(),
            );
            if let Some(entry) = self
                .devices
                .iter()
                .find_map(|d| d.cache.entry(&key).filter(|e| e.ready))
            {
                if records.len() >= crate::recovery::MAX_ENTRIES && !records.contains_key(&key) {
                    // A later checkpoint fails closed if complete coverage exceeds this bound.
                    continue;
                }
                records.entry(key).or_insert_with(|| Entry {
                    descriptor: {
                        let mut d = descriptor.clone();
                        d.candidate_id = 0;
                        d
                    },
                    tenant: entry.owner_tenant.clone(),
                    profile: request.scoring_profile.cache_tag().to_owned(),
                    quantization_contract: request.quantization_contract.clone(),
                    query_dtype: request.dtype,
                    pinned: entry.pinned,
                });
            }
        }
        // Drop evicted metadata at a bounded multiple of the current resident set.
        let count: usize = self.devices.iter().map(|d| d.cache.entry_count()).sum();
        if records.len() > count.saturating_mul(2) {
            records.retain(|key, _| self.devices.iter().any(|d| d.cache.contains(key)));
        }
    }
    pub fn recovery_snapshot(&mut self) -> Result<Vec<Entry>> {
        let records = self
            .recovery_entries
            .as_mut()
            .context("cache recovery tracking is disabled")?;
        records.retain(|key, _| self.devices.iter().any(|d| d.cache.contains(key)));
        let mut output = Vec::new();
        for (key, record) in records.iter() {
            if let Some(current) = self
                .devices
                .iter()
                .find_map(|d| d.cache.entry(key).filter(|e| e.ready))
            {
                let mut value = record.clone();
                value.pinned = current.pinned;
                value.tenant = current.owner_tenant.clone();
                output.push((key.clone(), value));
            }
        }
        let expected: usize = self.devices.iter().map(|d| d.cache.entry_count()).sum();
        if output.len() != expected {
            bail!("cache recovery metadata does not cover the complete resident set");
        }
        output.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(output.into_iter().map(|(_, entry)| entry).collect())
    }
    pub fn restore_recovery(&mut self, entries: &[Entry], batch_size: usize) -> Result<()> {
        if batch_size == 0 {
            bail!("cache recovery batch size must be positive");
        }
        if entries.len() > crate::recovery::MAX_ENTRIES {
            bail!("cache recovery entry limit exceeded");
        }
        let mut identities = HashSet::new();
        for entry in entries {
            crate::recovery::validate(entry)?;
            let key = gpu_cache_key(
                &entry.descriptor,
                crate::recovery::profile(&entry.profile)?,
                entry.quantization_contract.as_deref(),
            );
            if !identities.insert(key) {
                bail!("duplicate logical cache recovery entry");
            }
        }
        self.enable_recovery();
        // Group by logical owner and scoring contract, not a synthetic global tenant.
        let mut groups = std::collections::BTreeMap::<String, Vec<Entry>>::new();
        for entry in entries {
            let identity = serde_json::to_string(&(
                &entry.tenant,
                &entry.profile,
                &entry.quantization_contract,
                entry.query_dtype,
                &entry.descriptor.contract,
                entry.descriptor.dimension,
                entry.descriptor.dtype,
            ))?;
            groups.entry(identity).or_default().push(entry.clone());
        }
        for values in groups.values() {
            for batch in values.chunks(batch_size) {
                let first = &batch[0];
                let profile = crate::recovery::profile(&first.profile)?;
                let width = match first.query_dtype {
                    1 => 4,
                    2 => 2,
                    3 => 1,
                    _ => bail!("invalid cache recovery query dtype"),
                };
                let descriptors: Vec<_> = batch
                    .iter()
                    .enumerate()
                    .map(|(i, e)| {
                        let mut d = e.descriptor.clone();
                        d.candidate_id = i as u32;
                        d
                    })
                    .collect();
                // A one-row zero query reuses the original, validated payload preparation
                // for every profile, including its current quantizer contract. No saved
                // device pointer, reference count or request is replayed.
                let request = Request {
                    protocol_version: protocol::VERSION_EXTERNAL,
                    request_id: 0,
                    tenant: first.tenant.clone(),
                    model_contract: first.descriptor.contract.clone(),
                    priority: 0,
                    timeout_ms: 0,
                    query_rows: 1,
                    dimension: first.descriptor.dimension,
                    dtype: first.query_dtype,
                    candidate_dtype: first.descriptor.dtype,
                    scoring_profile: profile,
                    quantization_contract: first.quantization_contract.clone(),
                    top_k: Some(1),
                    query_reference: None,
                    query_ipc: None,
                    query: vec![0; first.descriptor.dimension as usize * width],
                    candidate_start: 0,
                    candidate_end: descriptors.len(),
                    candidates: Arc::new(descriptors),
                    manifest_digest: None,
                    catalog_digest: None,
                    catalog_selection_digest: None,
                    catalog_public_ids: Vec::new(),
                    catalog_registration: false,
                    scoped_candidate_ordinals: Vec::new(),
                };
                self.score(&request)?;
            }
        }
        let mut keys = HashSet::new();
        for record in entries {
            let key = gpu_cache_key(
                &record.descriptor,
                crate::recovery::profile(&record.profile)?,
                record.quantization_contract.as_deref(),
            );
            if !keys.insert(key.clone()) {
                bail!("duplicate logical cache recovery entry");
            }
            let target = self
                .devices
                .iter_mut()
                .find(|d| d.cache.contains(&key))
                .context("cache recovery exceeds current admission or capacity limits")?;
            let current = target.cache.entry(&key).unwrap();
            if current.owner_tenant != record.tenant {
                bail!("cache recovery owner changed");
            }
            target
                .cache
                .set_pinned(&key, record.pinned)
                .map_err(|e| anyhow!(e))?;
        }
        if self.recovery_snapshot()?.len() != entries.len() {
            bail!("cache recovery did not retain the complete checkpoint");
        }
        Ok(())
    }
}
