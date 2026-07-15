// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// Copyright (c) 2026 Hu Xinjing

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};

#[derive(Debug)]
pub struct PageRunAllocator {
    block_bytes: usize,
    block_count: usize,
    free_by_start: BTreeMap<usize, usize>,
    free_by_size: BTreeMap<usize, BTreeSet<usize>>,
    allocated: HashMap<usize, usize>,
    free_blocks: usize,
}

impl PageRunAllocator {
    pub fn new(capacity: usize, block_bytes: usize) -> Result<Self, &'static str> {
        if capacity == 0 || block_bytes == 0 || !block_bytes.is_multiple_of(256) {
            return Err("invalid fixed-block arena");
        }
        let block_count = capacity / block_bytes;
        if block_count == 0 {
            return Err("fixed-block arena is too small");
        }
        let mut allocator = Self {
            block_bytes,
            block_count,
            free_by_start: BTreeMap::new(),
            free_by_size: BTreeMap::new(),
            allocated: HashMap::new(),
            free_blocks: 0,
        };
        allocator.add_free_run(0, block_count)?;
        Ok(allocator)
    }

    fn add_free_run(&mut self, start: usize, blocks: usize) -> Result<(), &'static str> {
        if blocks == 0
            || start
                .checked_add(blocks)
                .is_none_or(|end| end > self.block_count)
        {
            return Err("free GPU page run is outside the arena");
        }
        if self.free_by_start.insert(start, blocks).is_some() {
            return Err("duplicate free GPU page run");
        }
        self.free_by_size.entry(blocks).or_default().insert(start);
        self.free_blocks = self
            .free_blocks
            .checked_add(blocks)
            .ok_or("free GPU page count overflow")?;
        Ok(())
    }

    fn remove_free_run(&mut self, start: usize, blocks: usize) -> Result<(), &'static str> {
        if self.free_by_start.remove(&start) != Some(blocks) {
            return Err("free GPU address index is inconsistent");
        }
        let remove_size = {
            let starts = self
                .free_by_size
                .get_mut(&blocks)
                .ok_or("free GPU size index is inconsistent")?;
            if !starts.remove(&start) {
                return Err("free GPU size index is inconsistent");
            }
            starts.is_empty()
        };
        if remove_size {
            self.free_by_size.remove(&blocks);
        }
        self.free_blocks = self
            .free_blocks
            .checked_sub(blocks)
            .ok_or("free GPU page count underflow")?;
        Ok(())
    }

    pub fn capacity(&self) -> usize {
        self.block_count * self.block_bytes
    }

    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    pub fn allocation_bytes(&self, payload_bytes: usize) -> Option<usize> {
        if payload_bytes == 0 {
            return None;
        }
        payload_bytes
            .div_ceil(self.block_bytes)
            .checked_mul(self.block_bytes)
    }

    pub fn largest_free(&self) -> usize {
        self.free_by_size
            .last_key_value()
            .map_or(0, |(blocks, _)| blocks * self.block_bytes)
    }

    pub fn allocate(&mut self, payload_bytes: usize) -> Option<(usize, usize)> {
        let allocation_bytes = self.allocation_bytes(payload_bytes)?;
        let required = allocation_bytes / self.block_bytes;
        let (&available, starts) = self.free_by_size.range(required..).next()?;
        let start = *starts.first()?;
        self.remove_free_run(start, available).ok()?;
        if available > required {
            self.add_free_run(start + required, available - required)
                .ok()?;
        }
        self.allocated.insert(start, required);
        Some((start * self.block_bytes, allocation_bytes))
    }

    pub fn release(&mut self, offset: usize) -> Result<(), &'static str> {
        if !offset.is_multiple_of(self.block_bytes) {
            return Err("unaligned fixed-block release");
        }
        let mut start = offset / self.block_bytes;
        let mut blocks = self
            .allocated
            .remove(&start)
            .ok_or("fixed-block page run was released twice")?;
        let previous = self
            .free_by_start
            .range(..start)
            .next_back()
            .map(|(&previous_start, &previous_blocks)| (previous_start, previous_blocks));
        if let Some((previous_start, previous_blocks)) = previous
            && previous_start + previous_blocks == start
        {
            self.remove_free_run(previous_start, previous_blocks)?;
            start = previous_start;
            blocks += previous_blocks;
        }
        let next_start = start + blocks;
        if let Some(&next_blocks) = self.free_by_start.get(&next_start) {
            self.remove_free_run(next_start, next_blocks)?;
            blocks += next_blocks;
        }
        self.add_free_run(start, blocks)
    }

    pub fn free_bytes(&self) -> usize {
        self.free_blocks * self.block_bytes
    }

    pub fn allocated_bytes(&self) -> usize {
        self.allocated
            .values()
            .map(|blocks| blocks * self.block_bytes)
            .sum()
    }

    #[cfg(test)]
    pub fn validate(&self) -> Result<(), &'static str> {
        let address_blocks = self.free_by_start.values().sum::<usize>();
        let size_blocks = self
            .free_by_size
            .iter()
            .map(|(blocks, starts)| blocks * starts.len())
            .sum::<usize>();
        if address_blocks != self.free_blocks || size_blocks != self.free_blocks {
            return Err("free GPU page accounting is inconsistent");
        }
        for (&start, &blocks) in &self.free_by_start {
            if !self
                .free_by_size
                .get(&blocks)
                .is_some_and(|starts| starts.contains(&start))
            {
                return Err("free GPU page indexes disagree");
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct TinyLfu {
    width: usize,
    tables: Vec<Vec<u16>>,
    samples: usize,
}

impl TinyLfu {
    pub fn new(width: usize) -> Self {
        Self {
            width,
            tables: vec![vec![0; width]; 4],
            samples: 0,
        }
    }

    fn indices<T: Hash>(&self, key: &T) -> [usize; 4] {
        std::array::from_fn(|row| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            row.hash(&mut hasher);
            key.hash(&mut hasher);
            hasher.finish() as usize % self.width
        })
    }

    pub fn increment<T: Hash>(&mut self, key: &T) -> u16 {
        let indices = self.indices(key);
        for (row, index) in indices.into_iter().enumerate() {
            self.tables[row][index] = self.tables[row][index].saturating_add(1);
        }
        self.samples += 1;
        let estimate = self.estimate(key).max(1);
        if self.samples >= self.width * 10 {
            for table in &mut self.tables {
                for value in table {
                    *value /= 2;
                }
            }
            self.samples /= 2;
        }
        estimate
    }

    pub fn estimate<T: Hash>(&self, key: &T) -> u16 {
        self.indices(key)
            .into_iter()
            .enumerate()
            .map(|(row, index)| self.tables[row][index])
            .min()
            .unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
pub struct CacheEntry {
    pub offset: u64,
    pub allocated_bytes: usize,
    pub payload_bytes: usize,
    pub rows: u32,
    pub dimension: u32,
    pub dtype: u8,
    pub references: usize,
    pub pinned: bool,
    /// False while an H2D upload is in progress. Loading entries reserve pages
    /// but are never visible as cache hits.
    pub ready: bool,
    owner_tenant: String,
    priority: f64,
}

#[derive(Debug, PartialEq)]
pub enum Admission {
    Admitted { offset: u64, allocated_bytes: usize },
    Rejected,
    Deferred,
}

pub struct GpuCache {
    allocator: PageRunAllocator,
    entries: HashMap<String, CacheEntry>,
    sketch: TinyLfu,
    inflation: f64,
    tenant_allocated: HashMap<String, usize>,
    tenant_reservations: HashMap<String, usize>,
    default_tenant_max_bytes: usize,
    pinned_max_bytes: usize,
    pinned_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub admission_rejections: u64,
}

impl GpuCache {
    #[cfg(test)]
    pub fn new(capacity: usize, block_bytes: usize) -> Result<Self, &'static str> {
        Self::new_with_limits(capacity, block_bytes, 100, 100, HashMap::new())
    }

    pub fn new_with_limits(
        capacity: usize,
        block_bytes: usize,
        tenant_max_percent: u8,
        pinned_max_percent: u8,
        tenant_reservations: HashMap<String, usize>,
    ) -> Result<Self, &'static str> {
        if tenant_max_percent == 0
            || tenant_max_percent > 100
            || pinned_max_percent > 100
            || tenant_reservations.values().sum::<usize>() > capacity
        {
            return Err("invalid GPU tenant cache policy");
        }
        Ok(Self {
            allocator: PageRunAllocator::new(capacity, block_bytes)?,
            entries: HashMap::new(),
            sketch: TinyLfu::new(4096),
            inflation: 0.0,
            tenant_allocated: HashMap::new(),
            tenant_reservations,
            default_tenant_max_bytes: capacity * tenant_max_percent as usize / 100,
            pinned_max_bytes: capacity * pinned_max_percent as usize / 100,
            pinned_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            admission_rejections: 0,
        })
    }

    pub fn capacity(&self) -> usize {
        self.allocator.capacity()
    }

    pub fn block_bytes(&self) -> usize {
        self.allocator.block_bytes()
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn pinned_entries(&self) -> usize {
        self.entries.values().filter(|entry| entry.pinned).count()
    }

    pub fn tenant_count(&self) -> usize {
        self.tenant_allocated.len()
    }

    pub fn pinned_bytes(&self) -> usize {
        self.pinned_bytes
    }

    pub fn free_bytes(&self) -> usize {
        self.allocator.free_bytes()
    }

    pub fn largest_free_extent(&self) -> usize {
        self.allocator.largest_free()
    }

    pub fn allocated_bytes(&self) -> usize {
        self.allocator.allocated_bytes()
    }

    pub fn payload_bytes(&self) -> usize {
        self.entries.values().map(|entry| entry.payload_bytes).sum()
    }

    pub fn get(&mut self, key: &str) -> Option<CacheEntry> {
        let frequency = self.sketch.increment(&key);
        let Some(entry) = self.entries.get_mut(key).filter(|entry| entry.ready) else {
            self.misses += 1;
            return None;
        };
        entry.references += 1;
        entry.priority = self.inflation + f64::from(frequency) / entry.allocated_bytes as f64;
        self.hits += 1;
        Some(entry.clone())
    }

    pub fn acquire_existing(&mut self, key: &str) -> Option<CacheEntry> {
        let entry = self.entries.get_mut(key).filter(|entry| entry.ready)?;
        entry.references += 1;
        Some(entry.clone())
    }

    pub fn record_access_miss(&mut self, key: &str) {
        self.sketch.increment(&key);
        self.misses += 1;
    }

    pub fn contains(&self, key: &str) -> bool {
        self.entries.get(key).is_some_and(|entry| entry.ready)
    }

    fn victim(&self) -> Option<(&String, &CacheEntry)> {
        self.entries
            .iter()
            .filter(|(_, entry)| self.can_evict(entry))
            .min_by(|left, right| left.1.priority.total_cmp(&right.1.priority))
    }

    fn tenant_victim(&self, tenant: &str) -> Option<(&String, &CacheEntry)> {
        self.entries
            .iter()
            .filter(|(_, entry)| {
                entry.owner_tenant == tenant && entry.references == 0 && !entry.pinned
            })
            .min_by(|left, right| left.1.priority.total_cmp(&right.1.priority))
    }

    fn can_evict(&self, entry: &CacheEntry) -> bool {
        if entry.references != 0 || entry.pinned {
            return false;
        }
        let usage = self
            .tenant_allocated
            .get(&entry.owner_tenant)
            .copied()
            .unwrap_or(0);
        let reservation = self
            .tenant_reservations
            .get(&entry.owner_tenant)
            .copied()
            .unwrap_or(0);
        usage.saturating_sub(entry.allocated_bytes) >= reservation
    }

    fn remove_entry(&mut self, key: &str) -> Result<CacheEntry, &'static str> {
        let entry = self
            .entries
            .remove(key)
            .ok_or("GPU cache entry disappeared")?;
        self.allocator.release(entry.offset as usize)?;
        if entry.pinned {
            self.pinned_bytes = self.pinned_bytes.saturating_sub(entry.allocated_bytes);
        }
        let remove_tenant = if let Some(usage) = self.tenant_allocated.get_mut(&entry.owner_tenant)
        {
            *usage = usage.saturating_sub(entry.allocated_bytes);
            *usage == 0
        } else {
            false
        };
        if remove_tenant {
            self.tenant_allocated.remove(&entry.owner_tenant);
        }
        Ok(entry)
    }

    fn evict_one(&mut self) -> bool {
        let Some(key) = self.victim().map(|(key, _)| key.clone()) else {
            return false;
        };
        let entry = self.remove_entry(&key).expect("victim disappeared");
        self.inflation = self.inflation.max(entry.priority);
        self.evictions += 1;
        true
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    pub fn admit(
        &mut self,
        key: String,
        payload_bytes: usize,
        rows: u32,
        dimension: u32,
        dtype: u8,
        pinned: bool,
        force: bool,
    ) -> Admission {
        self.admit_for_tenant(
            "__default__",
            key,
            payload_bytes,
            rows,
            dimension,
            dtype,
            pinned,
            force,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn admit_for_tenant(
        &mut self,
        tenant: &str,
        key: String,
        payload_bytes: usize,
        rows: u32,
        dimension: u32,
        dtype: u8,
        pinned: bool,
        force: bool,
    ) -> Admission {
        let Some(allocation_bytes) = self.allocator.allocation_bytes(payload_bytes) else {
            return Admission::Deferred;
        };
        if allocation_bytes > self.allocator.capacity() {
            return Admission::Deferred;
        }
        if pinned && self.pinned_bytes.saturating_add(allocation_bytes) > self.pinned_max_bytes {
            return Admission::Deferred;
        }
        if !pinned {
            let tenant_max_bytes = self
                .tenant_reservations
                .get(tenant)
                .copied()
                .unwrap_or(0)
                .max(self.default_tenant_max_bytes);
            while self
                .tenant_allocated
                .get(tenant)
                .copied()
                .unwrap_or(0)
                .saturating_add(allocation_bytes)
                > tenant_max_bytes
            {
                let Some((victim_key, victim_priority)) = self
                    .tenant_victim(tenant)
                    .map(|(victim_key, victim)| (victim_key.clone(), victim.priority))
                else {
                    return Admission::Deferred;
                };
                let candidate = self.inflation
                    + f64::from(self.sketch.estimate(&key).max(1)) / allocation_bytes as f64;
                if !force && candidate <= victim_priority {
                    self.admission_rejections += 1;
                    return Admission::Rejected;
                }
                let entry = self
                    .remove_entry(&victim_key)
                    .expect("tenant victim disappeared");
                self.inflation = self.inflation.max(entry.priority);
                self.evictions += 1;
            }
        }
        while self.allocator.largest_free() < allocation_bytes {
            let Some((_, victim)) = self.victim() else {
                return Admission::Deferred;
            };
            let candidate = self.inflation
                + f64::from(self.sketch.estimate(&key).max(1)) / allocation_bytes as f64;
            if !force && !pinned && candidate <= victim.priority {
                self.admission_rejections += 1;
                return Admission::Rejected;
            }
            if !self.evict_one() {
                return Admission::Deferred;
            }
        }
        let Some((offset, allocated_bytes)) = self.allocator.allocate(payload_bytes) else {
            return Admission::Deferred;
        };
        let priority =
            self.inflation + f64::from(self.sketch.estimate(&key).max(1)) / allocated_bytes as f64;
        self.entries.insert(
            key,
            CacheEntry {
                offset: offset as u64,
                allocated_bytes,
                payload_bytes,
                rows,
                dimension,
                dtype,
                references: 1,
                pinned,
                ready: false,
                owner_tenant: tenant.to_owned(),
                priority,
            },
        );
        *self.tenant_allocated.entry(tenant.to_owned()).or_default() += allocated_bytes;
        if pinned {
            self.pinned_bytes += allocated_bytes;
        }
        Admission::Admitted {
            offset: offset as u64,
            allocated_bytes,
        }
    }

    pub fn release(&mut self, key: &str) -> Result<(), &'static str> {
        let entry = self
            .entries
            .get_mut(key)
            .ok_or("GPU cache entry disappeared")?;
        if entry.references == 0 {
            return Err("GPU cache entry was released twice");
        }
        entry.references -= 1;
        Ok(())
    }

    pub fn mark_ready(&mut self, key: &str) -> Result<(), &'static str> {
        let entry = self
            .entries
            .get_mut(key)
            .ok_or("GPU cache entry disappeared before upload completed")?;
        entry.ready = true;
        Ok(())
    }

    #[cfg(test)]
    pub fn entry(&self, key: &str) -> Option<&CacheEntry> {
        self.entries.get(key)
    }

    pub fn remove(&mut self, key: &str) -> Result<(), &'static str> {
        self.remove_entry(key).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_test_random(state: &mut u64) -> u64 {
        // Deterministic xorshift64 sequence for allocator churn tests. This is
        // test data, not a pointer, address, secret, or production RNG.
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn page_runs_use_exact_blocks_and_coalesce() {
        let mut allocator = PageRunAllocator::new(8 * 256, 256).unwrap();
        let first = allocator.allocate(300).unwrap();
        let second = allocator.allocate(300).unwrap();
        let third = allocator.allocate(700).unwrap();
        assert_eq!(first, (0, 512));
        assert_eq!(second, (512, 512));
        assert_eq!(third, (1024, 768));
        assert_eq!(allocator.free_bytes(), 256);
        allocator.release(second.0).unwrap();
        allocator.release(first.0).unwrap();
        assert_eq!(allocator.largest_free(), 1024);
        allocator.release(third.0).unwrap();
        assert_eq!(allocator.largest_free(), 2048);
        allocator.validate().unwrap();
    }

    #[test]
    fn page_run_indexes_stay_consistent_under_churn() {
        let mut allocator = PageRunAllocator::new(64 * 256, 256).unwrap();
        let mut live = Vec::new();
        let mut state = 0x5eed_u64;
        for _ in 0..20_000 {
            let random = next_test_random(&mut state);
            if !live.is_empty() && random & 3 == 0 {
                let index = random as usize % live.len();
                let (offset, _) = live.swap_remove(index);
                allocator.release(offset).unwrap();
            } else {
                let payload = ((random >> 16) as usize % (12 * 256)) + 1;
                if let Some(allocation) = allocator.allocate(payload) {
                    live.push(allocation);
                }
            }
            allocator.validate().unwrap();
            assert_eq!(
                allocator.free_bytes() + allocator.allocated_bytes(),
                allocator.capacity()
            );
        }
    }

    #[test]
    fn tinylfu_protects_hot_entry() {
        let mut cache = GpuCache::new(256, 256).unwrap();
        cache.record_access_miss("hot");
        assert!(matches!(
            cache.admit("hot".into(), 64, 1, 32, 2, false, false),
            Admission::Admitted { .. }
        ));
        assert!(
            cache.get("hot").is_none(),
            "loading pages must not be visible"
        );
        cache.mark_ready("hot").unwrap();
        cache.release("hot").unwrap();
        for _ in 0..3 {
            cache.get("hot").unwrap();
            cache.release("hot").unwrap();
        }
        cache.record_access_miss("cold");
        assert_eq!(
            cache.admit("cold".into(), 64, 1, 32, 2, false, false),
            Admission::Rejected
        );
        assert!(cache.entry("hot").is_some());
    }

    #[test]
    fn failed_loading_entry_can_be_rolled_back_without_leaking_pages() {
        let mut cache = GpuCache::new(4 * 256, 256).unwrap();
        assert!(matches!(
            cache.admit("loading".into(), 300, 1, 150, 2, false, true),
            Admission::Admitted { .. }
        ));
        assert!(!cache.contains("loading"));
        assert_eq!(cache.allocated_bytes(), 512);
        cache.remove("loading").unwrap();
        assert_eq!(cache.allocated_bytes(), 0);
        assert_eq!(cache.free_bytes(), cache.capacity());
    }

    #[test]
    fn tenant_limit_prevents_one_tenant_from_owning_the_entire_arena() {
        let mut cache = GpuCache::new_with_limits(4 * 256, 256, 50, 100, HashMap::new()).unwrap();
        for key in ["a-1", "a-2"] {
            assert!(matches!(
                cache.admit_for_tenant("tenant-a", key.into(), 200, 1, 100, 2, false, true),
                Admission::Admitted { .. }
            ));
            cache.mark_ready(key).unwrap();
            cache.release(key).unwrap();
        }
        assert!(matches!(
            cache.admit_for_tenant("tenant-b", "b-1".into(), 200, 1, 100, 2, false, true,),
            Admission::Admitted { .. }
        ));
        assert_eq!(cache.tenant_count(), 2);
    }

    #[test]
    fn pinned_budget_fails_closed_before_consuming_pages() {
        let mut cache = GpuCache::new_with_limits(4 * 256, 256, 100, 25, HashMap::new()).unwrap();
        assert!(matches!(
            cache.admit_for_tenant(
                "__resident__",
                "resident".into(),
                200,
                1,
                100,
                2,
                true,
                true,
            ),
            Admission::Admitted { .. }
        ));
        assert_eq!(cache.pinned_bytes(), 256);
        assert_eq!(
            cache.admit_for_tenant(
                "__resident__",
                "too-much".into(),
                200,
                1,
                100,
                2,
                true,
                true,
            ),
            Admission::Deferred
        );
        assert_eq!(cache.allocated_bytes(), 256);
    }

    #[test]
    fn tenant_can_recycle_its_own_reserved_pages() {
        let mut reservations = HashMap::new();
        reservations.insert("tenant-a".to_owned(), 2 * 256);
        let mut cache = GpuCache::new_with_limits(2 * 256, 256, 50, 100, reservations).unwrap();
        for key in ["a-1", "a-2"] {
            assert!(matches!(
                cache.admit_for_tenant("tenant-a", key.into(), 200, 1, 100, 2, false, true),
                Admission::Admitted { .. }
            ));
            cache.mark_ready(key).unwrap();
            cache.release(key).unwrap();
        }
        assert!(matches!(
            cache.admit_for_tenant("tenant-a", "a-3".into(), 200, 1, 100, 2, false, true),
            Admission::Admitted { .. }
        ));
        assert_eq!(cache.entry_count(), 2);
    }
}
