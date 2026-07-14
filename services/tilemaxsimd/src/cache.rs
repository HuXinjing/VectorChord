use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};

#[derive(Debug)]
pub struct BuddyAllocator {
    block_bytes: usize,
    block_count: usize,
    free: BTreeMap<u32, BTreeSet<(usize, usize, u32)>>,
    allocated: HashMap<usize, (u32, usize, u32)>,
}

impl BuddyAllocator {
    pub fn new(capacity: usize, block_bytes: usize) -> Result<Self, &'static str> {
        if capacity == 0 || block_bytes == 0 || !block_bytes.is_multiple_of(256) {
            return Err("invalid fixed-block arena");
        }
        let block_count = capacity / block_bytes;
        if block_count == 0 {
            return Err("fixed-block arena is too small");
        }
        let mut free = BTreeMap::<u32, BTreeSet<_>>::new();
        let mut start = 0;
        let mut remaining = block_count;
        while remaining != 0 {
            let order = usize::BITS - 1 - remaining.leading_zeros();
            let size = 1_usize << order;
            free.entry(order).or_default().insert((start, start, order));
            start += size;
            remaining -= size;
        }
        Ok(Self {
            block_bytes,
            block_count,
            free,
            allocated: HashMap::new(),
        })
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
        let raw = payload_bytes.div_ceil(self.block_bytes);
        raw.checked_next_power_of_two()?
            .checked_mul(self.block_bytes)
    }

    pub fn largest_free(&self) -> usize {
        self.free
            .iter()
            .rev()
            .find(|(_, entries)| !entries.is_empty())
            .map_or(0, |(order, _)| (1_usize << order) * self.block_bytes)
    }

    pub fn allocate(&mut self, payload_bytes: usize) -> Option<(usize, usize)> {
        let allocation_bytes = self.allocation_bytes(payload_bytes)?;
        let blocks = allocation_bytes / self.block_bytes;
        let order = blocks.trailing_zeros();
        let available_order = self
            .free
            .range(order..)
            .find(|(_, entries)| !entries.is_empty())
            .map(|(candidate, _)| *candidate)?;
        let item = self.free.get_mut(&available_order)?.pop_first()?;
        let (start, root_start, root_order) = item;
        let mut current_order = available_order;
        while current_order > order {
            current_order -= 1;
            let buddy = start + (1_usize << current_order);
            self.free
                .entry(current_order)
                .or_default()
                .insert((buddy, root_start, root_order));
        }
        self.allocated
            .insert(start, (order, root_start, root_order));
        Some((start * self.block_bytes, allocation_bytes))
    }

    pub fn release(&mut self, offset: usize) -> Result<(), &'static str> {
        if !offset.is_multiple_of(self.block_bytes) {
            return Err("unaligned fixed-block release");
        }
        let mut start = offset / self.block_bytes;
        let (mut order, root_start, root_order) = self
            .allocated
            .remove(&start)
            .ok_or("fixed-block slab was released twice")?;
        while order < root_order {
            let buddy = root_start + ((start - root_start) ^ (1_usize << order));
            let buddy_item = (buddy, root_start, root_order);
            let entries = self.free.entry(order).or_default();
            if !entries.remove(&buddy_item) {
                break;
            }
            start = start.min(buddy);
            order += 1;
        }
        self.free
            .entry(order)
            .or_default()
            .insert((start, root_start, root_order));
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
    priority: f64,
}

#[derive(Debug, PartialEq)]
pub enum Admission {
    Admitted { offset: u64, allocated_bytes: usize },
    Rejected,
    Deferred,
}

pub struct GpuCache {
    allocator: BuddyAllocator,
    entries: HashMap<String, CacheEntry>,
    sketch: TinyLfu,
    inflation: f64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub admission_rejections: u64,
}

impl GpuCache {
    pub fn new(capacity: usize, block_bytes: usize) -> Result<Self, &'static str> {
        Ok(Self {
            allocator: BuddyAllocator::new(capacity, block_bytes)?,
            entries: HashMap::new(),
            sketch: TinyLfu::new(4096),
            inflation: 0.0,
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

    pub fn get(&mut self, key: &str) -> Option<CacheEntry> {
        let frequency = self.sketch.increment(&key);
        let Some(entry) = self.entries.get_mut(key) else {
            self.misses += 1;
            return None;
        };
        entry.references += 1;
        entry.priority = self.inflation + f64::from(frequency) / entry.allocated_bytes as f64;
        self.hits += 1;
        Some(entry.clone())
    }

    pub fn acquire_existing(&mut self, key: &str) -> Option<CacheEntry> {
        let entry = self.entries.get_mut(key)?;
        entry.references += 1;
        Some(entry.clone())
    }

    pub fn record_access_miss(&mut self, key: &str) {
        self.sketch.increment(&key);
        self.misses += 1;
    }

    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    fn victim(&self) -> Option<(&String, &CacheEntry)> {
        self.entries
            .iter()
            .filter(|(_, entry)| entry.references == 0 && !entry.pinned)
            .min_by(|left, right| left.1.priority.total_cmp(&right.1.priority))
    }

    fn evict_one(&mut self) -> bool {
        let Some(key) = self.victim().map(|(key, _)| key.clone()) else {
            return false;
        };
        let entry = self.entries.remove(&key).expect("victim disappeared");
        self.allocator
            .release(entry.offset as usize)
            .expect("cache slab release failed");
        self.inflation = self.inflation.max(entry.priority);
        self.evictions += 1;
        true
    }

    #[allow(clippy::too_many_arguments)]
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
        let Some(allocation_bytes) = self.allocator.allocation_bytes(payload_bytes) else {
            return Admission::Deferred;
        };
        if allocation_bytes > self.allocator.capacity() {
            return Admission::Deferred;
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
                priority,
            },
        );
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

    #[cfg(test)]
    pub fn entry(&self, key: &str) -> Option<&CacheEntry> {
        self.entries.get(key)
    }

    pub fn remove(&mut self, key: &str) -> Result<(), &'static str> {
        let entry = self
            .entries
            .remove(key)
            .ok_or("GPU cache entry disappeared")?;
        self.allocator.release(entry.offset as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buddy_coalesces_fixed_slabs() {
        let mut allocator = BuddyAllocator::new(8 * 256, 256).unwrap();
        let first = allocator.allocate(300).unwrap();
        let second = allocator.allocate(300).unwrap();
        let third = allocator.allocate(700).unwrap();
        assert_eq!(first, (0, 512));
        assert_eq!(second, (512, 512));
        assert_eq!(third, (1024, 1024));
        allocator.release(second.0).unwrap();
        allocator.release(first.0).unwrap();
        assert_eq!(allocator.largest_free(), 1024);
        allocator.release(third.0).unwrap();
        assert_eq!(allocator.largest_free(), 2048);
    }

    #[test]
    fn tinylfu_protects_hot_entry() {
        let mut cache = GpuCache::new(256, 256).unwrap();
        cache.record_access_miss("hot");
        assert!(matches!(
            cache.admit("hot".into(), 64, 1, 32, 2, false, false),
            Admission::Admitted { .. }
        ));
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
}
