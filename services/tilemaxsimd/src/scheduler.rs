use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulerPolicy {
    Fair,
    Priority,
    FairPriority,
}

impl SchedulerPolicy {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "fair" => Ok(Self::Fair),
            "priority" => Ok(Self::Priority),
            "fair-priority" => Ok(Self::FairPriority),
            _ => Err("scheduler policy must be fair, priority, or fair-priority".to_owned()),
        }
    }
}

pub struct Scheduled<T> {
    pub tenant: String,
    pub priority: i32,
    pub cost: u64,
    pub enqueued_at: Instant,
    pub deadline: Instant,
    pub payload: T,
    sequence: u64,
}

impl<T> Scheduled<T> {
    pub fn new(
        tenant: String,
        priority: i32,
        cost: u64,
        enqueued_at: Instant,
        deadline: Instant,
        payload: T,
    ) -> Self {
        Self {
            tenant,
            priority,
            cost: cost.max(1),
            enqueued_at,
            deadline,
            payload,
            sequence: 0,
        }
    }
}

/// Hierarchical request scheduler inspired by vLLM's priority queue, with a
/// tenant-fair first level that vLLM does not need in its usual single-owner
/// deployment. `fair-priority` admits the highest aged priority band and then
/// selects the tenant with the lowest normalized service time. This preserves
/// explicit urgency while preventing equal-priority noisy neighbours from
/// monopolizing the GPU.
pub struct RequestQueue<T> {
    policy: SchedulerPolicy,
    aging: Duration,
    priority_band: i32,
    pending: Vec<Scheduled<T>>,
    virtual_runtime: HashMap<String, f64>,
    weights: HashMap<String, f64>,
    next_sequence: u64,
}

impl<T> RequestQueue<T> {
    pub fn new(
        policy: SchedulerPolicy,
        aging: Duration,
        priority_band: i32,
        weights: HashMap<String, f64>,
    ) -> Self {
        Self {
            policy,
            aging,
            priority_band: priority_band.max(0),
            pending: Vec::new(),
            virtual_runtime: HashMap::new(),
            weights,
            next_sequence: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn push(&mut self, mut item: Scheduled<T>) {
        item.sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        if self.pending.is_empty() {
            // A new busy period should not inherit unbounded floating-point
            // service history from a previous burst.
            self.virtual_runtime.clear();
        }
        let tenant_is_active = self
            .pending
            .iter()
            .any(|pending| pending.tenant == item.tenant);
        let baseline = self
            .pending
            .iter()
            .filter_map(|pending| self.virtual_runtime.get(&pending.tenant).copied())
            .reduce(f64::min)
            .unwrap_or(0.0);
        let runtime = self
            .virtual_runtime
            .entry(item.tenant.clone())
            .or_insert(baseline);
        if !tenant_is_active {
            *runtime = runtime.max(baseline);
        }
        self.pending.push(item);
    }

    pub fn drain_expired(&mut self, now: Instant) -> Vec<Scheduled<T>> {
        let mut expired = Vec::new();
        let mut index = 0;
        while index < self.pending.len() {
            if self.pending[index].deadline <= now {
                expired.push(self.pending.swap_remove(index));
            } else {
                index += 1;
            }
        }
        expired.sort_by_key(|item| item.sequence);
        expired
    }

    pub fn pop(&mut self, now: Instant) -> Option<Scheduled<T>> {
        if self.pending.is_empty() {
            return None;
        }
        let effective = self
            .pending
            .iter()
            .map(|item| self.effective_priority(item, now))
            .collect::<Vec<_>>();
        let highest = effective.iter().copied().max().unwrap_or(0);
        let mut best = 0;
        for candidate in 1..self.pending.len() {
            if self.better(candidate, best, &effective, highest) {
                best = candidate;
            }
        }
        let item = self.pending.swap_remove(best);
        let weight = self.weights.get(&item.tenant).copied().unwrap_or(1.0);
        let runtime = self.virtual_runtime.entry(item.tenant.clone()).or_default();
        *runtime += item.cost as f64 / weight;
        Some(item)
    }

    fn effective_priority(&self, item: &Scheduled<T>, now: Instant) -> i32 {
        if self.aging.is_zero() {
            return item.priority;
        }
        let waited = now.saturating_duration_since(item.enqueued_at).as_millis();
        let steps = waited / self.aging.as_millis().max(1);
        item.priority
            .saturating_add(i32::try_from(steps.min(200)).unwrap_or(200))
            .min(100)
    }

    fn better(&self, candidate: usize, current: usize, effective: &[i32], highest: i32) -> bool {
        let left = &self.pending[candidate];
        let right = &self.pending[current];
        match self.policy {
            SchedulerPolicy::Fair => self.fair_cmp(left, right),
            SchedulerPolicy::Priority => {
                effective[candidate] > effective[current]
                    || (effective[candidate] == effective[current]
                        && left.sequence < right.sequence)
            }
            SchedulerPolicy::FairPriority => {
                let floor = highest.saturating_sub(self.priority_band);
                let left_eligible = effective[candidate] >= floor;
                let right_eligible = effective[current] >= floor;
                if left_eligible != right_eligible {
                    return left_eligible;
                }
                if !left_eligible {
                    return effective[candidate] > effective[current]
                        || (effective[candidate] == effective[current]
                            && left.sequence < right.sequence);
                }
                if left.tenant == right.tenant {
                    return effective[candidate] > effective[current]
                        || (effective[candidate] == effective[current]
                            && left.sequence < right.sequence);
                }
                let left_runtime = self
                    .virtual_runtime
                    .get(&left.tenant)
                    .copied()
                    .unwrap_or(0.0);
                let right_runtime = self
                    .virtual_runtime
                    .get(&right.tenant)
                    .copied()
                    .unwrap_or(0.0);
                left_runtime < right_runtime
                    || (left_runtime == right_runtime
                        && (effective[candidate] > effective[current]
                            || (effective[candidate] == effective[current]
                                && left.sequence < right.sequence)))
            }
        }
    }

    fn fair_cmp(&self, left: &Scheduled<T>, right: &Scheduled<T>) -> bool {
        let left_runtime = self
            .virtual_runtime
            .get(&left.tenant)
            .copied()
            .unwrap_or(0.0);
        let right_runtime = self
            .virtual_runtime
            .get(&right.tenant)
            .copied()
            .unwrap_or(0.0);
        left_runtime < right_runtime
            || (left_runtime == right_runtime && left.sequence < right.sequence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(tenant: &str, priority: i32, cost: u64, now: Instant, id: u32) -> Scheduled<u32> {
        Scheduled::new(
            tenant.to_owned(),
            priority,
            cost,
            now,
            now + Duration::from_secs(60),
            id,
        )
    }

    #[test]
    fn strict_priority_uses_higher_priority_then_fcfs() {
        let now = Instant::now();
        let mut queue = RequestQueue::new(
            SchedulerPolicy::Priority,
            Duration::from_secs(60),
            0,
            HashMap::new(),
        );
        queue.push(item("a", 0, 1, now, 1));
        queue.push(item("b", 10, 1, now, 2));
        queue.push(item("c", 10, 1, now, 3));
        assert_eq!(queue.pop(now).unwrap().payload, 2);
        assert_eq!(queue.pop(now).unwrap().payload, 3);
        assert_eq!(queue.pop(now).unwrap().payload, 1);
    }

    #[test]
    fn fair_priority_alternates_equal_priority_tenants() {
        let now = Instant::now();
        let mut queue = RequestQueue::new(
            SchedulerPolicy::FairPriority,
            Duration::from_secs(60),
            0,
            HashMap::new(),
        );
        queue.push(item("noisy", 0, 10, now, 1));
        queue.push(item("noisy", 0, 10, now, 2));
        queue.push(item("quiet", 0, 10, now, 3));
        assert_eq!(queue.pop(now).unwrap().payload, 1);
        assert_eq!(queue.pop(now).unwrap().payload, 3);
        assert_eq!(queue.pop(now).unwrap().payload, 2);
    }

    #[test]
    fn full_priority_band_keeps_priority_without_sacrificing_tenant_fairness() {
        let now = Instant::now();
        let mut queue = RequestQueue::new(
            SchedulerPolicy::FairPriority,
            Duration::from_secs(60),
            200,
            HashMap::new(),
        );
        queue.push(item("noisy", 100, 10, now, 1));
        queue.push(item("noisy", 100, 10, now, 2));
        queue.push(item("quiet", -100, 10, now, 3));
        assert_eq!(queue.pop(now).unwrap().payload, 1);
        assert_eq!(queue.pop(now).unwrap().payload, 3);
        assert_eq!(queue.pop(now).unwrap().payload, 2);
    }

    #[test]
    fn aging_eventually_promotes_waiting_work() {
        let now = Instant::now();
        let mut queue = RequestQueue::new(
            SchedulerPolicy::Priority,
            Duration::from_millis(10),
            0,
            HashMap::new(),
        );
        queue.push(item("old", 0, 1, now, 1));
        queue.push(item("new", 5, 1, now + Duration::from_millis(60), 2));
        assert_eq!(
            queue.pop(now + Duration::from_millis(60)).unwrap().payload,
            1
        );
    }

    #[test]
    fn expired_requests_leave_without_consuming_service_credit() {
        let now = Instant::now();
        let mut queue = RequestQueue::new(
            SchedulerPolicy::FairPriority,
            Duration::from_secs(1),
            0,
            HashMap::new(),
        );
        let mut expired = item("a", 0, 1, now, 1);
        expired.deadline = now;
        queue.push(expired);
        queue.push(item("b", 0, 1, now, 2));
        assert_eq!(queue.drain_expired(now).len(), 1);
        assert_eq!(queue.pop(now).unwrap().payload, 2);
    }
}
