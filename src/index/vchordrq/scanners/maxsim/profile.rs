// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2026 Hu Xinjing

use std::cell::RefCell;
use std::time::{Duration, Instant};

#[derive(Default)]
pub(super) struct MaxsimProfile {
    pub query_tokens: u64,
    pub token_search_calls: u64,
    pub token_search_us: u64,
    pub token_search_results: u64,
    pub token_refine_calls: u64,
    pub token_refine_us: u64,
    pub accurate_token_hits: u64,
    pub rough_token_hits: u64,
    pub hit_collect_us: u64,
    pub hit_updates: u64,
    pub page_token_updates: u64,
    pub hit_sort_us: u64,
    pub page_aggregate_us: u64,
    pub aggregated_pages: u64,
    pub preflight_us: u64,
    pub candidate_generation_us: u64,
    pub generated_candidates: u64,
    pub visibility_us: u64,
    pub visible_candidates: u64,
    pub descriptor_us: u64,
    pub descriptors: u64,
    pub sidecar_us: u64,
    pub result_finalize_us: u64,
    pub returned_rows: u64,
    pub total_us: u64,
}

impl MaxsimProfile {
    fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"query_tokens\":{},",
                "\"token_search_calls\":{},",
                "\"token_search_us\":{},",
                "\"token_search_results\":{},",
                "\"token_refine_calls\":{},",
                "\"token_refine_us\":{},",
                "\"accurate_token_hits\":{},",
                "\"rough_token_hits\":{},",
                "\"hit_collect_us\":{},",
                "\"hit_updates\":{},",
                "\"page_token_updates\":{},",
                "\"hit_sort_us\":{},",
                "\"page_aggregate_us\":{},",
                "\"aggregated_pages\":{},",
                "\"preflight_us\":{},",
                "\"candidate_generation_us\":{},",
                "\"generated_candidates\":{},",
                "\"visibility_us\":{},",
                "\"visible_candidates\":{},",
                "\"descriptor_us\":{},",
                "\"descriptors\":{},",
                "\"sidecar_us\":{},",
                "\"result_finalize_us\":{},",
                "\"returned_rows\":{},",
                "\"total_us\":{}",
                "}}"
            ),
            self.query_tokens,
            self.token_search_calls,
            self.token_search_us,
            self.token_search_results,
            self.token_refine_calls,
            self.token_refine_us,
            self.accurate_token_hits,
            self.rough_token_hits,
            self.hit_collect_us,
            self.hit_updates,
            self.page_token_updates,
            self.hit_sort_us,
            self.page_aggregate_us,
            self.aggregated_pages,
            self.preflight_us,
            self.candidate_generation_us,
            self.generated_candidates,
            self.visibility_us,
            self.visible_candidates,
            self.descriptor_us,
            self.descriptors,
            self.sidecar_us,
            self.result_finalize_us,
            self.returned_rows,
            self.total_us,
        )
    }
}

thread_local! {
    static ACTIVE_PROFILE: RefCell<Option<MaxsimProfile>> = const { RefCell::new(None) };
}

pub(super) struct ProfileGuard {
    enabled: bool,
    finished: bool,
}

impl ProfileGuard {
    pub fn start(enabled: bool) -> Self {
        if enabled {
            ACTIVE_PROFILE.with(|slot| {
                *slot.borrow_mut() = Some(MaxsimProfile::default());
            });
        }
        Self {
            enabled,
            finished: false,
        }
    }

    pub fn finish(mut self, total: Duration) {
        if self.enabled {
            let profile = ACTIVE_PROFILE.with(|slot| {
                let mut profile = slot.borrow_mut().take();
                if let Some(profile) = profile.as_mut() {
                    profile.total_us = duration_us(total);
                }
                profile
            });
            if let Some(profile) = profile {
                pgrx::notice!("vchordrq_maxsim_profile {}", profile.to_json());
            }
        }
        self.finished = true;
    }
}

impl Drop for ProfileGuard {
    fn drop(&mut self) {
        if self.enabled && !self.finished {
            ACTIVE_PROFILE.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }
}

pub(super) struct ProfileTimer(Option<Instant>);

impl ProfileTimer {
    pub fn start() -> Self {
        let enabled = ACTIVE_PROFILE.with(|slot| slot.borrow().is_some());
        Self(enabled.then(Instant::now))
    }

    pub fn elapsed(self) -> Duration {
        self.0.map_or(Duration::ZERO, |started| started.elapsed())
    }
}

pub(super) fn update(f: impl FnOnce(&mut MaxsimProfile)) {
    ACTIVE_PROFILE.with(|slot| {
        if let Some(profile) = slot.borrow_mut().as_mut() {
            f(profile);
        }
    });
}

pub(super) fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}
