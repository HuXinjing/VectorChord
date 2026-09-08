// This software is licensed under the repository's dual license model.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelKind {
    Warp,
    Tile,
    Tensor,
}

#[derive(Clone, Copy, Debug)]
pub struct DispatchInput {
    pub query_rows: u64,
    pub storage_bytes_per_scalar: u8,
    pub shared_candidate_ratio_milli: u16,
    pub tensor_available: bool,
    pub tile_available: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct DispatchThresholds {
    pub cuda_ridge: u64,
    pub tensor_ridge: u64,
    pub minimum_shared_ratio_milli: u16,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct TensorCalibrationBucket {
    pub candidate_count: u32,
    pub reference_document_rows: u32,
    pub reference_row_groups: u32,
    pub threshold_query_rows: u32,
    pub tile_time_ns: u64,
    pub tensor_time_ns: u64,
    pub tensor_chunk_candidates: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct TensorDispatchInput {
    pub candidate_count: usize,
    pub total_query_rows: u32,
    pub total_document_rows: u64,
    pub document_row_groups: u32,
    pub dimension: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorDispatchDecision {
    pub use_tensor: bool,
    pub selected_candidate_bucket: u32,
    pub selected_document_rows: u32,
    pub selected_row_groups: u32,
    pub adjusted_threshold_rows: u64,
    pub effective_query_rows: u64,
    pub estimated_dot_products: u128,
}

pub fn choose_tensor(
    input: TensorDispatchInput,
    buckets: &[TensorCalibrationBucket],
) -> Option<TensorDispatchDecision> {
    if input.candidate_count == 0
        || input.total_query_rows == 0
        || input.total_document_rows == 0
        || input.dimension == 0
    {
        return None;
    }
    let selected_group_profile = buckets
        .iter()
        .map(|bucket| bucket.reference_row_groups)
        .filter(|groups| *groups >= input.document_row_groups.max(1))
        .min()
        .or_else(|| {
            buckets
                .iter()
                .map(|bucket| bucket.reference_row_groups)
                .max()
        })?;
    let mut matching = buckets
        .iter()
        .filter(|bucket| bucket.reference_row_groups == selected_group_profile)
        .collect::<Vec<_>>();
    matching.sort_unstable_by_key(|bucket| bucket.candidate_count);
    let reference_rows = matching.first()?.reference_document_rows.max(1) as u64;
    let normalized_candidates =
        input.total_document_rows.saturating_add(reference_rows - 1) / reference_rows;
    let selected = matching
        .iter()
        .rev()
        .find(|bucket| u64::from(bucket.candidate_count) <= normalized_candidates)
        .copied()
        .unwrap_or(matching[0]);

    let candidate_penalty = u64::from(selected.candidate_count)
        .saturating_add(normalized_candidates - 1)
        / normalized_candidates;
    let adjusted_threshold_rows =
        u64::from(selected.threshold_query_rows).saturating_mul(candidate_penalty.max(1));
    let effective_query_rows = u64::from(input.total_query_rows)
        .saturating_mul(u64::from(input.dimension))
        .saturating_add(319)
        / 320;
    Some(TensorDispatchDecision {
        use_tensor: selected.threshold_query_rows != u32::MAX
            && effective_query_rows >= adjusted_threshold_rows,
        selected_candidate_bucket: selected.candidate_count,
        selected_document_rows: selected.reference_document_rows,
        selected_row_groups: selected.reference_row_groups,
        adjusted_threshold_rows,
        effective_query_rows,
        estimated_dot_products: u128::from(input.total_query_rows)
            .saturating_mul(u128::from(input.total_document_rows))
            .saturating_mul(u128::from(input.dimension)),
    })
}

impl Default for DispatchThresholds {
    fn default() -> Self {
        // Conservative shadow-only fallback. Runtime calibration will replace
        // this in phase four; unknown hardware must never assume tensor support.
        Self {
            cuda_ridge: 128,
            tensor_ridge: 512,
            minimum_shared_ratio_milli: 500,
        }
    }
}

pub fn device_thresholds(name: &str, major: i32, minor: i32) -> Option<DispatchThresholds> {
    let tensor_ridge = if name.contains("RTX 4090") {
        328
    } else if name.contains("H200") {
        206
    } else if name.contains("A100") {
        156
    } else if name.contains("L40S") {
        424
    } else if name.contains("GB10") || name.contains("DGX Spark") {
        512
    } else {
        match (major, minor) {
            (9, _) => 384,
            (10, _) => 384,
            (12, _) => 512,
            (8, 0) => 256,
            (8, _) => 512,
            _ => return None,
        }
    };
    Some(match (major, minor) {
        (9, _) => DispatchThresholds {
            cuda_ridge: 14,
            tensor_ridge,
            ..Default::default()
        },
        (8, 0) => DispatchThresholds {
            cuda_ridge: 10,
            tensor_ridge,
            ..Default::default()
        },
        (8, _) => DispatchThresholds {
            cuda_ridge: 82,
            tensor_ridge,
            ..Default::default()
        },
        (10, _) => DispatchThresholds {
            cuda_ridge: 16,
            tensor_ridge,
            ..Default::default()
        },
        (12, _) => DispatchThresholds {
            cuda_ridge: 64,
            tensor_ridge,
            ..Default::default()
        },
        _ => return None,
    })
}

pub fn choose(input: DispatchInput, thresholds: DispatchThresholds) -> KernelKind {
    let quantization_multiplier = match input.storage_bytes_per_scalar {
        0 | 1 => 2,
        _ => 1,
    };
    let effective_reuse = input.query_rows.saturating_mul(quantization_multiplier);
    if input.shared_candidate_ratio_milli < thresholds.minimum_shared_ratio_milli
        || effective_reuse < thresholds.cuda_ridge
    {
        KernelKind::Warp
    } else if !input.tensor_available || effective_reuse < thresholds.tensor_ridge {
        if input.tile_available {
            KernelKind::Tile
        } else {
            KernelKind::Warp
        }
    } else {
        KernelKind::Tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(rows: u64) -> DispatchInput {
        DispatchInput {
            query_rows: rows,
            storage_bytes_per_scalar: 2,
            shared_candidate_ratio_milli: 1000,
            tensor_available: true,
            tile_available: true,
        }
    }

    #[test]
    fn dispatches_across_all_shadow_paths() {
        let thresholds = DispatchThresholds {
            cuda_ridge: 64,
            tensor_ridge: 256,
            ..Default::default()
        };
        assert_eq!(choose(input(32), thresholds), KernelKind::Warp);
        assert_eq!(choose(input(96), thresholds), KernelKind::Tile);
        assert_eq!(choose(input(512), thresholds), KernelKind::Tensor);
    }

    #[test]
    fn poor_candidate_overlap_prevents_false_reuse_claims() {
        let mut request = input(512);
        request.shared_candidate_ratio_milli = 100;
        assert_eq!(
            choose(request, DispatchThresholds::default()),
            KernelKind::Warp
        );
    }

    #[test]
    fn unavailable_kernel_falls_back_deterministically() {
        let mut request = input(1024);
        request.tensor_available = false;
        assert_eq!(
            choose(request, DispatchThresholds::default()),
            KernelKind::Tile
        );
        request.tile_available = false;
        assert_eq!(
            choose(request, DispatchThresholds::default()),
            KernelKind::Warp
        );
    }

    #[test]
    fn architecture_table_is_a_conservative_fallback() {
        assert_eq!(
            device_thresholds("NVIDIA H200", 9, 0).unwrap().tensor_ridge,
            206
        );
        assert_eq!(
            device_thresholds("NVIDIA A100", 8, 0).unwrap().tensor_ridge,
            156
        );
        assert_eq!(
            device_thresholds("NVIDIA L40S", 8, 9).unwrap().tensor_ridge,
            424
        );
        assert_eq!(
            device_thresholds("unknown", 8, 9).unwrap().tensor_ridge,
            512
        );
        assert_eq!(
            device_thresholds("NVIDIA GB10", 12, 1)
                .unwrap()
                .tensor_ridge,
            512
        );
        assert!(device_thresholds("legacy", 7, 5).is_none());
    }

    fn bucket(candidates: u32, groups: u32, threshold: u32) -> TensorCalibrationBucket {
        TensorCalibrationBucket {
            candidate_count: candidates,
            reference_document_rows: 32,
            reference_row_groups: groups,
            threshold_query_rows: threshold,
            tile_time_ns: 200,
            tensor_time_ns: 100,
            tensor_chunk_candidates: candidates,
        }
    }

    #[test]
    fn tensor_dispatch_uses_candidate_work_and_query_rows() {
        let buckets = [bucket(64, 1, 256), bucket(512, 1, 64)];
        let small = choose_tensor(
            TensorDispatchInput {
                candidate_count: 64,
                total_query_rows: 64,
                total_document_rows: 64 * 32,
                document_row_groups: 1,
                dimension: 320,
            },
            &buckets,
        )
        .unwrap();
        assert!(!small.use_tensor);
        let large = choose_tensor(
            TensorDispatchInput {
                candidate_count: 512,
                total_query_rows: 64,
                total_document_rows: 512 * 32,
                document_row_groups: 1,
                dimension: 320,
            },
            &buckets,
        )
        .unwrap();
        assert!(large.use_tensor);
        assert!(large.estimated_dot_products > small.estimated_dot_products);
    }

    #[test]
    fn tensor_dispatch_selects_a_calibrated_grouping_profile() {
        let buckets = [bucket(512, 1, 64), bucket(512, 8, 256)];
        let decision = choose_tensor(
            TensorDispatchInput {
                candidate_count: 512,
                total_query_rows: 128,
                total_document_rows: 512 * 32,
                document_row_groups: 6,
                dimension: 320,
            },
            &buckets,
        )
        .unwrap();
        assert_eq!(decision.selected_row_groups, 8);
        assert!(!decision.use_tensor);
    }

    #[test]
    fn longer_documents_increase_the_effective_candidate_bucket() {
        let buckets = [bucket(64, 1, 256), bucket(512, 1, 64)];
        let decision = choose_tensor(
            TensorDispatchInput {
                candidate_count: 256,
                total_query_rows: 64,
                total_document_rows: 256 * 64,
                document_row_groups: 1,
                dimension: 320,
            },
            &buckets,
        )
        .unwrap();
        assert_eq!(decision.selected_candidate_bucket, 512);
        assert!(decision.use_tensor);
    }
}
