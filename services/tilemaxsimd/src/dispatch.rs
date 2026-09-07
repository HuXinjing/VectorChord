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
}
