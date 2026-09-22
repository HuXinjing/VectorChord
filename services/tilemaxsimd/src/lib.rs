// This software is licensed under the repository's dual license model.

pub mod backend;
pub mod backend_sdk;
pub mod cache;
#[cfg(feature = "backend-cpu")]
pub mod cpu;
pub mod daemon;
pub mod dispatch;
pub mod engine;
#[cfg(feature = "backend-cuda")]
pub mod gpu;
#[cfg(feature = "backend-metal")]
pub mod metal;
pub mod protocol;
pub mod quant;
pub mod scheduler;
pub use tilemaxsim_protocol as protocol_codec;
pub mod shard;
#[cfg(feature = "backend-vulkan")]
pub mod vulkan;

#[cfg(test)]
mod protocol_codec_contract_tests {
    use super::{protocol, protocol_codec};

    #[test]
    fn protocol_library_frames_are_accepted_by_the_daemon_parser() {
        let query = [0_u8; 8];
        let request = protocol_codec::CatalogRequest {
            request_id: 41,
            model_contract: "colqwen@1",
            tenant: "tenant-7",
            priority: 5,
            timeout_ms: 2_000,
            query_rows: 2,
            dimension: 2,
            query_dtype: protocol_codec::TensorDtype::Float16,
            candidate_dtype: protocol_codec::TensorDtype::Fp8E4m3,
            scoring_profile: protocol_codec::ScoringProfile::Fp8E4m3Raw,
            quantization_contract: None,
            top_k: 2,
            query: &query,
            catalog_revision: "catalog-17",
        };
        let ids = [11, 20, 42];
        let frame = protocol_codec::encode_catalog_selection_reference(
            &request,
            &ids,
            Some(&[20, 42]),
            true,
        )
        .unwrap();
        let parsed = protocol::parse(&frame).unwrap();
        assert_eq!(
            parsed.protocol_version,
            protocol_codec::VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE
        );
        assert_eq!(parsed.scoped_candidate_ordinals, [1, 2]);
    }
}

#[cfg(any(
    all(
        feature = "backend-vulkan",
        any(
            feature = "backend-cpu",
            feature = "backend-cuda",
            feature = "backend-metal"
        )
    ),
    all(feature = "backend-cuda", feature = "backend-cpu"),
    all(feature = "backend-cuda", feature = "backend-metal"),
    all(feature = "backend-cpu", feature = "backend-metal")
))]
compile_error!("select exactly one accelerator backend");
