// This software is licensed under the repository's dual license model.

use crate::backend::AcceleratorBackend;
use anyhow::{Result, bail};
use serde::Serialize;

/// Source-level contract used by independently built accelerator executors.
///
/// Vendor SDKs stay outside the core artifact: an Ascend or MetaX executable
/// depends on `backend-core`, implements this provider, and calls
/// `daemon::run_with_backend_provider`. A major mismatch is rejected before
/// command-line parsing, device allocation, cache mutation, or socket binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct BackendApiVersion {
    pub major: u16,
    pub minor: u16,
}

pub const BACKEND_API_VERSION: BackendApiVersion = BackendApiVersion { major: 1, minor: 1 };

pub trait BackendProvider {
    fn name(&self) -> &'static str;

    fn api_version(&self) -> BackendApiVersion {
        BACKEND_API_VERSION
    }

    fn create(
        &self,
        device: i32,
        total_bytes: usize,
        workspace_bytes: usize,
    ) -> Result<Box<dyn AcceleratorBackend>>;
}

pub fn validate_provider(provider: &dyn BackendProvider) -> Result<()> {
    let name = provider.name();
    if name.is_empty()
        || name.len() > 128
        || name
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte.is_ascii_whitespace())
    {
        bail!("accelerator provider name must be a non-empty token of at most 128 bytes");
    }
    let version = provider.api_version();
    if version.major != BACKEND_API_VERSION.major {
        bail!(
            "accelerator provider {name} uses backend API {}.{}, but this daemon requires {}.x",
            version.major,
            version.minor,
            BACKEND_API_VERSION.major
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct InvalidProvider {
        name: &'static str,
        version: BackendApiVersion,
    }

    impl BackendProvider for InvalidProvider {
        fn name(&self) -> &'static str {
            self.name
        }

        fn api_version(&self) -> BackendApiVersion {
            self.version
        }

        fn create(
            &self,
            _device: i32,
            _total_bytes: usize,
            _workspace_bytes: usize,
        ) -> Result<Box<dyn AcceleratorBackend>> {
            unreachable!("validation must run before device creation")
        }
    }

    #[test]
    fn rejects_incompatible_major_before_device_creation() {
        let provider = InvalidProvider {
            name: "example",
            version: BackendApiVersion { major: 2, minor: 0 },
        };
        assert!(validate_provider(&provider).is_err());
    }

    #[test]
    fn accepts_newer_minor_within_the_same_major() {
        let provider = InvalidProvider {
            name: "example",
            version: BackendApiVersion {
                major: BACKEND_API_VERSION.major,
                minor: BACKEND_API_VERSION.minor + 1,
            },
        };
        validate_provider(&provider).unwrap();
    }

    #[test]
    fn rejects_names_that_cannot_be_safe_structured_labels() {
        let provider = InvalidProvider {
            name: "bad provider\n",
            version: BACKEND_API_VERSION,
        };
        assert!(validate_provider(&provider).is_err());
    }
}
