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
// Copyright (c) 2025-2026 TensorChord Inc.

#[derive(Clone, Copy, Debug)]
pub struct MaxsimCostInput {
    pub heap_rows: f64,
    pub index_tokens: f64,
    pub token_nodes_per_query: f64,
    pub base_index_pages: f64,
    pub query_tokens: u32,
    pub limit_tuples: Option<f64>,
    pub filter_selectivity: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct MaxsimCostEstimate {
    pub startup_cost: f64,
    pub total_cost: f64,
    pub selectivity: f64,
    pub index_pages: f64,
}

/// Estimate the eager token search and page aggregation performed by MaxSim.
///
/// The constants are deliberately conservative rather than hardware-specific;
/// an optional exact reranker can add its own cost in a later layer.
pub fn estimate_maxsim_cost(input: MaxsimCostInput) -> MaxsimCostEstimate {
    let heap_rows = input.heap_rows.max(1.0);
    let index_tokens = input.index_tokens.max(heap_rows);
    let query_tokens = f64::from(input.query_tokens.max(1));
    let average_document_tokens = (index_tokens / heap_rows).clamp(1.0, 65_536.0);
    let token_visits = input.token_nodes_per_query.max(1.0) * query_tokens;

    // Until page-level candidate statistics exist, approximate the chance that
    // at least one token from a document is visited by the token index.
    let token_visit_fraction = (token_visits / index_tokens).clamp(0.0, 1.0);
    let candidate_probability = 1.0 - (1.0 - token_visit_fraction).powf(average_document_tokens);
    let generated_pages = (heap_rows * candidate_probability).clamp(1.0, heap_rows);

    let returned_pages = input
        .limit_tuples
        .map(|limit| limit.max(1.0) / input.filter_selectivity.clamp(1e-9, 1.0))
        .unwrap_or(generated_pages)
        .min(generated_pages);

    // Search and aggregation are eager in the current scanner, so LIMIT does
    // not remove this work from startup cost.
    let startup_cost = 0.001 * token_visits + 0.01 * token_visits + 0.05 * generated_pages;
    let total_cost = startup_cost + returned_pages;
    let selectivity = (returned_pages / heap_rows).clamp(1e-9, 1.0);
    let index_pages =
        input.base_index_pages.max(1.0) * (1.0 + 0.25 * (query_tokens - 1.0).max(0.0));

    MaxsimCostEstimate {
        startup_cost,
        total_cost,
        selectivity,
        index_pages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> MaxsimCostInput {
        MaxsimCostInput {
            heap_rows: 10_000.0,
            index_tokens: 1_000_000.0,
            token_nodes_per_query: 2_000.0,
            base_index_pages: 1_000.0,
            query_tokens: 16,
            limit_tuples: Some(20.0),
            filter_selectivity: 1.0,
        }
    }

    #[test]
    fn maxsim_cost_is_finite_and_nonzero() {
        let estimate = estimate_maxsim_cost(input());
        assert!(estimate.startup_cost > 0.0);
        assert!(estimate.total_cost >= estimate.startup_cost);
        assert!((1e-9..=1.0).contains(&estimate.selectivity));
        assert!(estimate.index_pages >= 1.0);
    }

    #[test]
    fn query_token_count_increases_eager_work() {
        let one = estimate_maxsim_cost(MaxsimCostInput {
            query_tokens: 1,
            ..input()
        });
        let many = estimate_maxsim_cost(MaxsimCostInput {
            query_tokens: 64,
            ..input()
        });
        assert!(many.startup_cost > one.startup_cost);
        assert!(many.index_pages > one.index_pages);
    }

    #[test]
    fn missing_statistics_remain_finite() {
        let estimate = estimate_maxsim_cost(MaxsimCostInput {
            heap_rows: -1.0,
            index_tokens: 0.0,
            token_nodes_per_query: 0.0,
            base_index_pages: 0.0,
            filter_selectivity: 0.0,
            limit_tuples: None,
            ..input()
        });
        assert!(estimate.startup_cost.is_finite());
        assert!(estimate.total_cost.is_finite());
        assert!(estimate.selectivity.is_finite());
        assert!(estimate.index_pages.is_finite());
    }
}
