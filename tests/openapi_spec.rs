//! Guards `openapi/openapi.yaml` against drift from the axum router.
//!
//! The spec is the source of truth for `/compute` and friends, so every route
//! registered in `src/main.rs` must be documented and vice versa.

const SPEC: &str = include_str!("../openapi/openapi.yaml");
const MAIN_RS: &str = include_str!("../src/main.rs");

/// Paths registered with `.route("...")` in the router.
fn router_paths(source: &str) -> Vec<String> {
    let mut paths = Vec::new();
    // The path literal may sit on the next line for multi-line routes, so take
    // the first string literal after each `.route(`.
    for tail in source.split(".route(").skip(1) {
        let Some(path) = tail.split('"').nth(1) else {
            continue;
        };
        if path.starts_with('/') && !paths.iter().any(|p| p == path) {
            paths.push(path.to_string());
        }
    }
    paths
}

/// Top-level keys under `paths:` (two-space indented, ending in a colon).
fn spec_paths(spec: &str) -> Vec<String> {
    let mut paths = Vec::new();
    let mut in_paths = false;
    for line in spec.lines() {
        if line.starts_with("paths:") {
            in_paths = true;
            continue;
        }
        if in_paths && !line.starts_with(' ') && !line.trim().is_empty() {
            break; // next top-level section (e.g. `components:`)
        }
        if !in_paths {
            continue;
        }
        let Some(rest) = line.strip_prefix("  ") else {
            continue;
        };
        if rest.starts_with(' ') || !rest.starts_with('/') {
            continue;
        }
        if let Some(path) = rest.strip_suffix(':') {
            paths.push(path.to_string());
        }
    }
    paths
}

#[test]
fn spec_documents_every_router_route() {
    let routes = router_paths(MAIN_RS);
    assert!(
        !routes.is_empty(),
        "no .route(\"/...\") entries found in src/main.rs"
    );

    let documented = spec_paths(SPEC);
    for route in &routes {
        assert!(
            documented.contains(route),
            "route {route} is missing from openapi/openapi.yaml (documented: {documented:?})"
        );
    }
}

#[test]
fn spec_has_no_undocumented_extra_paths() {
    let routes = router_paths(MAIN_RS);
    for path in spec_paths(SPEC) {
        assert!(
            routes.contains(&path),
            "openapi/openapi.yaml documents {path}, which has no .route() in src/main.rs"
        );
    }
}

#[test]
fn spec_is_openapi_3() {
    assert!(
        SPEC.starts_with("openapi: 3."),
        "openapi.yaml must declare an OpenAPI 3.x version"
    );
    assert!(
        SPEC.contains("title: WQC Core Compute HTTP API"),
        "openapi.yaml is missing the expected title"
    );
}

/// Single-parameter gates take a bare integer; a one-element array is rejected
/// by serde. Keep that stated so the examples are not copied incorrectly.
#[test]
fn spec_records_bare_integer_gate_params() {
    assert!(
        SPEC.contains("bare integer"),
        "openapi.yaml must document the bare-integer params form for single-parameter gates"
    );
}
