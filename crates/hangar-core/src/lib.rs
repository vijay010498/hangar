//! Shared types for Hangar: config loading and small helpers used across the
//! daemon (`hangard`), the agent CLI (`hangar`), and the dashboard (`hangar-tui`).

pub mod config;
pub mod memory;
pub mod router;
pub mod runtime;

/// True if `raw` is an HTTP(S) URL whose host is a loopback address or
/// `localhost`. Used to enforce flight mode: in offline mode the daemon refuses
/// to bind to, or talk to, anything that isn't loopback.
pub fn is_loopback_url(raw: &str) -> bool {
    match url::Url::parse(raw) {
        Ok(u) => match u.host() {
            Some(url::Host::Domain(d)) => d.eq_ignore_ascii_case("localhost"),
            Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
            Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
            None => false,
        },
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::is_loopback_url;

    #[test]
    fn loopback_urls_are_recognized() {
        assert!(is_loopback_url("http://127.0.0.1:11434"));
        assert!(is_loopback_url("http://localhost:11434"));
        assert!(is_loopback_url("http://[::1]:11434"));
    }

    #[test]
    fn remote_urls_are_rejected() {
        assert!(!is_loopback_url("http://10.0.0.5:11434"));
        assert!(!is_loopback_url("https://api.openai.com"));
        assert!(!is_loopback_url("not a url"));
    }
}
