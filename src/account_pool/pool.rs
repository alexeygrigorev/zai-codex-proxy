use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use crate::config::{AccountConfig, RouteTargetConfig, RoutingHealthConfig};
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AccountAuth {
    ApiKey { api_key: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub id: String,
    pub auth: AccountAuth,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, skip_serializing)]
    pub weight: u32,
    #[serde(default)]
    pub models: Option<Vec<String>>,
}

impl Account {
    pub fn supports_model(&self, model: &str) -> bool {
        match &self.models {
            Some(models) => models.iter().any(|allowed| allowed == model),
            None => true,
        }
    }
}

impl From<AccountConfig> for Account {
    fn from(value: AccountConfig) -> Self {
        Self {
            id: value.id,
            auth: value.auth,
            enabled: value.enabled,
            weight: value.weight,
            models: value.models,
        }
    }
}

struct AccountState {
    alive: bool,
    consecutive_failures: u32,
    cache_key_hits: AtomicU64,
    last_failure_at: Option<SystemTime>,
    last_error: Option<String>,
    unhealthy_until: Option<SystemTime>,
    recovery_probe_due: bool,
    probe_in_progress: AtomicBool,
}

impl AccountState {
    fn new() -> Self {
        Self {
            alive: true,
            consecutive_failures: 0,
            cache_key_hits: AtomicU64::new(0),
            last_failure_at: None,
            last_error: None,
            unhealthy_until: None,
            recovery_probe_due: false,
            probe_in_progress: AtomicBool::new(false),
        }
    }
}

pub struct AccountPool {
    accounts: RwLock<Vec<(Account, RwLock<AccountState>)>>,
    health: RwLock<RoutingHealthConfig>,
}

impl AccountPool {
    pub fn new() -> Self {
        Self {
            accounts: RwLock::new(Vec::new()),
            health: RwLock::new(RoutingHealthConfig::default()),
        }
    }

    pub fn configure_health(&self, config: RoutingHealthConfig) {
        *self.health.write() = config;
    }

    pub fn load_accounts(&self, accounts: Vec<Account>) {
        let mut next = Vec::new();
        {
            let guard = self.accounts.read();
            for account in accounts {
                if !account.enabled {
                    continue;
                }
                let existing_state = guard
                    .iter()
                    .find(|(existing, _)| existing.id == account.id)
                    .map(|(_, state)| {
                        let snapshot = state.read();
                        RwLock::new(AccountState {
                            alive: snapshot.alive,
                            consecutive_failures: snapshot.consecutive_failures,
                            cache_key_hits: AtomicU64::new(
                                snapshot.cache_key_hits.load(Ordering::Relaxed),
                            ),
                            last_failure_at: snapshot.last_failure_at,
                            last_error: snapshot.last_error.clone(),
                            unhealthy_until: snapshot.unhealthy_until,
                            recovery_probe_due: snapshot.recovery_probe_due,
                            probe_in_progress: AtomicBool::new(
                                snapshot.probe_in_progress.load(Ordering::Relaxed),
                            ),
                        })
                    })
                    .unwrap_or_else(|| RwLock::new(AccountState::new()));
                next.push((account, existing_state));
            }
        }
        *self.accounts.write() = next;
        info!(
            "Account pool loaded: {} accounts",
            self.accounts.read().len()
        );
    }

    pub fn healthy_compatible_accounts_for_target(&self, target: &RouteTargetConfig) -> Vec<usize> {
        let guard = self.accounts.read();
        guard
            .iter()
            .enumerate()
            .filter_map(|(i, (account, _))| {
                (account.supports_model(&target.model)
                    && self.get_account(i).map(|(_, s)| s.alive).unwrap_or(false))
                .then_some(i)
            })
            .collect()
    }

    pub fn recovery_candidates(&self) -> Vec<usize> {
        let guard = self.accounts.read();
        let now = SystemTime::now();
        guard
            .iter()
            .enumerate()
            .filter_map(|(i, (_, state))| {
                let state = state.read();
                let due = state.recovery_probe_due
                    && state
                        .unhealthy_until
                        .map(|until| until <= now)
                        .unwrap_or(false)
                    && !state.probe_in_progress.load(Ordering::Relaxed);
                due.then_some(i)
            })
            .collect()
    }

    pub fn begin_recovery_probe(&self, index: usize) -> bool {
        let guard = self.accounts.read();
        let Some((_, state)) = guard.get(index) else {
            return false;
        };
        let state = state.read();
        !state.probe_in_progress.swap(true, Ordering::AcqRel)
    }

    pub fn finish_recovery_probe(&self, index: usize, success: bool, error: Option<&str>) {
        let guard = self.accounts.read();
        if let Some((account, state)) = guard.get(index) {
            let mut state = state.write();
            state.probe_in_progress.store(false, Ordering::Release);
            if success {
                state.alive = true;
                state.consecutive_failures = 0;
                state.last_error = None;
                state.unhealthy_until = None;
                state.recovery_probe_due = false;
                info!("Recovery probe succeeded for account {}", account.id);
            } else {
                let health = self.health.read().clone();
                let now = SystemTime::now();
                state.last_failure_at = Some(now);
                state.last_error = error.map(str::to_string);
                state.consecutive_failures += 1;
                state.alive = false;
                let backoff_factor = state
                    .consecutive_failures
                    .saturating_sub(1)
                    .min(health.failure_threshold.saturating_sub(1));
                let cooldown_multiplier = 1u64 << backoff_factor;
                let cooldown_seconds = health.cooldown_seconds.saturating_mul(cooldown_multiplier);
                state.unhealthy_until = Some(now + Duration::from_secs(cooldown_seconds));
                state.recovery_probe_due = true;
                let reason = sanitize_error_for_log(error.unwrap_or("unknown"));
                warn!(
                    "Recovery probe failed for account {} reason={} next backoff={}s failures={}",
                    account.id, reason, cooldown_seconds, state.consecutive_failures
                );
            }
        }
    }

    pub fn get_account(&self, index: usize) -> Option<(Account, AccountSnapshot)> {
        let guard = self.accounts.read();
        guard.get(index).map(|(account, state)| {
            let state = state.read();
            (
                account.clone(),
                AccountSnapshot {
                    alive: state.alive,
                    consecutive_failures: state.consecutive_failures,
                    cache_key_hits: state.cache_key_hits.load(Ordering::Relaxed),
                    last_failure_at: state.last_failure_at,
                    last_error: state.last_error.clone(),
                    unhealthy_until: state.unhealthy_until,
                    recovery_probe_due: state.recovery_probe_due,
                    probe_in_progress: state.probe_in_progress.load(Ordering::Relaxed),
                },
            )
        })
    }

    pub fn mark_success(&self, index: usize) {
        let guard = self.accounts.read();
        if let Some((account, state)) = guard.get(index) {
            let mut state = state.write();
            state.alive = true;
            state.consecutive_failures = 0;
            state.last_error = None;
            state.unhealthy_until = None;
            state.recovery_probe_due = false;
            state.probe_in_progress.store(false, Ordering::Release);
            debug!("Account {} marked healthy", account.id);
        }
    }

    pub fn mark_failure(&self, index: usize, is_auth_error: bool, error: Option<&str>) {
        let guard = self.accounts.read();
        if let Some((account, state)) = guard.get(index) {
            let mut state = state.write();
            let health = self.health.read().clone();
            let threshold = health.failure_threshold.max(1);
            let now = SystemTime::now();
            state.last_failure_at = Some(now);
            if let Some(error) = error {
                state.last_error = Some(error.to_string());
            }
            state.consecutive_failures += 1;

            let backoff_factor = state
                .consecutive_failures
                .saturating_sub(1)
                .min(threshold.saturating_sub(1));
            let cooldown_multiplier = 1u64 << backoff_factor;
            let cooldown_seconds = health.cooldown_seconds.saturating_mul(cooldown_multiplier);

            let should_mark_unhealthy = (is_auth_error && health.auth_failure_immediate_unhealthy)
                || state.consecutive_failures >= threshold;
            if should_mark_unhealthy {
                state.alive = false;
                state.recovery_probe_due = true;
                state.unhealthy_until = Some(now + Duration::from_secs(cooldown_seconds));
                state.probe_in_progress.store(false, Ordering::Release);
                warn!(
                    "Account {} marked unhealthy (failures={}, auth_error={}, backoff={}s)",
                    account.id, state.consecutive_failures, is_auth_error, cooldown_seconds
                );
            } else {
                state.alive = true;
                state.recovery_probe_due = false;
                state.unhealthy_until = None;
                debug!(
                    "Account {} recorded failure {}/{} but remains healthy",
                    account.id, state.consecutive_failures, threshold
                );
            }
        }
    }

    pub fn mark_rate_limited(
        &self,
        index: usize,
        retry_after: Option<Duration>,
        error: Option<&str>,
    ) {
        let guard = self.accounts.read();
        if let Some((account, state)) = guard.get(index) {
            let mut state = state.write();
            let health = self.health.read().clone();
            let now = SystemTime::now();
            state.last_failure_at = Some(now);
            if let Some(error) = error {
                state.last_error = Some(error.to_string());
            }
            state.consecutive_failures += 1;
            state.alive = false;
            state.recovery_probe_due = true;
            state.probe_in_progress.store(false, Ordering::Release);
            let cooldown =
                retry_after.unwrap_or_else(|| Duration::from_secs(health.cooldown_seconds));
            state.unhealthy_until = Some(now + cooldown);
            warn!(
                "Account {} marked rate limited (failures={}, cooldown={}s)",
                account.id,
                state.consecutive_failures,
                cooldown.as_secs()
            );
        }
    }

    pub fn mark_nonfatal_failure(&self, index: usize, error: Option<&str>) {
        let guard = self.accounts.read();
        if let Some((account, state)) = guard.get(index) {
            let mut state = state.write();
            let now = SystemTime::now();
            state.last_failure_at = Some(now);
            if let Some(error) = error {
                state.last_error = Some(error.to_string());
            }
            state.alive = true;
            state.recovery_probe_due = false;
            state.unhealthy_until = None;
            state.probe_in_progress.store(false, Ordering::Release);
            debug!("Account {} recorded nonfatal failure", account.id);
        }
    }

    pub fn increment_cache_hits(&self, index: usize) {
        let guard = self.accounts.read();
        if let Some((_, state)) = guard.get(index) {
            state.read().cache_key_hits.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn all_accounts_snapshot(&self) -> Vec<AccountStatus> {
        let guard = self.accounts.read();
        guard
            .iter()
            .map(|(account, state)| {
                let state = state.read();
                AccountStatus {
                    id: account.id.clone(),
                    models: account.models.clone(),
                    weight: account.weight,
                    auth: mask_auth(&account.auth),
                    alive: state.alive,
                    consecutive_failures: state.consecutive_failures,
                    cache_key_hits: state.cache_key_hits.load(Ordering::Relaxed),
                    last_failure_at: state.last_failure_at,
                    last_error: state.last_error.clone(),
                    unhealthy_until: state.unhealthy_until,
                    recovery_probe_due: state.recovery_probe_due,
                    probe_in_progress: state.probe_in_progress.load(Ordering::Relaxed),
                }
            })
            .collect()
    }

    pub fn account_count(&self) -> usize {
        self.accounts.read().len()
    }
}

impl Default for AccountPool {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AccountSnapshot {
    pub alive: bool,
    pub consecutive_failures: u32,
    pub cache_key_hits: u64,
    pub last_failure_at: Option<SystemTime>,
    pub last_error: Option<String>,
    pub unhealthy_until: Option<SystemTime>,
    pub recovery_probe_due: bool,
    pub probe_in_progress: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct MaskedAccountAuth {
    pub auth_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_masked: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub creds_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_id_masked: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_secret_masked: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AccountStatus {
    pub id: String,
    pub models: Option<Vec<String>>,
    pub weight: u32,
    pub auth: MaskedAccountAuth,
    pub alive: bool,
    pub consecutive_failures: u32,
    pub cache_key_hits: u64,
    pub last_failure_at: Option<SystemTime>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    pub unhealthy_until: Option<SystemTime>,
    pub recovery_probe_due: bool,
    pub probe_in_progress: bool,
}

fn sanitize_error_for_log(message: &str) -> String {
    let mut out = message.replace('\n', "\\n").replace('\r', "\\r");
    out.truncate(1024);
    out
}

fn mask_auth(auth: &AccountAuth) -> MaskedAccountAuth {
    match auth {
        AccountAuth::ApiKey { api_key } => MaskedAccountAuth {
            auth_type: "api_key".into(),
            api_key_masked: Some(mask_key(api_key)),
            creds_path: None,
            client_id_masked: None,
            client_secret_masked: None,
        },
    }
}

fn mask_key(key: &str) -> String {
    if key.len() <= 8 {
        return "***".into();
    }
    format!("{}...{}", &key[..4], &key[key.len() - 4..])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn account(id: &str, models: Option<Vec<&str>>, weight: u32) -> Account {
        Account {
            id: id.into(),
            auth: AccountAuth::ApiKey {
                api_key: "test-key".into(),
            },
            enabled: true,
            weight,
            models: models.map(|values| values.into_iter().map(str::to_string).collect()),
        }
    }

    fn target(model: &str) -> RouteTargetConfig {
        RouteTargetConfig {
            model: model.into(),
            reasoning: None,
        }
    }

    #[test]
    fn filters_accounts_by_model_capability() {
        let pool = AccountPool::new();
        pool.load_accounts(vec![
            account("a", Some(vec!["glm-4.6"]), 1),
            account("b", Some(vec!["glm-4.6-fast"]), 1),
        ]);
        pool.mark_success(0);
        pool.mark_success(1);

        let compatible = pool.healthy_compatible_accounts_for_target(&target("glm-4.6"));
        assert_eq!(compatible, vec![0]);
    }

    #[test]
    fn keeps_account_healthy_until_failure_threshold_is_reached() {
        let pool = AccountPool::new();
        pool.load_accounts(vec![account("a", None, 1)]);
        pool.mark_success(0);

        pool.mark_failure(0, false, Some("boom-1"));
        let (_, snapshot) = pool.get_account(0).unwrap();
        assert!(snapshot.alive);
        assert!(!snapshot.recovery_probe_due);
        assert!(snapshot.unhealthy_until.is_none());
        assert_eq!(snapshot.consecutive_failures, 1);
        assert_eq!(snapshot.last_error.as_deref(), Some("boom-1"));

        pool.mark_failure(0, false, Some("boom-2"));
        let (_, snapshot) = pool.get_account(0).unwrap();
        assert!(snapshot.alive);
        assert_eq!(snapshot.consecutive_failures, 2);

        pool.mark_failure(0, false, Some("boom-3"));
        let (_, snapshot) = pool.get_account(0).unwrap();
        assert!(!snapshot.alive);
        assert!(snapshot.recovery_probe_due);
        assert!(snapshot.unhealthy_until.is_some());
        assert_eq!(snapshot.consecutive_failures, 3);
        assert_eq!(snapshot.last_error.as_deref(), Some("boom-3"));
    }

    #[test]
    fn marks_auth_failure_unhealthy_immediately() {
        let pool = AccountPool::new();
        pool.load_accounts(vec![account("a", None, 1)]);
        pool.mark_success(0);
        pool.mark_failure(0, true, Some("auth boom"));
        let (_, snapshot) = pool.get_account(0).unwrap();
        assert!(!snapshot.alive);
        assert!(snapshot.recovery_probe_due);
        assert!(snapshot.unhealthy_until.is_some());
        assert_eq!(snapshot.consecutive_failures, 1);
        assert_eq!(snapshot.last_error.as_deref(), Some("auth boom"));
    }

    #[test]
    fn marks_rate_limit_unhealthy_immediately() {
        let pool = AccountPool::new();
        pool.load_accounts(vec![account("a", None, 1)]);
        pool.mark_success(0);
        pool.mark_rate_limited(0, Some(Duration::from_secs(2)), Some("too many requests"));
        let (_, snapshot) = pool.get_account(0).unwrap();
        assert!(!snapshot.alive);
        assert!(snapshot.recovery_probe_due);
        assert!(snapshot.unhealthy_until.is_some());
        assert_eq!(snapshot.consecutive_failures, 1);
        assert_eq!(snapshot.last_error.as_deref(), Some("too many requests"));
    }
}
