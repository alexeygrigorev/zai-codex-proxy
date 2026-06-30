use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::account_pool::{AccountPool, RoutingState};
use crate::config::ConfigHandle;
use crate::providers::zai::ZAIProvider;
use crate::session::SessionStore;
use crate::usage::UsageStore;

#[derive(Clone)]
pub struct AppState {
    inner: Arc<AppStateInner>,
}

struct AppStateInner {
    pub config: ConfigHandle,
    pub accounts: AccountPool,
    pub routing: RoutingState,
    pub usage: UsageStore,
    pub zai: ZAIProvider,
    pub recovery_probes_started: AtomicBool,
    pub sessions: SessionStore,
}

impl AppState {
    pub fn new(config: ConfigHandle) -> Self {
        let ttl_seconds =
            crate::config::with_config(&config, |cfg| cfg.session.response_id_ttl_seconds);
        let sessions = SessionStore::new(std::time::Duration::from_secs(ttl_seconds));
        Self {
            inner: Arc::new(AppStateInner {
                config: config.clone(),
                accounts: AccountPool::new(),
                routing: RoutingState::new(),
                usage: UsageStore::default(),
                zai: ZAIProvider::new(),
                recovery_probes_started: AtomicBool::new(false),
                sessions,
            }),
        }
    }

    pub fn config(&self) -> &ConfigHandle {
        &self.inner.config
    }

    pub fn accounts(&self) -> &AccountPool {
        &self.inner.accounts
    }

    pub fn routing(&self) -> &RoutingState {
        &self.inner.routing
    }

    pub fn usage(&self) -> &UsageStore {
        &self.inner.usage
    }

    pub fn zai(&self) -> &ZAIProvider {
        &self.inner.zai
    }

    pub fn recovery_started_flag(&self) -> &AtomicBool {
        &self.inner.recovery_probes_started
    }

    pub fn sessions(&self) -> &SessionStore {
        &self.inner.sessions
    }
}
