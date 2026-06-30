use parking_lot::RwLock;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

use crate::account_pool::AccountAuth;
use crate::error::ConfigError;

fn validate_port(port_str: &str) -> Result<u16, ConfigError> {
    let port: u16 = port_str
        .parse()
        .map_err(|_| ConfigError::InvalidPort(port_str.into()))?;
    if port == 0 {
        return Err(ConfigError::InvalidPort("port must be 1-65535".into()));
    }
    Ok(port)
}

fn validate_url(url: &str, name: &str) -> Result<String, ConfigError> {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(ConfigError::InvalidUrl(format!(
            "{name} must start with http:// or https://"
        )));
    }
    Ok(url.to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EffortLevel {
    pub budget: u64,
    pub level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReasoningConfig {
    pub effort_levels: HashMap<String, EffortLevel>,
    #[serde(default)]
    pub default_effort: Option<String>,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        let mut effort_levels = HashMap::new();
        effort_levels.insert(
            "none".into(),
            EffortLevel {
                budget: 0,
                level: "LOW".into(),
            },
        );
        effort_levels.insert(
            "minimal".into(),
            EffortLevel {
                budget: 2048,
                level: "LOW".into(),
            },
        );
        effort_levels.insert(
            "low".into(),
            EffortLevel {
                budget: 4096,
                level: "LOW".into(),
            },
        );
        effort_levels.insert(
            "medium".into(),
            EffortLevel {
                budget: 16384,
                level: "MEDIUM".into(),
            },
        );
        effort_levels.insert(
            "high".into(),
            EffortLevel {
                budget: 32768,
                level: "HIGH".into(),
            },
        );
        effort_levels.insert(
            "xhigh".into(),
            EffortLevel {
                budget: 65536,
                level: "HIGH".into(),
            },
        );
        Self {
            effort_levels,
            default_effort: Some("medium".into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EffectiveReasoningConfig {
    pub budget: u64,
    pub level: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RouteReasoningConfig {
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub budget: Option<u64>,
    #[serde(default)]
    pub level: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub log_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZaiProviderConfig {
    pub api_url: String,
    #[serde(default)]
    pub models: Vec<String>,
}

impl ZaiProviderConfig {
    pub fn models(&self) -> &Vec<String> {
        &self.models
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_url(&self.api_url, "zai.api_url")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelsConfig {
    #[serde(default)]
    pub served: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    #[serde(default = "default_session_header_name")]
    pub header_name: String,
    #[serde(default = "default_session_metadata_key")]
    pub metadata_key: String,
    #[serde(default = "default_session_response_id_ttl_seconds")]
    pub response_id_ttl_seconds: u64,
}

fn default_session_header_name() -> String {
    "x-codex-proxy-session".into()
}

fn default_session_metadata_key() -> String {
    "codex_proxy_session".into()
}

fn default_session_response_id_ttl_seconds() -> u64 {
    24 * 60 * 60
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            header_name: default_session_header_name(),
            metadata_key: default_session_metadata_key(),
            response_id_ttl_seconds: default_session_response_id_ttl_seconds(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoCompactionConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_auto_compaction_max_attempts")]
    pub max_attempts_per_request: u32,
    #[serde(default = "default_auto_compaction_tail_items")]
    pub tail_items_to_keep: usize,
}

fn default_auto_compaction_max_attempts() -> u32 {
    1
}

fn default_auto_compaction_tail_items() -> usize {
    8
}

pub const AUTO_COMPACTION_COMPACT_INSTRUCTIONS: &str = "Compact the conversation history for continued use. Preserve all tool and file context needed to continue the session.";

pub const AUTO_COMPACTION_SUMMARY_INSTRUCTIONS: &str = "Summarize the conversation history so far for continued use. Preserve key decisions, constraints, open tasks, file paths, and relevant technical details. Be concise but complete.";

impl Default for AutoCompactionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_attempts_per_request: default_auto_compaction_max_attempts(),
            tail_items_to_keep: default_auto_compaction_tail_items(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricingConfig {
    #[serde(default)]
    pub input_per_mtoken: Option<f64>,
    #[serde(default)]
    pub output_per_mtoken: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelMetadataConfig {
    #[serde(default)]
    pub context_window: Option<u32>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub pricing: Option<ModelPricingConfig>,
}

pub type ModelMetadataConfigMap = HashMap<String, ModelMetadataConfig>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouteTargetConfig {
    pub model: String,
    #[serde(default)]
    pub reasoning: Option<RouteReasoningConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelRouteStepConfig {
    Physical {
        model: String,
        #[serde(default)]
        reasoning: Option<RouteReasoningConfig>,
    },
    Logical {
        model: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRouteStepWire {
    Shorthand(String),
    Structured(ModelRouteStepConfig),
}

impl Serialize for ModelRouteStepWire {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Shorthand(input) => serializer.serialize_str(input),
            Self::Structured(step) => step.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ModelRouteStepWire {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        if let Some(input) = value.as_str() {
            return Ok(Self::Shorthand(input.to_string()));
        }

        let serde_json::Value::Object(mut object) = value else {
            return Err(serde::de::Error::custom(
                "route step must be a string or object",
            ));
        };

        let step_type = object
            .remove("type")
            .and_then(|value| value.as_str().map(str::to_string))
            .ok_or_else(|| serde::de::Error::custom("route step object requires string 'type'"))?;
        let model = object
            .remove("model")
            .and_then(|value| value.as_str().map(str::to_string))
            .ok_or_else(|| serde::de::Error::custom("route step object requires string 'model'"))?;
        let reasoning = object
            .remove("reasoning")
            .map(serde_json::from_value)
            .transpose()
            .map_err(serde::de::Error::custom)?;

        if !object.is_empty() {
            let mut keys = object.keys().cloned().collect::<Vec<_>>();
            keys.sort();
            return Err(serde::de::Error::custom(format!(
                "unknown route step field(s): {}",
                keys.join(", ")
            )));
        }

        match step_type.as_str() {
            "physical" => Ok(Self::Structured(ModelRouteStepConfig::Physical {
                model,
                reasoning,
            })),
            "logical" => {
                if reasoning.is_some() {
                    return Err(serde::de::Error::custom(
                        "logical route step must not set reasoning",
                    ));
                }
                Ok(Self::Structured(ModelRouteStepConfig::Logical { model }))
            }
            other => Err(serde::de::Error::custom(format!(
                "unsupported route step type '{other}'"
            ))),
        }
    }
}

impl ModelRouteStepWire {
    fn into_step(self) -> Result<ModelRouteStepConfig, ConfigError> {
        let input = match self {
            ModelRouteStepWire::Structured(step) => return Ok(step),
            ModelRouteStepWire::Shorthand(input) => input,
        };

        let input = input.trim();
        if input.is_empty() {
            return Err(ConfigError::InvalidValue(
                "route step shorthand must not be empty".into(),
            ));
        }
        let Some((prefix, rest)) = input.split_once(':') else {
            return Ok(ModelRouteStepConfig::Physical {
                model: input.to_string(),
                reasoning: None,
            });
        };
        let prefix = prefix.trim();
        let rest = rest.trim();
        if prefix.is_empty() || rest.is_empty() {
            return Err(ConfigError::InvalidValue(format!(
                "invalid route step shorthand '{input}': expected 'model' or 'proxy:logical_model'"
            )));
        }
        if prefix == "proxy" {
            return Ok(ModelRouteStepConfig::Logical {
                model: rest.to_string(),
            });
        }
        Err(ConfigError::InvalidValue(format!(
            "invalid route step shorthand '{input}': only the 'proxy' prefix is supported"
        )))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoutingHealthConfig {
    #[serde(default = "default_auth_failure_immediate_unhealthy")]
    pub auth_failure_immediate_unhealthy: bool,
    #[serde(default = "default_failure_threshold")]
    pub failure_threshold: u32,
    #[serde(default = "default_cooldown_seconds")]
    pub cooldown_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingConfig {
    #[serde(default)]
    pub model_routes: HashMap<String, Vec<ModelRouteStepConfig>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingConfigWire {
    #[serde(default)]
    pub model_routes: HashMap<String, Vec<ModelRouteStepWire>>,
}

impl RoutingConfigWire {
    pub fn into_runtime(self) -> Result<RoutingConfig, ConfigError> {
        let mut model_routes = HashMap::with_capacity(self.model_routes.len());
        for (route_key, steps) in self.model_routes {
            let mut out_steps = Vec::with_capacity(steps.len());
            for (idx, step) in steps.into_iter().enumerate() {
                let scope = format!("routing.model_routes['{}'][{}]", route_key, idx);
                let parsed = step
                    .into_step()
                    .map_err(|e| ConfigError::InvalidValue(format!("{scope}: {e}")))?;
                out_steps.push(parsed);
            }
            model_routes.insert(route_key, out_steps);
        }
        Ok(RoutingConfig { model_routes })
    }
}

impl RoutingConfig {
    pub fn to_wire(&self) -> RoutingConfigWire {
        let mut model_routes = HashMap::with_capacity(self.model_routes.len());
        for (route_key, steps) in &self.model_routes {
            model_routes.insert(
                route_key.clone(),
                steps
                    .iter()
                    .cloned()
                    .map(ModelRouteStepWire::Structured)
                    .collect(),
            );
        }
        RoutingConfigWire { model_routes }
    }
}

impl Default for RoutingHealthConfig {
    fn default() -> Self {
        Self {
            auth_failure_immediate_unhealthy: true,
            failure_threshold: default_failure_threshold(),
            cooldown_seconds: default_cooldown_seconds(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutsConfig {
    pub connect_seconds: u64,
    pub read_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    pub temperature: f64,
    #[serde(default)]
    pub preferred_targets: Vec<String>,
}

fn default_retry_max_attempts() -> u32 {
    3
}
fn default_retry_initial_delay_ms() -> u64 {
    1000
}
fn default_retry_max_delay_ms() -> u64 {
    30000
}
fn default_retry_backoff_multiplier() -> f64 {
    2.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_retry_max_attempts")]
    pub max_attempts: u32,
    #[serde(default = "default_retry_initial_delay_ms")]
    pub initial_delay_ms: u64,
    #[serde(default = "default_retry_max_delay_ms")]
    pub max_delay_ms: u64,
    #[serde(default = "default_retry_backoff_multiplier")]
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_attempts: default_retry_max_attempts(),
            initial_delay_ms: default_retry_initial_delay_ms(),
            max_delay_ms: default_retry_max_delay_ms(),
            backoff_multiplier: default_retry_backoff_multiplier(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AccessKeyRole {
    Admin,
    Api,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessKeyConfig {
    pub id: String,
    #[serde(default)]
    pub key_sha256: String,
    #[serde(default, skip_serializing)]
    pub plaintext: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub role: Option<AccessKeyRole>,
    #[serde(default, skip_serializing)]
    pub is_admin: bool,
}

impl AccessKeyConfig {
    pub fn effective_role(&self) -> AccessKeyRole {
        self.role.unwrap_or({
            if self.is_admin {
                AccessKeyRole::Admin
            } else {
                AccessKeyRole::Api
            }
        })
    }

    pub fn normalize_secret(&mut self) -> Result<(), ConfigError> {
        let has_hash = !self.key_sha256.trim().is_empty();
        let plaintext = self.plaintext.as_deref().unwrap_or("").trim();
        let has_plaintext = !plaintext.is_empty();

        match (has_hash, has_plaintext) {
            (true, true) => Err(ConfigError::InvalidValue(format!(
                "access.keys['{}'] must set only one of key_sha256 or plaintext",
                self.id
            ))),
            (false, false) => Err(ConfigError::InvalidValue(format!(
                "access.keys['{}'] must set key_sha256 or plaintext",
                self.id
            ))),
            (false, true) => {
                self.key_sha256 = crate::access::sha256_hex(plaintext);
                self.plaintext = None;
                Ok(())
            }
            (true, false) => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AccessControlConfig {
    #[serde(default)]
    pub require_key: bool,
    #[serde(default)]
    pub keys: Vec<AccessKeyConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountConfig {
    pub id: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_weight")]
    pub weight: u32,
    #[serde(default)]
    pub models: Option<Vec<String>>,
    pub auth: AccountAuth,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub config_path: PathBuf,
    pub server: ServerConfig,
    pub zai: ZaiProviderConfig,
    pub models: ModelsConfig,
    pub model_metadata: ModelMetadataConfigMap,
    pub session: SessionConfig,
    pub auto_compaction: AutoCompactionConfig,
    pub routing: RoutingConfig,
    pub health: RoutingHealthConfig,
    pub accounts: Vec<AccountConfig>,
    pub access: AccessControlConfig,
    pub reasoning: ReasoningConfig,
    pub timeouts: TimeoutsConfig,
    pub compaction: CompactionConfig,
    pub retry: RetryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedConfig {
    pub server: ServerConfig,
    pub zai: ZaiProviderConfig,
    pub models: ModelsConfig,
    #[serde(default)]
    pub model_metadata: ModelMetadataConfigMap,
    #[serde(default)]
    pub session: SessionConfig,
    #[serde(default)]
    pub auto_compaction: AutoCompactionConfig,
    pub routing: RoutingConfigWire,
    #[serde(default)]
    pub health: RoutingHealthConfig,
    pub accounts: Vec<AccountConfig>,
    #[serde(default)]
    pub access: AccessControlConfig,
    pub reasoning: ReasoningConfig,
    pub timeouts: TimeoutsConfig,
    pub compaction: CompactionConfig,
    #[serde(default)]
    pub retry: RetryConfig,
}

impl PersistedConfig {
    pub fn into_runtime(mut self, config_path: PathBuf) -> Config {
        for key in &mut self.access.keys {
            key.plaintext = None;
        }
        Config {
            config_path,
            server: self.server,
            zai: self.zai,
            models: self.models,
            model_metadata: self.model_metadata,
            session: self.session,
            auto_compaction: self.auto_compaction,
            routing: self
                .routing
                .into_runtime()
                .expect("invalid routing configuration"),
            health: self.health,
            accounts: self.accounts,
            access: self.access,
            reasoning: self.reasoning,
            timeouts: self.timeouts,
            compaction: self.compaction,
            retry: self.retry,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct FileConfig {
    #[serde(default)]
    pub server: Option<ServerConfig>,
    #[serde(default)]
    pub zai: Option<ZaiProviderConfig>,
    #[serde(default)]
    pub models: Option<ModelsConfig>,
    #[serde(default)]
    pub model_metadata: Option<ModelMetadataConfigMap>,
    #[serde(default)]
    pub session: Option<SessionConfig>,
    #[serde(default)]
    pub auto_compaction: Option<AutoCompactionConfig>,
    #[serde(default)]
    pub routing: Option<RoutingConfigWire>,
    #[serde(default)]
    pub health: Option<RoutingHealthConfig>,
    #[serde(default)]
    pub accounts: Option<Vec<AccountConfig>>,
    #[serde(default)]
    pub access: Option<AccessControlConfig>,
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    pub timeouts: Option<TimeoutsConfig>,
    #[serde(default)]
    pub compaction: Option<CompactionConfig>,
    #[serde(default)]
    pub retry: Option<RetryConfig>,
}

pub type ConfigHandle = Arc<RwLock<Config>>;

pub fn with_config<T>(handle: &ConfigHandle, f: impl FnOnce(&Config) -> T) -> T {
    let guard = handle.read();
    f(&guard)
}

pub fn with_config_mut<T>(handle: &ConfigHandle, f: impl FnOnce(&mut Config) -> T) -> T {
    let mut guard = handle.write();
    f(&mut guard)
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    pub fn new() -> Self {
        let mut cfg = Self::defaults();
        let loaded = cfg.load_from_file();
        if !loaded {
            let tried = default_config_search_paths(&dirs_home())
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            panic!("No config file found. Tried: {tried}");
        }
        cfg.validate().expect("invalid configuration");
        cfg
    }

    pub fn new_from_path(path: impl AsRef<Path>) -> Self {
        let mut cfg = Self::defaults();
        cfg.load_from_path(path.as_ref());
        cfg.validate().expect("invalid configuration");
        cfg
    }

    fn defaults() -> Self {
        let home = dirs_home();
        let host = env::var("CODEX_PROXY_HOST").unwrap_or_else(|_| "127.0.0.1".into());
        let port = env::var("CODEX_PROXY_PORT")
            .map(|p| validate_port(&p).unwrap_or(8765))
            .unwrap_or(8765);
        let log_level = env::var("CODEX_PROXY_LOG_LEVEL")
            .unwrap_or_else(|_| "DEBUG".into())
            .to_uppercase();

        let z_ai_url = env::var("CODEX_PROXY_ZAI_URL")
            .unwrap_or_else(|_| "https://api.z.ai/api/coding/paas/v4/chat/completions".into());

        let served_models: Vec<String> = env::var("CODEX_PROXY_MODELS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let mut accounts = Vec::new();
        // Prefer the explicit proxy-specific key, but fall back to the generic
        // `ZAI_API_KEY` so the proxy starts correctly whether it is launched by
        // a systemd unit (EnvironmentFile=~/.zodex/zai.env) or by hand.
        let zai_api_key = env::var("CODEX_PROXY_ZAI_API_KEY")
            .ok()
            .filter(|key| !key.is_empty())
            .or_else(|| env::var("ZAI_API_KEY").ok().filter(|key| !key.is_empty()));
        if let Some(key) = zai_api_key {
            accounts.push(AccountConfig {
                id: "zai-default".into(),
                enabled: true,
                weight: 1,
                models: None,
                auth: AccountAuth::ApiKey { api_key: key },
            });
        }

        let zai = ZaiProviderConfig {
            api_url: validate_url(&z_ai_url, "Z.AI URL").unwrap(),
            models: Vec::new(),
        };

        Self {
            config_path: default_config_search_paths(&home)[0].clone(),
            server: ServerConfig {
                host,
                port,
                log_level,
            },
            zai,
            models: ModelsConfig {
                served: served_models,
            },
            model_metadata: ModelMetadataConfigMap::new(),
            session: SessionConfig::default(),
            auto_compaction: AutoCompactionConfig::default(),
            routing: RoutingConfig {
                model_routes: HashMap::new(),
            },
            health: RoutingHealthConfig::default(),
            accounts,
            access: AccessControlConfig::default(),
            reasoning: ReasoningConfig::default(),
            timeouts: TimeoutsConfig {
                connect_seconds: 10,
                read_seconds: 600,
            },
            compaction: CompactionConfig {
                temperature: 0.1,
                preferred_targets: Vec::new(),
            },
            retry: RetryConfig::default(),
        }
    }

    fn load_from_path(&mut self, path: &Path) {
        if !path.exists() {
            panic!(
                "Config {} does not exist. Refusing to start.",
                path.display()
            );
        }
        let content = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read config {}: {e}", path.display()));
        let file_cfg: FileConfig = parse_json_with_path(&content)
            .unwrap_or_else(|e| panic!("Failed to parse config {}: {e}", path.display()));
        if file_cfg.access.is_none() {
            panic!(
                "Config {} is missing required 'access' section. Refusing to start.",
                path.display()
            );
        }

        if let Some(server) = file_cfg.server {
            self.server = server;
        }
        if let Some(zai) = file_cfg.zai {
            self.zai = zai;
        }
        if let Some(models) = file_cfg.models {
            self.models = models;
        }
        if let Some(model_metadata) = file_cfg.model_metadata {
            self.model_metadata = model_metadata;
        }
        if let Some(session) = file_cfg.session {
            self.session = session;
        }
        if let Some(auto_compaction) = file_cfg.auto_compaction {
            self.auto_compaction = auto_compaction;
        }
        if let Some(routing) = file_cfg.routing {
            self.routing = routing
                .into_runtime()
                .unwrap_or_else(|e| panic!("Invalid routing config in {}: {e}", path.display()));
        }
        if let Some(health) = file_cfg.health {
            self.health = health;
        }
        if let Some(accounts) = file_cfg.accounts {
            self.accounts = accounts;
        }
        if let Some(access) = file_cfg.access {
            self.access = access;
        }
        if let Some(reasoning) = file_cfg.reasoning {
            self.reasoning = reasoning;
        }
        if let Some(timeouts) = file_cfg.timeouts {
            self.timeouts = timeouts;
        }
        if let Some(compaction) = file_cfg.compaction {
            self.compaction = compaction;
        }
        if let Some(retry) = file_cfg.retry {
            self.retry = retry;
        }

        self.config_path = path.to_path_buf();
        info!("Loaded config from {}", self.config_path.display());
    }

    fn load_from_file(&mut self) -> bool {
        for path in default_config_search_paths(&dirs_home()) {
            if !path.exists() {
                continue;
            }
            let content = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("Failed to read config {}: {e}", path.display()));
            let file_cfg: FileConfig = parse_json_with_path(&content).unwrap_or_else(|e| {
                panic!("Failed to parse config {}: {e}", path.display());
            });
            if file_cfg.access.is_none() {
                panic!(
                    "Config {} is missing required 'access' section. Refusing to start.",
                    path.display()
                );
            }

            if let Some(server) = file_cfg.server {
                self.server = server;
            }
            if let Some(zai) = file_cfg.zai {
                self.zai = zai;
            }
            if let Some(models) = file_cfg.models {
                self.models = models;
            }
            if let Some(model_metadata) = file_cfg.model_metadata {
                self.model_metadata = model_metadata;
            }
            if let Some(session) = file_cfg.session {
                self.session = session;
            }
            if let Some(auto_compaction) = file_cfg.auto_compaction {
                self.auto_compaction = auto_compaction;
            }
            if let Some(routing) = file_cfg.routing {
                self.routing = routing.into_runtime().unwrap_or_else(|e| {
                    panic!("Invalid routing config in {}: {e}", path.display())
                });
            }
            if let Some(health) = file_cfg.health {
                self.health = health;
            }
            if let Some(accounts) = file_cfg.accounts {
                self.accounts = accounts;
            }
            if let Some(access) = file_cfg.access {
                self.access = access;
            }
            if let Some(reasoning) = file_cfg.reasoning {
                self.reasoning = reasoning;
            }
            if let Some(timeouts) = file_cfg.timeouts {
                self.timeouts = timeouts;
            }
            if let Some(compaction) = file_cfg.compaction {
                self.compaction = compaction;
            }
            if let Some(retry) = file_cfg.retry {
                self.retry = retry;
            }

            self.config_path = path.clone();
            info!("Loaded config from {}", self.config_path.display());
            return true;
        }
        false
    }

    pub fn to_persisted(&self) -> PersistedConfig {
        PersistedConfig {
            server: self.server.clone(),
            zai: self.zai.clone(),
            models: self.models.clone(),
            model_metadata: self.model_metadata.clone(),
            session: self.session.clone(),
            auto_compaction: self.auto_compaction.clone(),
            routing: self.routing.to_wire(),
            health: self.health.clone(),
            accounts: self.accounts.clone(),
            access: self.access.clone(),
            reasoning: self.reasoning.clone(),
            timeouts: self.timeouts.clone(),
            compaction: self.compaction.clone(),
            retry: self.retry.clone(),
        }
    }

    pub fn save_to_path(&self, path: &Path) -> Result<(), ConfigError> {
        let persisted = self.to_persisted();
        let json = serde_json::to_string_pretty(&persisted)?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let tmp_path = parent.join(format!(
            ".{}.tmp",
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("config.json")
        ));
        fs::write(&tmp_path, json)?;
        fs::rename(&tmp_path, path)?;
        Ok(())
    }

    pub fn validate(&mut self) -> Result<(), ConfigError> {
        self.zai.validate()?;

        for (model, metadata) in &self.model_metadata {
            if model.trim().is_empty() {
                return Err(ConfigError::InvalidValue(
                    "model_metadata contains an empty model id".into(),
                ));
            }
            if let Some(value) = metadata.context_window
                && value == 0
            {
                return Err(ConfigError::InvalidValue(format!(
                    "model_metadata['{}'].context_window must be > 0",
                    model
                )));
            }
            if let Some(value) = metadata.max_output_tokens
                && value == 0
            {
                return Err(ConfigError::InvalidValue(format!(
                    "model_metadata['{}'].max_output_tokens must be > 0",
                    model
                )));
            }
            if let Some(pricing) = &metadata.pricing {
                if let Some(v) = pricing.input_per_mtoken
                    && v < 0.0
                {
                    return Err(ConfigError::InvalidValue(format!(
                        "model_metadata['{}'].pricing.input_per_mtoken must be >= 0",
                        model
                    )));
                }
                if let Some(v) = pricing.output_per_mtoken
                    && v < 0.0
                {
                    return Err(ConfigError::InvalidValue(format!(
                        "model_metadata['{}'].pricing.output_per_mtoken must be >= 0",
                        model
                    )));
                }
            }
        }

        if self.server.port == 0 {
            return Err(ConfigError::InvalidPort("port must be 1-65535".into()));
        }
        if let Some(default_effort) = &self.reasoning.default_effort
            && !self.reasoning.effort_levels.contains_key(default_effort)
        {
            return Err(ConfigError::InvalidValue(format!(
                "reasoning.default_effort '{}' is not defined in reasoning.effort_levels",
                default_effort
            )));
        }

        let mut seen_access_ids = HashSet::new();
        let mut enabled_access_key_count = 0usize;
        for key in &mut self.access.keys {
            if !seen_access_ids.insert(key.id.clone()) {
                return Err(ConfigError::InvalidValue(format!(
                    "duplicate access key id: {}",
                    key.id
                )));
            }
            if key.enabled {
                enabled_access_key_count += 1;
            }
            key.normalize_secret()?;
            let hash = key.key_sha256.trim();
            let is_hex_64 = hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit());
            if !is_hex_64 {
                return Err(ConfigError::InvalidValue(format!(
                    "access.keys['{}'] key_sha256 must be 64 hex chars",
                    key.id
                )));
            }
        }
        if self.access.require_key && enabled_access_key_count == 0 {
            return Err(ConfigError::InvalidValue(
                "access.require_key is true but no enabled access keys are configured".into(),
            ));
        }

        let mut seen_ids = HashSet::new();
        let enabled_accounts: Vec<&AccountConfig> =
            self.accounts.iter().filter(|a| a.enabled).collect();
        if enabled_accounts.is_empty() {
            return Err(ConfigError::InvalidValue(
                "accounts must contain at least one enabled account".into(),
            ));
        }
        for account in &self.accounts {
            if !seen_ids.insert(account.id.clone()) {
                return Err(ConfigError::InvalidValue(format!(
                    "duplicate account id: {}",
                    account.id
                )));
            }
            match &account.auth {
                AccountAuth::ApiKey { api_key } => {
                    if api_key.is_empty() {
                        return Err(ConfigError::InvalidValue(format!(
                            "account '{}' has empty api_key auth",
                            account.id
                        )));
                    }
                }
            }

            if let Some(models) = &account.models {
                if models.is_empty() {
                    return Err(ConfigError::InvalidValue(format!(
                        "account '{}' models restriction must not be empty when present",
                        account.id
                    )));
                }
                if let Some(provider_models) = self.zai_catalog() {
                    for model in models {
                        if !provider_models.contains(model) {
                            return Err(ConfigError::InvalidValue(format!(
                                "account '{}' references model '{}' not present in zai.models",
                                account.id, model
                            )));
                        }
                    }
                }
            }
        }

        if self.routing.model_routes.is_empty() {
            return Err(ConfigError::InvalidValue(
                "routing.model_routes must not be empty".into(),
            ));
        }

        for (route_key, steps) in &self.routing.model_routes {
            if route_key.trim().is_empty() {
                return Err(ConfigError::InvalidValue(
                    "routing.model_routes keys must not be empty".into(),
                ));
            }
            if steps.is_empty() {
                return Err(ConfigError::InvalidValue(format!(
                    "routing.model_routes['{}'] must not be empty",
                    route_key
                )));
            }

            for (idx, step) in steps.iter().enumerate() {
                match step {
                    ModelRouteStepConfig::Physical { model, reasoning } => {
                        let target = RouteTargetConfig {
                            model: model.clone(),
                            reasoning: reasoning.clone(),
                        };
                        self.validate_route_target(
                            &target,
                            &format!("routing.model_routes['{}'][{}]", route_key, idx),
                        )?;
                    }
                    ModelRouteStepConfig::Logical { model } => {
                        if model.trim().is_empty() {
                            return Err(ConfigError::InvalidValue(format!(
                                "routing.model_routes['{}'][{}] logical model reference must not be empty",
                                route_key, idx
                            )));
                        }
                        if !self.routing.model_routes.contains_key(model.as_str()) {
                            return Err(ConfigError::InvalidValue(format!(
                                "routing.model_routes['{}'][{}] references undefined logical model '{}'",
                                route_key, idx, model
                            )));
                        }
                    }
                }
            }

            let targets = self.expand_model_route(route_key)?;
            if !self.has_compatible_enabled_account(&targets) {
                return Err(ConfigError::InvalidValue(format!(
                    "routing.model_routes['{}'] has no compatible enabled account",
                    route_key
                )));
            }
        }

        for logical_model in &self.compaction.preferred_targets {
            if !self
                .routing
                .model_routes
                .contains_key(logical_model.as_str())
            {
                return Err(ConfigError::InvalidValue(format!(
                    "compaction.preferred_targets references logical model '{}' which is not defined in routing.model_routes",
                    logical_model
                )));
            }
        }

        if !self.models.served.is_empty() {
            for served_model in &self.models.served {
                let Some((logical_model, _)) = self.route_targets_for_model(served_model) else {
                    return Err(ConfigError::InvalidValue(format!(
                        "served model '{}' has no routing targets after overrides/fallbacks",
                        served_model
                    )));
                };

                let _ = logical_model;
            }
        } else if !self.routing.model_routes.contains_key("*") {
            return Err(ConfigError::InvalidValue(
                "models.served is empty, so routing.model_routes must define a '*' fallback route"
                    .into(),
            ));
        }

        Ok(())
    }

    /// Resolve a request model into a routing logical model plus its target list.
    ///
    /// Resolution order:
    /// 1) `routing.model_routes[requested_model]`
    /// 2) `routing.model_routes["*"]`
    pub fn route_targets_for_model(
        &self,
        requested_model: &str,
    ) -> Option<(String, Vec<RouteTargetConfig>)> {
        let entry_key = if self.routing.model_routes.contains_key(requested_model) {
            requested_model
        } else if self.routing.model_routes.contains_key("*") {
            "*"
        } else {
            return None;
        };

        let logical_key = self.canonical_logical_route_key(entry_key).ok()?;
        let targets = self.expand_model_route(&logical_key).ok()?;
        Some((logical_key, targets))
    }

    pub fn preferred_targets_for_model(
        &self,
        requested_model: &str,
    ) -> Option<Vec<RouteTargetConfig>> {
        self.route_targets_for_model(requested_model)
            .map(|(_, targets)| targets)
    }

    pub fn compaction_targets(&self) -> Vec<RouteTargetConfig> {
        let mut targets = Vec::new();
        for logical_model in &self.compaction.preferred_targets {
            if let Ok(expanded) = self.expand_model_route(logical_model) {
                targets.extend(expanded);
            }
        }
        targets
    }

    fn canonical_logical_route_key(&self, entry_key: &str) -> Result<String, ConfigError> {
        let mut current = entry_key.to_string();
        let mut seen = std::collections::HashSet::<String>::new();
        loop {
            let steps = self.routing.model_routes.get(&current).ok_or_else(|| {
                ConfigError::InvalidValue(format!(
                    "routing.model_routes references undefined model route '{}'",
                    current
                ))
            })?;
            if steps.len() != 1 {
                return Ok(current);
            }
            let ModelRouteStepConfig::Logical { model: next } = &steps[0] else {
                return Ok(current);
            };

            if !seen.insert(current.clone()) {
                return Err(ConfigError::InvalidValue(format!(
                    "routing.model_routes contains a logical cycle involving '{}'",
                    current
                )));
            }
            current = next.clone();
        }
    }

    fn expand_model_route(&self, route_key: &str) -> Result<Vec<RouteTargetConfig>, ConfigError> {
        let mut stack = Vec::new();
        let targets = self.expand_model_route_inner(route_key, &mut stack)?;
        if targets.is_empty() {
            return Err(ConfigError::InvalidValue(format!(
                "routing.model_routes['{}'] expands to zero physical targets",
                route_key
            )));
        }
        Ok(targets)
    }

    fn expand_model_route_inner(
        &self,
        route_key: &str,
        stack: &mut Vec<String>,
    ) -> Result<Vec<RouteTargetConfig>, ConfigError> {
        if stack.iter().any(|k| k == route_key) {
            stack.push(route_key.to_string());
            return Err(ConfigError::InvalidValue(format!(
                "routing.model_routes contains a logical cycle: {}",
                stack.join(" -> ")
            )));
        }

        let steps = self.routing.model_routes.get(route_key).ok_or_else(|| {
            ConfigError::InvalidValue(format!(
                "routing.model_routes references undefined logical model '{}'",
                route_key
            ))
        })?;

        stack.push(route_key.to_string());
        let mut out = Vec::new();
        for step in steps {
            match step {
                ModelRouteStepConfig::Physical { model, reasoning } => {
                    out.push(RouteTargetConfig {
                        model: model.clone(),
                        reasoning: reasoning.clone(),
                    })
                }
                ModelRouteStepConfig::Logical { model } => {
                    out.extend(self.expand_model_route_inner(model.as_str(), stack)?);
                }
            }
        }
        stack.pop();
        Ok(out)
    }

    pub fn recovery_probe_target(&self) -> Option<RouteTargetConfig> {
        let has_enabled_account = self.accounts.iter().any(|account| account.enabled);
        if !has_enabled_account {
            return None;
        }

        let catalog = self.zai_catalog();
        let resolve_model = |model: &str| -> Option<String> {
            let model = model.trim();
            if model.is_empty() {
                return None;
            }
            match &catalog {
                Some(catalog) => catalog
                    .iter()
                    .any(|m| m == model)
                    .then(|| model.to_string()),
                None => Some(model.to_string()),
            }
        };

        let model = self
            .accounts
            .iter()
            .filter(|account| account.enabled)
            .filter_map(|account| account.models.as_ref())
            .flat_map(|models| models.iter())
            .find_map(|model| resolve_model(model))
            .or_else(|| {
                catalog
                    .as_ref()
                    .and_then(|catalog| catalog.first())
                    .cloned()
            })?;

        Some(RouteTargetConfig {
            model,
            reasoning: None,
        })
    }

    pub fn is_served_model_allowed(&self, model: &str) -> bool {
        self.models.served.is_empty() || self.models.served.iter().any(|m| m == model)
    }

    pub fn resolve_reasoning(
        &self,
        reasoning: Option<&RouteReasoningConfig>,
    ) -> Result<Option<EffectiveReasoningConfig>, ConfigError> {
        let Some(reasoning) = reasoning else {
            return Ok(self.reasoning.default_effort.as_ref().map(|preset| {
                let cfg = self
                    .reasoning
                    .effort_levels
                    .get(preset)
                    .expect("validated default reasoning preset must exist");
                EffectiveReasoningConfig {
                    budget: cfg.budget,
                    level: cfg.level.clone(),
                    preset: Some(preset.clone()),
                }
            }));
        };

        if let Some(preset) = &reasoning.effort {
            let cfg = self.reasoning.effort_levels.get(preset).ok_or_else(|| {
                ConfigError::InvalidValue(format!(
                    "reasoning preset '{}' is not defined in reasoning.effort_levels",
                    preset
                ))
            })?;
            let budget = reasoning.budget.unwrap_or(cfg.budget);
            let level = reasoning.level.clone().unwrap_or_else(|| cfg.level.clone());
            return Ok(Some(EffectiveReasoningConfig {
                budget,
                level,
                preset: Some(preset.clone()),
            }));
        }

        if reasoning.budget.is_none() && reasoning.level.is_none() {
            return Ok(self.reasoning.default_effort.as_ref().map(|preset| {
                let cfg = self
                    .reasoning
                    .effort_levels
                    .get(preset)
                    .expect("validated default reasoning preset must exist");
                EffectiveReasoningConfig {
                    budget: cfg.budget,
                    level: cfg.level.clone(),
                    preset: Some(preset.clone()),
                }
            }));
        }

        Ok(Some(EffectiveReasoningConfig {
            budget: reasoning.budget.unwrap_or(0),
            level: reasoning.level.clone().unwrap_or_else(|| "LOW".into()),
            preset: None,
        }))
    }

    pub fn zai_catalog(&self) -> Option<&Vec<String>> {
        let models = self.zai.models();
        (!models.is_empty()).then_some(models)
    }

    pub fn model_metadata(&self, model: &str) -> Option<&ModelMetadataConfig> {
        self.model_metadata.get(model)
    }

    fn validate_route_target(
        &self,
        target: &RouteTargetConfig,
        scope: &str,
    ) -> Result<(), ConfigError> {
        if target.model.trim().is_empty() {
            return Err(ConfigError::InvalidValue(format!(
                "{scope} contains an empty target model"
            )));
        }
        if let Some(provider_models) = self.zai_catalog()
            && !provider_models.contains(&target.model)
        {
            return Err(ConfigError::InvalidValue(format!(
                "{scope} target '{}' is not present in zai.models",
                target.model
            )));
        }
        self.resolve_reasoning(target.reasoning.as_ref())?;
        Ok(())
    }

    fn has_compatible_enabled_account(&self, targets: &[RouteTargetConfig]) -> bool {
        targets.iter().any(|target| {
            self.accounts.iter().any(|account| {
                if !account.enabled {
                    return false;
                }
                match &account.models {
                    Some(models) => models.contains(&target.model),
                    None => true,
                }
            })
        })
    }
}

fn default_config_search_paths(home: &Path) -> Vec<PathBuf> {
    vec![
        PathBuf::from("config/config.json.local"),
        home.join(".config/codex-proxy/config.json"),
        PathBuf::from("config/config.json"),
    ]
}

fn default_true() -> bool {
    true
}

fn default_weight() -> u32 {
    1
}

fn default_failure_threshold() -> u32 {
    3
}

fn default_cooldown_seconds() -> u64 {
    300
}

fn default_auth_failure_immediate_unhealthy() -> bool {
    true
}

fn dirs_home() -> PathBuf {
    resolve_home_dir(
        env::var_os("HOME").map(PathBuf::from),
        env::var_os("USERPROFILE").map(PathBuf::from),
        env::var_os("HOMEDRIVE"),
        env::var_os("HOMEPATH"),
        cfg!(windows),
    )
}

fn resolve_home_dir(
    home: Option<PathBuf>,
    userprofile: Option<PathBuf>,
    homedrive: Option<OsString>,
    homepath: Option<OsString>,
    prefer_windows_env: bool,
) -> PathBuf {
    let home = home.filter(|p| !p.as_os_str().is_empty());
    let userprofile = userprofile.filter(|p| !p.as_os_str().is_empty());
    let windows_home = join_windows_home(homedrive, homepath);

    if prefer_windows_env {
        if let Some(path) = userprofile.as_ref() {
            return path.clone();
        }
        if let Some(path) = windows_home.as_ref() {
            return path.clone();
        }
    }

    if let Some(path) = home {
        return path;
    }

    if !prefer_windows_env {
        if let Some(path) = userprofile {
            return path;
        }
        if let Some(path) = windows_home {
            return path;
        }
    }

    PathBuf::from("/root")
}

fn join_windows_home(homedrive: Option<OsString>, homepath: Option<OsString>) -> Option<PathBuf> {
    let mut homedrive = homedrive.filter(|value| !value.is_empty())?;
    let homepath = homepath.filter(|value| !value.is_empty())?;
    homedrive.push(homepath);
    Some(PathBuf::from(homedrive))
}

#[cfg(test)]
mod home_dir_tests {
    use super::resolve_home_dir;
    use std::ffi::OsString;
    use std::path::PathBuf;

    #[test]
    fn resolve_home_dir_prefers_userprofile_on_windows() {
        let home = resolve_home_dir(
            Some(PathBuf::from("/root")),
            Some(PathBuf::from(r"C:\Users\woodrow")),
            None,
            None,
            true,
        );

        assert_eq!(home, PathBuf::from(r"C:\Users\woodrow"));
    }

    #[test]
    fn resolve_home_dir_falls_back_to_home_when_windows_vars_missing() {
        let home = resolve_home_dir(Some(PathBuf::from("/custom/home")), None, None, None, true);

        assert_eq!(home, PathBuf::from("/custom/home"));
    }

    #[test]
    fn resolve_home_dir_combines_home_drive_and_path_on_windows() {
        let home = resolve_home_dir(
            None,
            None,
            Some(OsString::from("C:")),
            Some(OsString::from(r"\Users\woodrow")),
            true,
        );

        assert_eq!(home, PathBuf::from(r"C:\Users\woodrow"));
    }

    #[test]
    fn resolve_home_dir_prefers_home_on_non_windows() {
        let home = resolve_home_dir(
            Some(PathBuf::from("/custom/home")),
            Some(PathBuf::from(r"C:\Users\woodrow")),
            None,
            None,
            false,
        );

        assert_eq!(home, PathBuf::from("/custom/home"));
    }
}

fn parse_json_with_path<T: DeserializeOwned>(content: &str) -> Result<T, String> {
    let mut deserializer = serde_json::Deserializer::from_str(content);
    let result: Result<T, _> = serde_path_to_error::deserialize(&mut deserializer);
    match result {
        Ok(v) => Ok(v),
        Err(e) => {
            let path = e.path().to_string();
            let inner = e.into_inner();
            if path.is_empty() {
                Err(inner.to_string())
            } else {
                Err(format!("{inner} (at {path})"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> Config {
        Config {
            config_path: PathBuf::from("/tmp/config.json"),
            server: ServerConfig {
                host: "127.0.0.1".into(),
                port: 8765,
                log_level: "INFO".into(),
            },
            zai: ZaiProviderConfig {
                api_url: "https://z.ai/chat".into(),
                models: vec!["glm-4.6".into()],
            },
            models: ModelsConfig {
                served: vec!["claude-sonnet-4-6".into()],
            },
            model_metadata: ModelMetadataConfigMap::new(),
            session: SessionConfig::default(),
            auto_compaction: AutoCompactionConfig::default(),
            routing: RoutingConfig {
                model_routes: HashMap::from([(
                    "claude-sonnet-4-6".into(),
                    vec![ModelRouteStepConfig::Physical {
                        model: "glm-4.6".into(),
                        reasoning: Some(RouteReasoningConfig {
                            effort: Some("medium".into()),
                            budget: None,
                            level: None,
                        }),
                    }],
                )]),
            },
            health: RoutingHealthConfig::default(),
            accounts: vec![AccountConfig {
                id: "zai-a".into(),
                enabled: true,
                weight: 1,
                models: Some(vec!["glm-4.6".into()]),
                auth: AccountAuth::ApiKey {
                    api_key: "sk-test".into(),
                },
            }],
            access: AccessControlConfig::default(),
            reasoning: ReasoningConfig::default(),
            timeouts: TimeoutsConfig {
                connect_seconds: 10,
                read_seconds: 30,
            },
            compaction: CompactionConfig {
                temperature: 0.1,
                preferred_targets: vec!["claude-sonnet-4-6".into()],
            },
            retry: RetryConfig::default(),
        }
    }

    #[test]
    fn validates_capability_routing_config() {
        let mut cfg = base_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn rejects_account_model_outside_zai_catalog() {
        let mut cfg = base_config();
        cfg.accounts[0].models = Some(vec!["unknown-model".into()]);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("not present in zai.models"));
    }

    #[test]
    fn resolves_effective_reasoning_from_preset_override() {
        let cfg = base_config();
        let (_, targets) = cfg.route_targets_for_model("claude-sonnet-4-6").unwrap();
        let target = &targets[0];
        let reasoning = cfg
            .resolve_reasoning(target.reasoning.as_ref())
            .unwrap()
            .unwrap();
        assert_eq!(reasoning.budget, 16384);
        assert_eq!(reasoning.level, "MEDIUM");
        assert_eq!(reasoning.preset.as_deref(), Some("medium"));
    }

    #[test]
    fn has_direct_zai_config() {
        let cfg = base_config();
        assert_eq!(cfg.zai.api_url, "https://z.ai/chat");
    }

    #[test]
    fn allows_empty_compaction_targets() {
        let mut cfg = base_config();
        cfg.compaction.preferred_targets = Vec::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn serializes_routing_with_model_routes_only() {
        let cfg = base_config();
        let value = serde_json::to_value(&cfg.routing).unwrap();

        assert!(value.get("model_routes").is_some());
        assert!(value.get("model_provider_priority").is_none());
        assert!(value.get("model_overrides").is_none());
    }

    #[test]
    fn parses_route_step_shorthand_physical_and_logical() {
        let physical: ModelRouteStepWire =
            serde_json::from_value(serde_json::json!("glm-4.6")).unwrap();
        assert_eq!(
            physical.into_step().unwrap(),
            ModelRouteStepConfig::Physical {
                model: "glm-4.6".into(),
                reasoning: None,
            }
        );

        let logical: ModelRouteStepWire =
            serde_json::from_value(serde_json::json!("proxy:glm-5-turbo")).unwrap();
        assert_eq!(
            logical.into_step().unwrap(),
            ModelRouteStepConfig::Logical {
                model: "glm-5-turbo".into()
            }
        );
    }

    #[test]
    fn wire_serializes_shorthand_and_structured_steps() {
        let shorthand = ModelRouteStepWire::Shorthand("glm-4.6".into());
        assert_eq!(
            serde_json::to_value(shorthand).unwrap(),
            serde_json::json!("glm-4.6")
        );

        let structured = ModelRouteStepWire::Structured(ModelRouteStepConfig::Physical {
            model: "glm-4.6".into(),
            reasoning: Some(RouteReasoningConfig {
                effort: Some("medium".into()),
                budget: None,
                level: None,
            }),
        });
        let value = serde_json::to_value(structured).unwrap();
        assert_eq!(value["type"], "physical");
        assert!(value.get("provider").is_none());
        assert_eq!(value["model"], "glm-4.6");
        assert_eq!(value["reasoning"]["effort"], "medium");
    }

    #[test]
    fn parses_structured_physical_route_step_with_reasoning() {
        let step: ModelRouteStepWire = serde_json::from_value(serde_json::json!({
            "type": "physical",
            "model": "glm-4.6",
            "reasoning": { "effort": "medium" }
        }))
        .unwrap();

        assert_eq!(
            step.into_step().unwrap(),
            ModelRouteStepConfig::Physical {
                model: "glm-4.6".into(),
                reasoning: Some(RouteReasoningConfig {
                    effort: Some("medium".into()),
                    budget: None,
                    level: None,
                }),
            }
        );
    }

    #[test]
    fn persisted_config_uses_top_level_health() {
        let cfg = base_config();
        let value = serde_json::to_value(cfg.to_persisted()).unwrap();

        assert!(value.get("health").is_some());
        assert!(
            value
                .get("routing")
                .and_then(|routing| routing.get("health"))
                .is_none()
        );
    }

    #[test]
    fn persisted_config_accepts_top_level_health() {
        let persisted: PersistedConfig = serde_json::from_value(serde_json::json!({
            "server": {
                "host": "127.0.0.1",
                "port": 8765,
                "log_level": "INFO"
            },
            "zai": {
                "api_url": "https://z.ai/chat",
                "models": ["glm-4.6"]
            },
            "models": {
                "served": ["claude-sonnet-4-6"]
            },
            "routing": {
                "model_routes": {
                    "claude-sonnet-4-6": [
                        "glm-4.6"
                    ]
                },
            },
            "health": {
                "auth_failure_immediate_unhealthy": false,
                "failure_threshold": 7,
                "cooldown_seconds": 90
            },
            "accounts": [
                {
                    "id": "zai-a",
                    "enabled": true,
                    "weight": 1,
                    "models": ["glm-4.6"],
                    "auth": { "type": "api_key", "api_key": "sk-test" }
                }
            ],
            "access": {
                "require_key": false,
                "keys": []
            },
            "reasoning": {
                "effort_levels": {
                    "none": { "budget": 0, "level": "LOW" }
                },
                "default_effort": null
            },
            "timeouts": {
                "connect_seconds": 10,
                "read_seconds": 30
            },
            "compaction": {
                "temperature": 0.1,
                "preferred_targets": []
            }
        }))
        .unwrap();

        assert_eq!(persisted.health.failure_threshold, 7);
        assert!(!persisted.retry.enabled);
        assert_eq!(persisted.retry.max_attempts, 3);
        assert!(
            persisted
                .routing
                .model_routes
                .contains_key("claude-sonnet-4-6")
        );
    }

    #[test]
    fn parses_retry_config_from_persisted_config() {
        let mut value = serde_json::json!({
            "server": {
                "host": "127.0.0.1",
                "port": 8765,
                "log_level": "INFO"
            },
            "zai": {
                "api_url": "https://z.ai/chat",
                "models": ["glm-4.6"]
            },
            "models": {
                "served": ["claude-sonnet-4-6"]
            },
            "routing": {
                "model_routes": {
                    "claude-sonnet-4-6": [
                        "glm-4.6"
                    ]
                }
            },
            "health": {
                "auth_failure_immediate_unhealthy": true,
                "failure_threshold": 3,
                "cooldown_seconds": 30
            },
            "accounts": [],
            "access": {
                "require_key": false,
                "keys": []
            },
            "reasoning": {
                "effort_levels": {
                    "none": { "budget": 0, "level": "LOW" }
                },
                "default_effort": null
            },
            "timeouts": {
                "connect_seconds": 10,
                "read_seconds": 30
            },
            "compaction": {
                "temperature": 0.1,
                "preferred_targets": []
            }
        });
        value["retry"] = serde_json::json!({
            "enabled": true,
            "max_attempts": 5,
            "initial_delay_ms": 250,
            "max_delay_ms": 2000,
            "backoff_multiplier": 1.5
        });

        let persisted: PersistedConfig = serde_json::from_value(value).unwrap();

        assert!(persisted.retry.enabled);
        assert_eq!(persisted.retry.max_attempts, 5);
        assert_eq!(persisted.retry.initial_delay_ms, 250);
        assert_eq!(persisted.retry.max_delay_ms, 2000);
        assert_eq!(persisted.retry.backoff_multiplier, 1.5);
    }

    #[test]
    fn zodex_zai_only_config_validates_with_env_loaded_zai_account() {
        let path = std::env::temp_dir().join(format!(
            "zai-codex-proxy-zodex-config-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let config = serde_json::json!({
            "server": {
                "host": "127.0.0.1",
                "port": 18765,
                "log_level": "INFO"
            },
            "zai": {
                "api_url": "https://api.z.ai/api/coding/paas/v4/chat/completions",
                "models": ["glm-5.2", "glm-5-turbo"]
            },
            "models": {
                "served": ["glm-5.2", "glm-5-turbo", "compact-default"]
            },
            "routing": {
                "model_routes": {
                    "*": ["proxy:glm-5.2"],
                    "glm-5.2": [{
                        "type": "physical",
                        "model": "glm-5.2",
                        "reasoning": { "effort": "high" }
                    }],
                    "glm-5-turbo": [{
                        "type": "physical",
                        "model": "glm-5-turbo",
                        "reasoning": { "effort": "medium" }
                    }],
                    "compact-default": [{
                        "type": "physical",
                        "model": "glm-5-turbo",
                        "reasoning": { "effort": "none" }
                    }]
                }
            },
            "health": {
                "auth_failure_immediate_unhealthy": true,
                "failure_threshold": 3,
                "cooldown_seconds": 60
            },
            "access": {
                "require_key": false,
                "keys": []
            },
            "auto_compaction": {
                "enabled": true,
                "max_attempts_per_request": 1,
                "tail_items_to_keep": 8
            },
            "reasoning": {
                "default_effort": "high",
                "effort_levels": {
                    "none": { "budget": 0, "level": "LOW" },
                    "medium": { "budget": 16384, "level": "MEDIUM" },
                    "high": { "budget": 32768, "level": "HIGH" }
                }
            },
            "timeouts": {
                "connect_seconds": 10,
                "read_seconds": 600
            },
            "compaction": {
                "temperature": 0.1,
                "preferred_targets": ["compact-default"]
            },
            "retry": {
                "enabled": true,
                "max_attempts": 5,
                "initial_delay_ms": 1000,
                "max_delay_ms": 60000,
                "backoff_multiplier": 2.0
            }
        });
        std::fs::write(&path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();

        let mut cfg = Config::defaults();
        cfg.accounts = vec![AccountConfig {
            id: "zai-default".into(),
            enabled: true,
            weight: 1,
            models: None,
            auth: AccountAuth::ApiKey {
                api_key: "test-key".into(),
            },
        }];
        cfg.load_from_path(&path);
        let result = cfg.validate();
        let _ = std::fs::remove_file(&path);

        assert!(result.is_ok(), "{result:?}");
        assert!(cfg.retry.enabled);
        assert_eq!(cfg.retry.max_attempts, 5);
        assert_eq!(cfg.accounts.len(), 1);
    }

    #[test]
    fn rejects_legacy_nested_routing_health() {
        let err = serde_json::from_value::<PersistedConfig>(serde_json::json!({
            "server": {
                "host": "127.0.0.1",
                "port": 8765,
                "log_level": "INFO"
            },
            "zai": {
                "api_url": "https://z.ai/chat",
                "models": ["glm-4.6"]
            },
            "models": {
                "served": ["claude-sonnet-4-6"]
            },
            "routing": {
                "model_routes": {
                    "claude-sonnet-4-6": [
                        "glm-4.6"
                    ]
                },
                "health": {
                    "auth_failure_immediate_unhealthy": true,
                    "failure_threshold": 3,
                    "cooldown_seconds": 30
                }
            },
            "health": {
                "auth_failure_immediate_unhealthy": true,
                "failure_threshold": 3,
                "cooldown_seconds": 30
            },
            "accounts": [],
            "access": {
                "require_key": false,
                "keys": []
            },
            "reasoning": {
                "effort_levels": {
                    "none": { "budget": 0, "level": "LOW" }
                },
                "default_effort": null
            },
            "timeouts": {
                "connect_seconds": 10,
                "read_seconds": 30
            },
            "compaction": {
                "temperature": 0.1,
                "preferred_targets": []
            }
        }))
        .unwrap_err();

        assert!(err.to_string().contains("unknown field `health`"));
    }

    #[test]
    fn rejects_legacy_sticky_routing_config() {
        let err = serde_json::from_value::<PersistedConfig>(serde_json::json!({
            "server": {
                "host": "127.0.0.1",
                "port": 8765,
                "log_level": "INFO"
            },
            "zai": {
                "api_url": "https://z.ai/chat",
                "models": ["glm-4.6"]
            },
            "models": {
                "served": ["claude-sonnet-4-6"]
            },
            "routing": {
                "model_routes": {
                    "claude-sonnet-4-6": [
                        "glm-4.6"
                    ]
                },
                "sticky_routing": {
                    "enabled": true
                }
            },
            "health": {
                "auth_failure_immediate_unhealthy": true,
                "failure_threshold": 3,
                "cooldown_seconds": 30
            },
            "accounts": [],
            "access": {
                "require_key": false,
                "keys": []
            },
            "reasoning": {
                "effort_levels": {
                    "none": { "budget": 0, "level": "LOW" }
                },
                "default_effort": null
            },
            "timeouts": {
                "connect_seconds": 10,
                "read_seconds": 30
            },
            "compaction": {
                "temperature": 0.1,
                "preferred_targets": []
            }
        }))
        .unwrap_err();

        assert!(err.to_string().contains("unknown field `sticky_routing`"));
    }

    #[test]
    fn wildcard_route_can_be_logical_alias() {
        let mut cfg = base_config();
        cfg.routing.model_routes.insert(
            "*".into(),
            vec![ModelRouteStepConfig::Logical {
                model: "claude-sonnet-4-6".into(),
            }],
        );

        cfg.validate().unwrap();
        let (logical_model, targets) = cfg.route_targets_for_model("unmapped-model").unwrap();
        assert_eq!(logical_model, "claude-sonnet-4-6");
        assert_eq!(targets[0].model, "glm-4.6");
    }

    #[test]
    fn route_steps_support_physical_then_logical_fallback() {
        let mut cfg = base_config();
        cfg.routing.model_routes.insert(
            "glm-4.6-fast".into(),
            vec![
                ModelRouteStepConfig::Physical {
                    model: "glm-4.6-fast".into(),
                    reasoning: None,
                },
                ModelRouteStepConfig::Logical {
                    model: "claude-sonnet-4-6".into(),
                },
            ],
        );
        cfg.zai.models.push("glm-4.6-fast".into());
        cfg.accounts[0].models = None;

        cfg.validate().unwrap();
        let (logical_model, targets) = cfg.route_targets_for_model("glm-4.6-fast").unwrap();
        assert_eq!(logical_model, "glm-4.6-fast");
        assert_eq!(targets[0].model, "glm-4.6-fast");
        assert_eq!(targets[1].model, "glm-4.6");
    }

    #[test]
    fn rejects_missing_logical_reference() {
        let mut cfg = base_config();
        cfg.routing.model_routes.insert(
            "*".into(),
            vec![ModelRouteStepConfig::Logical {
                model: "missing-logical".into(),
            }],
        );

        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("references undefined logical model 'missing-logical'"));
    }

    #[test]
    fn rejects_logical_cycle() {
        let mut cfg = base_config();
        cfg.models.served = vec!["a".into()];
        cfg.compaction.preferred_targets = Vec::new();
        cfg.routing.model_routes = HashMap::from([
            (
                "a".into(),
                vec![ModelRouteStepConfig::Logical { model: "b".into() }],
            ),
            (
                "b".into(),
                vec![ModelRouteStepConfig::Logical { model: "a".into() }],
            ),
        ]);

        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("logical cycle"));
    }

    #[test]
    fn recovery_probe_target_ignores_routing_target() {
        let mut cfg = base_config();
        cfg.zai.models = vec!["account-model".into(), "catalog-model".into()];
        cfg.routing.model_routes.insert(
            "route-only-logical".into(),
            vec![ModelRouteStepConfig::Physical {
                model: "routed-model".into(),
                reasoning: None,
            }],
        );
        cfg.accounts.push(AccountConfig {
            id: "route-only-a".into(),
            enabled: true,
            weight: 1,
            models: Some(vec!["account-model".into()]),
            auth: AccountAuth::ApiKey {
                api_key: "sk-test".into(),
            },
        });

        let target = cfg.recovery_probe_target().unwrap();
        assert_eq!(target.model, "account-model");
    }

    #[test]
    fn recovery_probe_target_uses_account_model_when_no_routing_target_exists() {
        let mut cfg = base_config();
        cfg.zai.models = vec!["account-model".into(), "other-model".into()];
        cfg.accounts.clear();
        cfg.accounts.push(AccountConfig {
            id: "account-only-a".into(),
            enabled: true,
            weight: 1,
            models: Some(vec!["account-model".into()]),
            auth: AccountAuth::ApiKey {
                api_key: "sk-test".into(),
            },
        });

        let target = cfg.recovery_probe_target().unwrap();
        assert_eq!(target.model, "account-model");
    }

    #[test]
    fn recovery_probe_target_uses_zai_catalog_when_account_models_missing() {
        let mut cfg = base_config();
        cfg.zai.models = vec!["catalog-model".into()];
        cfg.accounts.clear();
        cfg.accounts.push(AccountConfig {
            id: "no-probe-a".into(),
            enabled: true,
            weight: 1,
            models: None,
            auth: AccountAuth::ApiKey {
                api_key: "sk-test".into(),
            },
        });

        let target = cfg.recovery_probe_target().unwrap();
        assert_eq!(target.model, "catalog-model");
        cfg.accounts.clear();
        assert!(cfg.recovery_probe_target().is_none());
    }

    #[test]
    fn preferred_targets_rejects_object_items() {
        let err = serde_json::from_value::<CompactionConfig>(serde_json::json!({
            "temperature": 0.1,
            "preferred_targets": [
              "glm-5-turbo",
              { "model": "claude-sonnet-4-6" }
            ]
        }))
        .unwrap_err()
        .to_string();

        assert!(err.contains("expected a string"));
    }
}
