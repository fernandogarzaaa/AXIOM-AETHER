use futures::future::join_all;
use reqwest::Client;

use candle_core::Result;

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct StateDeltaUpdate {
    pub session_id: String,
    pub layer_index: usize,
    pub delta_bytes: Vec<u8>,
}

#[derive(Clone, Default)]
pub struct ClusterStateManager {
    client: Client,
}

impl ClusterStateManager {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    pub async fn broadcast(&self, payload: StateDeltaUpdate, peers: Vec<String>) -> Result<()> {
        let requests = peers.into_iter().map(|peer| {
            let client = self.client.clone();
            let payload = payload.clone();
            async move {
                let url = format!("{}/v1/cluster/sync", peer.trim_end_matches('/'));
                let response = client
                    .post(url)
                    .json(&payload)
                    .send()
                    .await
                    .map_err(|err| {
                        candle_core::Error::Msg(format!("cluster sync request failed: {err}"))
                    })?;
                if !response.status().is_success() {
                    let status = response.status();
                    let body = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "unable to read peer error body".to_string());
                    return Err(candle_core::Error::Msg(format!(
                        "cluster sync peer returned {status}: {body}"
                    )));
                }
                Ok(())
            }
        });

        for outcome in join_all(requests).await {
            outcome?;
        }
        Ok(())
    }
}

pub async fn broadcast_delta(payload: StateDeltaUpdate, peers: Vec<String>) -> Result<()> {
    ClusterStateManager::new().broadcast(payload, peers).await
}
