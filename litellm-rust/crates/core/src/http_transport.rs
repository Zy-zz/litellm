use std::sync::{Arc, OnceLock};
use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::{Method, Response, Url};
use tokio::sync::Mutex;

use crate::Error;
use crate::constants::{HTTP_TRANSPORT_CONNECT_TIMEOUT_SECS, HTTP_TRANSPORT_READ_TIMEOUT_SECS};

pub struct HttpTransportRequest {
    pub method: Method,
    pub url: Url,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
    pub read_timeout: Duration,
}

impl HttpTransportRequest {
    pub fn new(method: Method, url: Url, headers: HeaderMap, body: Vec<u8>) -> Self {
        Self {
            method,
            url,
            headers,
            body,
            read_timeout: Duration::from_secs(HTTP_TRANSPORT_READ_TIMEOUT_SECS),
        }
    }

    pub fn from_parts(
        method: &str,
        url: &str,
        headers: impl IntoIterator<Item = (String, String)>,
        body: Vec<u8>,
        read_timeout: Option<Duration>,
    ) -> Result<Self, Error> {
        let method = method
            .parse::<Method>()
            .map_err(|_| Error::InvalidRequest("invalid HTTP method".to_string()))?;
        let url =
            Url::parse(url).map_err(|_| Error::InvalidRequest("invalid HTTP URL".to_string()))?;
        let headers = headers
            .into_iter()
            .map(|(name, value)| {
                let name = HeaderName::from_bytes(name.as_bytes())
                    .map_err(|_| Error::InvalidRequest("invalid HTTP header name".to_string()))?;
                let value = HeaderValue::from_str(&value)
                    .map_err(|_| Error::InvalidRequest("invalid HTTP header value".to_string()))?;
                Ok((name, value))
            })
            .collect::<Result<HeaderMap, Error>>()?;
        let mut request = Self::new(method, url, headers, body);
        request.read_timeout = read_timeout.unwrap_or(request.read_timeout);
        Ok(request)
    }
}

#[derive(Clone)]
pub struct HttpTransportResponse {
    pub status: u16,
    pub headers: Vec<(String, Vec<u8>)>,
    response: Arc<Mutex<Option<Response>>>,
    read_timeout: Duration,
}

impl HttpTransportResponse {
    pub async fn next_chunk(&self) -> Result<Option<Vec<u8>>, Error> {
        let mut response = self.response.lock().await;
        let Some(upstream) = response.as_mut() else {
            return Ok(None);
        };
        let chunk = tokio::time::timeout(self.read_timeout, upstream.chunk())
            .await
            .map_err(|_| Error::Network("upstream response chunk timed out".to_string()))?
            .map_err(|error| Error::Network(error.to_string()))?;
        if chunk.is_none() {
            *response = None;
        }
        Ok(chunk.map(|bytes| bytes.to_vec()))
    }

    pub async fn close(&self) {
        *self.response.lock().await = None;
    }
}

fn client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(HTTP_TRANSPORT_CONNECT_TIMEOUT_SECS))
            .build()
            .expect("valid HTTP client configuration")
    })
}

pub async fn send(request: HttpTransportRequest) -> Result<HttpTransportResponse, Error> {
    let response = tokio::time::timeout(
        request.read_timeout,
        client()
            .request(request.method, request.url)
            .headers(request.headers)
            .body(request.body)
            .send(),
    )
    .await
    .map_err(|_| Error::Network("upstream response headers timed out".to_string()))?
    .map_err(|error| {
        if error.is_connect() || error.is_builder() {
            Error::Connect(error.to_string())
        } else {
            Error::Network(error.to_string())
        }
    })?;
    let status = response.status().as_u16();
    let headers = response
        .headers()
        .iter()
        .map(|(name, value)| (name.as_str().to_string(), value.as_bytes().to_vec()))
        .collect();
    Ok(HttpTransportResponse {
        status,
        headers,
        response: Arc::new(Mutex::new(Some(response))),
        read_timeout: request.read_timeout,
    })
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use axum::Router;
    use axum::body::Body;
    use axum::http::Response as AxumResponse;
    use axum::routing::post;
    use tokio::net::TcpListener;

    use super::*;

    #[tokio::test]
    async fn streams_every_upstream_chunk() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/stream",
            post(|| async {
                AxumResponse::builder()
                    .status(200)
                    .header("x-test", "yes")
                    .body(Body::from_stream(futures_util::stream::iter([
                        Ok::<_, Infallible>("data: one\n\n"),
                        Ok::<_, Infallible>("data: two\n\n"),
                    ])))
                    .unwrap()
            }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let request = HttpTransportRequest::new(
            Method::POST,
            Url::parse(&format!("http://{address}/stream")).unwrap(),
            HeaderMap::new(),
            Vec::new(),
        );

        let response = send(request).await.unwrap();
        assert_eq!(response.status, 200);
        assert!(
            response
                .headers
                .iter()
                .any(|(name, value)| name == "x-test" && value == b"yes")
        );
        let mut body = Vec::new();
        while let Some(chunk) = response.next_chunk().await.unwrap() {
            body.extend(chunk);
        }
        assert_eq!(body, b"data: one\n\ndata: two\n\n");
        server.abort();
    }
}
