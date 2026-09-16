mod diagnostics;
mod errors;
mod execution;
#[cfg(feature = "trace-parity")]
mod function_trace;
mod marshal;
mod routes;

use litellm_ai_gateway::io::responses_ws::ResponsesWebSocketConnection as RustResponsesWebSocketConnection;
use litellm_core::http_transport::{
    HttpTransportRequest, HttpTransportResponse as RustHttpTransportResponse,
    send as send_http_request,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
use serde_json::Value;

use crate::errors::core_error_to_pyerr;
use crate::marshal::{marshal_headers, optional_timeout};

#[pyclass]
struct HttpResponseConnection {
    inner: RustHttpTransportResponse,
}

#[pymethods]
impl HttpResponseConnection {
    #[classmethod]
    #[pyo3(signature = (method, url, headers=None, body=Vec::new(), read_timeout_seconds=None))]
    fn request<'py>(
        _cls: &Bound<'py, pyo3::types::PyType>,
        py: Python<'py>,
        method: String,
        url: String,
        #[pyo3(from_py_with = litellm_python_interop::from_py)] headers: Option<Value>,
        body: Vec<u8>,
        read_timeout_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = HttpTransportRequest::from_parts(
            &method,
            &url,
            marshal_headers(headers)?,
            body,
            optional_timeout(read_timeout_seconds),
        )
        .map_err(core_error_to_pyerr)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let inner = send_http_request(request)
                .await
                .map_err(core_error_to_pyerr)?;
            Ok(HttpResponseConnection { inner })
        })
    }

    #[getter]
    fn status_code(&self) -> u16 {
        self.inner.status
    }

    fn headers(&self, py: Python<'_>) -> Vec<(String, Py<PyBytes>)> {
        self.inner
            .headers
            .iter()
            .map(|(name, value)| (name.clone(), PyBytes::new(py, value).unbind()))
            .collect()
    }

    fn next_chunk<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chunk = inner.next_chunk().await.map_err(core_error_to_pyerr)?;
            Python::attach(|py| Ok(chunk.map(|bytes| PyBytes::new(py, &bytes).unbind())))
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.close().await;
            Ok(())
        })
    }
}

#[pyclass]
struct ResponsesWebSocketConnection {
    inner: RustResponsesWebSocketConnection,
}

#[pymethods]
impl ResponsesWebSocketConnection {
    #[classmethod]
    #[pyo3(signature = (url, headers=None, timeout_seconds=None))]
    fn connect<'py>(
        _cls: &Bound<'py, pyo3::types::PyType>,
        py: Python<'py>,
        url: String,
        #[pyo3(from_py_with = litellm_python_interop::from_py)] headers: Option<Value>,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let headers = marshal_headers(headers)?;
        let timeout = optional_timeout(timeout_seconds);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let inner = RustResponsesWebSocketConnection::connect_url(&url, &headers, timeout)
                .await
                .map_err(core_error_to_pyerr)?;
            Ok(ResponsesWebSocketConnection { inner })
        })
    }

    fn send_text<'py>(&self, py: Python<'py>, text: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.send_text(text).await.map_err(core_error_to_pyerr)
        })
    }

    fn recv_text<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.recv_text().await.map_err(core_error_to_pyerr)
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.close().await.map_err(core_error_to_pyerr)
        })
    }
}

#[pymodule(gil_used = false)]
mod _native {
    use pyo3::prelude::*;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        super::errors::register(module)?;
        super::routes::register(module)?;
        module.add_class::<super::HttpResponseConnection>()?;
        module.add_class::<super::ResponsesWebSocketConnection>()?;
        super::diagnostics::register(module)
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::time::Duration;

    use futures_util::{SinkExt, StreamExt};
    use pyo3::types::PyDict;
    use tokio::net::TcpListener;
    use tokio_tungstenite::{accept_async, tungstenite::Message};

    use super::*;

    #[test]
    fn module_registration_preserves_the_public_surface() {
        Python::initialize();
        Python::attach(|py| {
            let module = pyo3::wrap_pymodule!(_native)(py).into_bound(py);

            let expected = [
                "RustBridgeDeclined",
                "RustUpstreamError",
                "ocr",
                "aocr",
                "transcription",
                "atranscription",
                "messages",
                "amessages",
                "chat_completions_decline",
                "chat_completions",
                "achat_completions",
                "HttpResponseConnection",
                "ResponsesWebSocketConnection",
                "gil_stats",
            ];

            let public_names: Vec<String> = module
                .dict()
                .keys()
                .extract::<Vec<String>>()
                .expect("module names should be strings")
                .into_iter()
                .filter(|name| !name.starts_with('_'))
                .collect();
            assert_eq!(public_names, expected);

            #[cfg(not(feature = "trace-parity"))]
            assert!(!module.hasattr("_trace").expect("module lookup should work"));

            #[cfg(feature = "trace-parity")]
            {
                let trace = module
                    .getattr("_trace")
                    .expect("trace build should expose its diagnostic namespace");
                let trace_names: Vec<String> = trace
                    .cast::<PyModule>()
                    .expect("trace namespace should be a module")
                    .dict()
                    .keys()
                    .extract::<Vec<String>>()
                    .expect("trace names should be strings")
                    .into_iter()
                    .filter(|name| !name.starts_with("__"))
                    .collect();
                assert_eq!(
                    trace_names,
                    [
                        "ocr",
                        "aocr",
                        "transcription",
                        "atranscription",
                        "messages",
                        "amessages",
                        "chat_completions",
                        "achat_completions",
                        "gateway_messages",
                    ]
                );
            }
        });
    }

    #[test]
    fn responses_websocket_connection_round_trips_through_python() {
        Python::initialize();
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        let listener = runtime
            .block_on(TcpListener::bind("127.0.0.1:0"))
            .expect("listener should bind");
        let address = listener
            .local_addr()
            .expect("listener should have an address");
        let server = runtime.spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("handshake should succeed");

            let message = socket
                .next()
                .await
                .expect("client should send a frame")
                .expect("client frame should be valid");
            assert_eq!(message, Message::Text("from-python".into()));
            socket
                .send(Message::Text("from-server".into()))
                .await
                .expect("server should reply");
            assert!(matches!(socket.next().await, Some(Ok(Message::Close(_)))));
        });

        Python::attach(|py| {
            let module = pyo3::wrap_pymodule!(_native)(py).into_bound(py);
            let locals = PyDict::new(py);
            locals
                .set_item("native", &module)
                .expect("module should enter Python locals");
            locals
                .set_item("url", format!("ws://{address}"))
                .expect("URL should enter Python locals");
            let code = CString::new(
                r#"
import asyncio

async def exercise():
    connection = await native.ResponsesWebSocketConnection.connect(url)
    assert type(connection) is native.ResponsesWebSocketConnection
    await connection.send_text("from-python")
    assert await connection.recv_text() == "from-server"
    await connection.close()
    assert await connection.recv_text() is None

asyncio.run(asyncio.wait_for(exercise(), timeout=5))
"#,
            )
            .expect("Python source should not contain null bytes");
            py.run(&code, Some(&locals), Some(&locals))
                .expect("Python WebSocket methods should round trip");
        });

        runtime
            .block_on(async { tokio::time::timeout(Duration::from_secs(5), server).await })
            .expect("server should finish")
            .expect("server task should not panic");
    }
}
