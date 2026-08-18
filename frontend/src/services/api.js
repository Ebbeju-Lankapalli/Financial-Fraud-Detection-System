/**
 * Base API Service Layer
 *
 * Handles base URL configuration, request formatting, JSON parsing,
 * backend detail error extraction, and network error handling.
 */

const getBaseUrl = () => {
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  const baseUrl = envUrl && envUrl.trim() !== "" ? envUrl : "http://localhost:8000";
  return baseUrl.replace(/\/+$/, "");
};

export const API_BASE_URL = getBaseUrl();

/**
 * Execute an HTTP request against the API backend.
 *
 * @param {string} endpoint - Relative path (e.g. '/api/transactions/analyze')
 * @param {RequestInit} [options={}] - Fetch options
 * @returns {Promise<any>}
 */
export async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;

  const headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    ...(options.headers || {}),
  };

  try {
    const response = await fetch(url, {
      ...options,
      headers,
    });

    // Handle 204 No Content
    if (response.status === 204) {
      return null;
    }

    const contentType = response.headers.get("content-type");
    let data = null;

    if (contentType && contentType.includes("application/json")) {
      data = await response.json();
    } else {
      const text = await response.text();
      data = text ? { detail: text } : null;
    }

    if (!response.ok) {
      let errorMessage = `HTTP Error ${response.status} (${response.statusText})`;

      if (data) {
        if (typeof data.detail === "string") {
          errorMessage = data.detail;
        } else if (Array.isArray(data.detail)) {
          // FastAPI validation errors format
          errorMessage = data.detail.map((err) => `${err.loc ? err.loc.join('.') + ': ' : ''}${err.msg}`).join("; ");
        } else if (data.message) {
          errorMessage = data.message;
        }
      }

      const error = new Error(errorMessage);
      error.status = response.status;
      error.data = data;
      throw error;
    }

    return data;
  } catch (error) {
    if (error.status) {
      throw error;
    }
    // Network or parse error
    const networkError = new Error(
      `Unable to connect to Fraud Detection API at ${API_BASE_URL}. Please verify the backend service is running.`
    );
    networkError.originalError = error;
    throw networkError;
  }
}
