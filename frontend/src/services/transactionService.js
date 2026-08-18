/**
 * Transaction Service
 *
 * Handles interaction with backend transaction analysis and history endpoints.
 * strictly adhering to IEEE-CIS CatBoost 20-feature schema.
 */

import { apiRequest } from "./api";

/**
 * Format raw transaction form values to conform to backend schema requirements.
 * Ensures required fields (card1, transaction_amt) are numbers,
 * and optional empty/blank values become null instead of empty strings.
 */
export function sanitizeTransactionPayload(rawForm) {
  const parseNum = (val) => {
    if (val === null || val === undefined || val === "") return null;
    const num = Number(val);
    return isNaN(num) ? null : num;
  };

  const parseStr = (val) => {
    if (val === null || val === undefined) return null;
    const trimmed = String(val).trim();
    return trimmed === "" ? null : trimmed;
  };

  return {
    card1: Number(rawForm.card1),
    transaction_amt: Number(rawForm.transaction_amt),

    card2: parseNum(rawForm.card2),
    addr1: parseNum(rawForm.addr1),
    card5: parseNum(rawForm.card5),

    C1: parseNum(rawForm.C1),
    C2: parseNum(rawForm.C2),
    C5: parseNum(rawForm.C5),
    C6: parseNum(rawForm.C6),
    C13: parseNum(rawForm.C13),
    C14: parseNum(rawForm.C14),

    D1: parseNum(rawForm.D1),
    D2: parseNum(rawForm.D2),
    D10: parseNum(rawForm.D10),
    D15: parseNum(rawForm.D15),

    M4_enc: parseNum(rawForm.M4_enc),
    M5_enc: parseNum(rawForm.M5_enc),
    M6_enc: parseNum(rawForm.M6_enc),

    purchaser_email_domain: parseStr(rawForm.purchaser_email_domain),
    log_amt: parseNum(rawForm.log_amt),
  };
}

export const transactionService = {
  /**
   * Submit transaction for fraud analysis.
   * POST /api/transactions/analyze
   */
  async analyzeTransaction(rawForm) {
    const payload = sanitizeTransactionPayload(rawForm);
    return apiRequest("/api/transactions/analyze", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  /**
   * Fetch recent transaction history.
   * GET /api/transactions/history?limit={limit}
   */
  async getTransactionHistory(limit = 50) {
    return apiRequest(`/api/transactions/history?limit=${limit}`);
  },

  /**
   * Fetch a single transaction analysis record by ID.
   * GET /api/transactions/{analysis_id}
   */
  async getTransactionById(analysisId) {
    return apiRequest(`/api/transactions/${encodeURIComponent(analysisId)}`);
  },
};
