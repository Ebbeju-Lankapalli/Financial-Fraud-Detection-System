/**
 * Fraud Investigation Agent Service
 *
 * Interacts with /api/agent and /api/agent/analyze endpoints.
 */

import { apiRequest } from "./api";
import { sanitizeTransactionPayload } from "./transactionService";

export const agentService = {
  /**
   * Fetch agent runtime status (Groq configuration state, model name).
   * GET /api/agent
   */
  async getAgentStatus() {
    return apiRequest("/api/agent");
  },

  /**
   * Submit transaction and optional investigator question for AI investigation.
   * POST /api/agent/analyze
   */
  async analyzeWithAgent(rawTransactionForm, question = null) {
    const sanitizedTransaction = sanitizeTransactionPayload(rawTransactionForm);
    const payload = {
      transaction: sanitizedTransaction,
      question: question && String(question).trim() !== "" ? String(question).trim() : null,
    };

    return apiRequest("/api/agent/analyze", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
};
