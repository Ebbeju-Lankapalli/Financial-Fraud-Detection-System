/**
 * Model Evaluation Service
 *
 * Interacts with /api/evaluation routes for active CatBoost production metrics,
 * threshold selection summaries, full reference comparisons, and QLoRA research metrics.
 */

import { apiRequest } from "./api";

export const evaluationService = {
  /**
   * Fetch production-first evaluation summary.
   * GET /api/evaluation
   */
  async getSummary() {
    return apiRequest("/api/evaluation");
  },

  /**
   * Fetch active CatBoost production model metrics (test split).
   * GET /api/evaluation/production
   */
  async getProductionEvaluation() {
    return apiRequest("/api/evaluation/production");
  },

  /**
   * Fetch production model validation metrics.
   * GET /api/evaluation/production/validation
   */
  async getProductionValidation() {
    return apiRequest("/api/evaluation/production/validation");
  },

  /**
   * Fetch validation-only threshold selection data.
   * GET /api/evaluation/production/threshold
   */
  async getProductionThreshold() {
    return apiRequest("/api/evaluation/production/threshold");
  },

  /**
   * Fetch 61-feature CatBoost full reference model evaluation.
   * GET /api/evaluation/production/full-reference
   */
  async getFullReferenceEvaluation() {
    return apiRequest("/api/evaluation/production/full-reference");
  },

  /**
   * Fetch preserved QLoRA research experiment evaluation.
   * GET /api/evaluation/research
   */
  async getResearchEvaluation() {
    return apiRequest("/api/evaluation/research");
  },
};
