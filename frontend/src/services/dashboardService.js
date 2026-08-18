/**
 * Dashboard Service
 *
 * Interacts with GET /api/dashboard endpoint to fetch system-wide metrics
 * and recent transaction analysis records.
 */

import { apiRequest } from "./api";

export const dashboardService = {
  /**
   * Fetch dashboard metrics and recent analyses.
   * GET /api/dashboard?recent_limit={recentLimit}
   */
  async getDashboardData(recentLimit = 10) {
    return apiRequest(`/api/dashboard?recent_limit=${recentLimit}`);
  },
};
