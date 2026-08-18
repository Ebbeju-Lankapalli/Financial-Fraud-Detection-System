import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  History as HistoryIcon,
  Search,
  Filter,
  Eye,
  RefreshCw,
  SlidersHorizontal
} from "lucide-react";

import { transactionService } from "../services/transactionService";
import RiskBadge from "../components/common/RiskBadge";
import LoadingSpinner from "../components/common/LoadingSpinner";
import ErrorMessage from "../components/common/ErrorMessage";
import EmptyState from "../components/common/EmptyState";

export default function History() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filters & Search State
  const [limit, setLimit] = useState(50);
  const [riskFilter, setRiskFilter] = useState("ALL");
  const [searchQuery, setSearchQuery] = useState("");

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await transactionService.getTransactionHistory(limit);
      setRecords(data || []);
    } catch (err) {
      setError(err.message || "Failed to load transaction history.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [limit]);

  // Client-side filtering
  const filteredRecords = records.filter((item) => {
    // Risk Filter
    if (riskFilter !== "ALL" && item.risk !== riskFilter) {
      return false;
    }

    // Search Query
    if (searchQuery.trim() !== "") {
      const query = searchQuery.toLowerCase();
      const matchId = item.analysis_id?.toLowerCase().includes(query);
      const matchCard = String(item.card1).includes(query);
      const matchAmt = String(item.transaction_amt).includes(query);
      return matchId || matchCard || matchAmt;
    }

    return true;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Transaction History</h1>
          <p className="text-sm text-gray-400">Persistent audit records of all evaluated transactions</p>
        </div>

        <button
          onClick={fetchHistory}
          disabled={loading}
          className="btn-secondary text-xs px-3 py-2"
        >
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Control Bar: Search, Risk Filter, Limit Selector */}
      <div className="card-glass p-4 flex flex-col md:flex-row gap-4 items-stretch md:items-center justify-between">
        {/* Search Input */}
        <div className="relative flex-1">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by Analysis ID, Card1, or Amount..."
            className="form-input pl-9 text-xs w-full"
          />
        </div>

        {/* Risk Filter Buttons */}
        <div className="flex items-center space-x-2 text-xs">
          <span className="text-gray-400 font-medium flex items-center space-x-1 mr-1">
            <Filter size={14} />
            <span>Risk:</span>
          </span>
          {["ALL", "HIGH", "LOW"].map((risk) => (
            <button
              key={risk}
              onClick={() => setRiskFilter(risk)}
              className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                riskFilter === risk
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                  : "bg-gray-800/60 text-gray-400 hover:text-white"
              }`}
            >
              {risk === "ALL" ? "All" : risk === "HIGH" ? "High Risk" : "Low Risk"}
            </button>
          ))}
        </div>

        {/* Limit Selector */}
        <div className="flex items-center space-x-2 text-xs font-mono">
          <span className="text-gray-400">Limit:</span>
          <select
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="form-input text-xs py-1 px-2 font-mono"
          >
            <option value={20}>20</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
            <option value={200}>200</option>
          </select>
        </div>
      </div>

      {/* Main Table Content */}
      {loading && <LoadingSpinner message="Fetching audit records..." />}
      {error && <ErrorMessage title="History Error" message={error} onRetry={fetchHistory} />}

      {!loading && !error && (
        <>
          {filteredRecords.length > 0 ? (
            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Created At</th>
                    <th>Analysis ID</th>
                    <th>Card1</th>
                    <th>Amount ($)</th>
                    <th>Risk</th>
                    <th>Fraud Probability</th>
                    <th>Decision Source</th>
                    <th className="text-right">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRecords.map((item) => {
                    const dateStr = item.created_at
                      ? new Date(item.created_at).toLocaleString()
                      : "N/A";
                    const probPct = (item.fraud_probability * 100).toFixed(2);

                    return (
                      <tr key={item.analysis_id}>
                        <td className="font-mono text-xs text-gray-400">{dateStr}</td>
                        <td className="font-mono text-xs text-cyan-400">{item.analysis_id}</td>
                        <td className="font-mono">{item.card1}</td>
                        <td className="font-mono font-semibold">${item.transaction_amt?.toFixed(2)}</td>
                        <td>
                          <RiskBadge risk={item.risk} size="sm" />
                        </td>
                        <td className="font-mono font-bold">
                          <span className={item.risk === "HIGH" ? "text-rose-400" : "text-emerald-400"}>
                            {probPct}%
                          </span>
                        </td>
                        <td className="font-mono text-xs text-gray-400">{item.decision_source}</td>
                        <td className="text-right">
                          <Link
                            to={`/history/${item.analysis_id}`}
                            className="btn-secondary text-xs px-2.5 py-1 inline-flex items-center space-x-1"
                          >
                            <Eye size={14} />
                            <span>View</span>
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState
              title="No Matching Audit Records"
              message={
                searchQuery || riskFilter !== "ALL"
                  ? "No transactions match your current search and filter settings."
                  : "No transaction analyses stored in the audit repository."
              }
              action={
                <Link to="/analyze" className="btn-primary text-xs px-4 py-2">
                  Analyze First Transaction
                </Link>
              }
            />
          )}
        </>
      )}
    </div>
  );
}
