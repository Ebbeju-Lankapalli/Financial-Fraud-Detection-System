import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  ShieldAlert,
  ShieldCheck,
  Percent,
  CheckCircle2,
  XCircle,
  Eye,
  BarChart2,
  PieChart as PieChartIcon
} from "lucide-react";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid
} from "recharts";

import { dashboardService } from "../services/dashboardService";
import MetricCard from "../components/common/MetricCard";
import RiskBadge from "../components/common/RiskBadge";
import LoadingSpinner from "../components/common/LoadingSpinner";
import ErrorMessage from "../components/common/ErrorMessage";

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchDashboard = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await dashboardService.getDashboardData(10);
      setData(res);
    } catch (err) {
      setError(err.message || "Failed to load dashboard data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboard();
  }, []);

  if (loading) return <LoadingSpinner message="Fetching dashboard metrics..." />;
  if (error) return <ErrorMessage title="Dashboard Error" message={error} onRetry={fetchDashboard} />;
  if (!data) return null;

  const { metrics, recent_analyses = [] } = data;

  // Chart data built strictly from real API metrics
  const riskPieData = [
    { name: "Low Risk", value: metrics.low_risk_count, color: "#10b981" },
    { name: "High Risk", value: metrics.high_risk_count, color: "#f43f5e" },
  ];

  // Bar chart data from recent analyses
  const recentBarData = recent_analyses.slice().reverse().map((item, idx) => ({
    name: `#${idx + 1}`,
    id: item.analysis_id?.slice(0, 6) || `#${idx + 1}`,
    amt: item.transaction_amt,
    probability: Number((item.fraud_probability * 100).toFixed(2)),
    risk: item.risk,
  }));

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">System Dashboard</h1>
          <p className="text-sm text-gray-400">Live metrics and persistent audit telemetry for CatBoost fraud analyses</p>
        </div>

        <Link to="/analyze" className="btn-primary text-xs px-4 py-2.5">
          <ShieldCheck size={16} />
          <span>New Analysis</span>
        </Link>
      </div>

      {/* 6 Key Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <MetricCard
          title="Total Analyses"
          value={metrics.total_analyses.toLocaleString()}
          subtext="Total transaction records"
          icon={Activity}
          color="cyan"
        />

        <MetricCard
          title="High Risk Count"
          value={metrics.high_risk_count.toLocaleString()}
          subtext="Probabilities ≥ 0.83 threshold"
          icon={ShieldAlert}
          color="rose"
        />

        <MetricCard
          title="Low Risk Count"
          value={metrics.low_risk_count.toLocaleString()}
          subtext="Probabilities < 0.83 threshold"
          icon={ShieldCheck}
          color="emerald"
        />

        <MetricCard
          title="High Risk Rate"
          value={`${metrics.high_risk_percentage}%`}
          subtext="Ratio of high-risk evaluations"
          icon={Percent}
          color="amber"
        />

        <MetricCard
          title="Valid Outputs"
          value={metrics.valid_output_count.toLocaleString()}
          subtext="Schema & prediction validated"
          icon={CheckCircle2}
          color="cyan"
        />

        <MetricCard
          title="Invalid Outputs"
          value={metrics.invalid_output_count.toLocaleString()}
          subtext="Failed or anomalous evaluations"
          icon={XCircle}
          color="purple"
        />
      </div>

      {/* Charts Section (Built strictly from real API metrics & recent analyses) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Risk Distribution Donut Chart */}
        <div className="card-glass p-5 space-y-4">
          <div className="flex items-center space-x-2 border-b border-gray-800 pb-3">
            <PieChartIcon size={18} className="text-cyan-400" />
            <h3 className="font-semibold text-white text-sm">Risk Distribution</h3>
          </div>

          {metrics.total_analyses > 0 ? (
            <div className="h-64 w-full flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={riskPieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={85}
                    paddingAngle={4}
                    dataKey="value"
                  >
                    {riskPieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: "#111827", borderColor: "#1f293d", borderRadius: "8px", color: "#fff" }}
                    itemStyle={{ color: "#fff", fontSize: "12px" }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-xs text-gray-500 font-mono">
              No transactions recorded yet
            </div>
          )}

          <div className="flex justify-center space-x-6 text-xs font-medium">
            <div className="flex items-center space-x-2">
              <span className="w-3 h-3 rounded-full bg-emerald-500" />
              <span className="text-gray-300">Low Risk ({metrics.low_risk_count})</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="w-3 h-3 rounded-full bg-rose-500" />
              <span className="text-gray-300">High Risk ({metrics.high_risk_count})</span>
            </div>
          </div>
        </div>

        {/* Recent Fraud Probabilities Bar Chart */}
        <div className="card-glass p-5 lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between border-b border-gray-800 pb-3">
            <div className="flex items-center space-x-2">
              <BarChart2 size={18} className="text-cyan-400" />
              <h3 className="font-semibold text-white text-sm">Recent Fraud Probabilities (%)</h3>
            </div>
            <span className="text-xs text-gray-400 font-mono">Threshold: 83.00%</span>
          </div>

          {recentBarData.length > 0 ? (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={recentBarData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" />
                  <XAxis dataKey="id" stroke="#6b7280" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 100]} stroke="#6b7280" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#111827", borderColor: "#1f293d", borderRadius: "8px", color: "#fff" }}
                    formatter={(val) => [`${val}%`, "Fraud Probability"]}
                  />
                  <Bar dataKey="probability" radius={[4, 4, 0, 0]}>
                    {recentBarData.map((entry, index) => (
                      <Cell key={`bar-${index}`} fill={entry.risk === "HIGH" ? "#f43f5e" : "#06b6d4"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-xs text-gray-500 font-mono">
              No transactions recorded yet
            </div>
          )}
        </div>
      </div>

      {/* Recent Analyses Data Table */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-bold text-white text-lg tracking-tight">Recent Analyses</h3>
          <Link to="/history" className="text-xs text-cyan-400 hover:text-cyan-300 font-medium">
            View full history →
          </Link>
        </div>

        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Created At</th>
                <th>Card1</th>
                <th>Amount ($)</th>
                <th>Risk</th>
                <th>Fraud Probability</th>
                <th>Model</th>
                <th className="text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {recent_analyses.length > 0 ? (
                recent_analyses.map((item) => {
                  const dateStr = item.created_at
                    ? new Date(item.created_at).toLocaleString()
                    : "N/A";
                  const probPct = (item.fraud_probability * 100).toFixed(2);

                  return (
                    <tr key={item.analysis_id}>
                      <td className="font-mono text-xs text-gray-400">{dateStr}</td>
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
                      <td className="font-mono text-xs text-gray-400">{item.model}</td>
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
                })
              ) : (
                <tr>
                  <td colSpan={7} className="text-center py-8 text-gray-500 font-mono text-xs">
                    No recent analyses found. Submit a transaction to populate dashboard.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
