import React from "react";

/**
 * ProbabilityBar Component
 *
 * Renders the fraud probability percentage with threshold indicator.
 * Strictly formats decimals (e.g. 0.5801 -> 58.01%).
 */
export default function ProbabilityBar({ probability, threshold = 0.83, risk = "LOW" }) {
  const probNum = typeof probability === "number" ? probability : 0;
  const threshNum = typeof threshold === "number" ? threshold : 0.83;

  const pct = (probNum * 100).toFixed(2);
  const threshPct = (threshNum * 100).toFixed(2);

  const isHigh = risk === "HIGH" || probNum >= threshNum;

  const barColor = isHigh ? "bg-gradient-to-r from-amber-500 to-rose-600" : "bg-gradient-to-r from-emerald-500 to-cyan-500";
  const textColor = isHigh ? "text-rose-400" : "text-emerald-400";

  return (
    <div className="w-full space-y-2">
      <div className="flex justify-between items-end text-sm">
        <span className="text-gray-400 font-medium">Fraud Probability</span>
        <div className="text-right">
          <span className={`text-xl font-bold font-mono ${textColor}`}>{pct}%</span>
          <span className="text-xs text-gray-500 ml-2">(Threshold: {threshPct}%)</span>
        </div>
      </div>

      {/* Progress Track */}
      <div className="relative w-full h-3 bg-gray-900 rounded-full overflow-hidden border border-gray-800">
        {/* Filled bar */}
        <div
          className={`h-full ${barColor} transition-all duration-500 ease-out`}
          style={{ width: `${Math.min(Math.max(probNum * 100, 0), 100)}%` }}
        />

        {/* Threshold Marker Line */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-yellow-400 z-10 shadow-sm"
          style={{ left: `${Math.min(Math.max(threshNum * 100, 0), 100)}%` }}
          title={`Decision Threshold: ${threshPct}%`}
        />
      </div>

      <div className="flex justify-between text-xs text-gray-500 font-mono">
        <span>0.00% (Legitimate)</span>
        <span>Threshold Marker ({threshPct}%)</span>
        <span>100.00% (High Risk)</span>
      </div>
    </div>
  );
}
