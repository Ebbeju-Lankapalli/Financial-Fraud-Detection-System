import React from "react";

/**
 * MetricCard Component
 *
 * Display key numerical metrics with enterprise fintech aesthetic.
 */
export default function MetricCard({ title, value, subtext, icon: Icon, color = "cyan" }) {
  const colorMap = {
    cyan: {
      text: "#06b6d4",
      bg: "rgba(6, 182, 212, 0.1)",
      border: "rgba(6, 182, 212, 0.25)",
    },
    rose: {
      text: "#fb7185",
      bg: "rgba(244, 63, 94, 0.1)",
      border: "rgba(244, 63, 94, 0.25)",
    },
    emerald: {
      text: "#34d399",
      bg: "rgba(16, 185, 129, 0.1)",
      border: "rgba(16, 185, 129, 0.25)",
    },
    amber: {
      text: "#fbbf24",
      bg: "rgba(245, 158, 11, 0.1)",
      border: "rgba(245, 158, 11, 0.25)",
    },
    purple: {
      text: "#c084fc",
      bg: "rgba(168, 85, 247, 0.1)",
      border: "rgba(168, 85, 247, 0.25)",
    },
  };

  const scheme = colorMap[color] || colorMap.cyan;

  return (
    <div className="card-glass p-5 flex flex-col justify-between relative overflow-hidden">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1">
            {title}
          </p>
          <h3 className="text-2xl font-bold text-white font-mono">
            {value}
          </h3>
        </div>
        {Icon && (
          <div
            className="p-3 rounded-xl flex items-center justify-center"
            style={{ backgroundColor: scheme.bg, color: scheme.text, border: `1px solid ${scheme.border}` }}
          >
            <Icon size={22} />
          </div>
        )}
      </div>

      {subtext && (
        <div className="mt-4 pt-3 border-t border-gray-800 text-xs text-gray-400 flex items-center justify-between">
          <span>{subtext}</span>
        </div>
      )}
    </div>
  );
}
