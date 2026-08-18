import React from "react";
import { ShieldAlert, ShieldCheck } from "lucide-react";

/**
 * RiskBadge Component
 *
 * Displays HIGH or LOW risk status with icon, color, and proper wording.
 * Never refers to LOW risk as "fraud-free".
 */
export default function RiskBadge({ risk, size = "md" }) {
  const isHigh = risk === "HIGH";

  const paddingClass = size === "sm" ? "px-2 py-0.5 text-xs" : "px-3 py-1 text-sm";
  const iconSize = size === "sm" ? 14 : 16;

  if (isHigh) {
    return (
      <span className={`badge-high ${paddingClass}`}>
        <ShieldAlert size={iconSize} className="text-rose-400" />
        High Risk
      </span>
    );
  }

  return (
    <span className={`badge-low ${paddingClass}`}>
      <ShieldCheck size={iconSize} className="text-emerald-400" />
      Low Risk
    </span>
  );
}
