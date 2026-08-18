import React from "react";
import { AlertCircle } from "lucide-react";

export default function Footer() {
  return (
    <footer className="mt-auto border-t border-[#1f293d] bg-[#0b0f19] px-6 py-4 text-xs text-gray-500">
      <div className="flex flex-col md:flex-row items-center justify-between gap-3 max-w-7xl mx-auto">
        <div className="flex items-center space-x-2 text-gray-400">
          <AlertCircle size={14} className="text-amber-400 shrink-0" />
          <span>
            <strong className="text-gray-300">Disclaimer:</strong> AI-assisted fraud screening — not a final fraud determination.
          </span>
        </div>

        <div className="flex items-center space-x-4 font-mono text-[11px]">
          <span>Decision Source: <strong className="text-gray-300">catboost_ieee_cis</strong></span>
          <span>•</span>
          <span>Threshold: <strong className="text-gray-300">0.83</strong></span>
        </div>
      </div>
    </footer>
  );
}
