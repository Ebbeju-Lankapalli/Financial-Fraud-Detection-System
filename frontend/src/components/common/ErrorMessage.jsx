import React from "react";
import { AlertCircle, RefreshCw } from "lucide-react";

export default function ErrorMessage({ title = "API Request Failed", message, onRetry }) {
  return (
    <div className="card-glass p-6 border-rose-900/50 bg-rose-950/20 text-rose-200 rounded-xl space-y-4">
      <div className="flex items-start space-x-3">
        <AlertCircle size={22} className="text-rose-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h4 className="font-semibold text-rose-300">{title}</h4>
          <p className="text-sm text-rose-200/80 leading-relaxed font-mono text-xs">
            {message || "An error occurred while communicating with the Fraud Detection API."}
          </p>
        </div>
      </div>

      {onRetry && (
        <div className="pt-2 flex justify-end">
          <button
            onClick={onRetry}
            className="btn-secondary text-xs px-3 py-1.5 flex items-center space-x-2 text-rose-200 border-rose-800 hover:bg-rose-900/40"
          >
            <RefreshCw size={14} />
            <span>Retry Request</span>
          </button>
        </div>
      )}
    </div>
  );
}
