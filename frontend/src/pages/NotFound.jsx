import React from "react";
import { Link } from "react-router-dom";
import { ShieldAlert, Home } from "lucide-react";

export default function NotFound() {
  return (
    <div className="card-glass p-12 text-center max-w-md mx-auto my-12 space-y-6">
      <div className="p-4 rounded-full bg-rose-950/40 text-rose-400 w-fit mx-auto border border-rose-800">
        <ShieldAlert size={48} />
      </div>
      <div className="space-y-2">
        <h1 className="text-3xl font-bold font-mono text-white">404</h1>
        <h2 className="text-lg font-semibold text-gray-200">Page Not Found</h2>
        <p className="text-xs text-gray-400">
          The requested page or route does not exist in the Financial Fraud Detection System.
        </p>
      </div>

      <Link to="/" className="btn-primary text-xs px-4 py-2.5 inline-flex items-center space-x-2">
        <Home size={16} />
        <span>Return to Overview</span>
      </Link>
    </div>
  );
}
