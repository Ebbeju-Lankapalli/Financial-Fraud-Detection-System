import React, { useEffect, useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  ShieldCheck,
  Bot,
  Calendar,
  Key,
  Database,
  HelpCircle,
  AlertCircle
} from "lucide-react";

import { transactionService } from "../services/transactionService";
import RiskBadge from "../components/common/RiskBadge";
import ProbabilityBar from "../components/common/ProbabilityBar";
import LoadingSpinner from "../components/common/LoadingSpinner";
import ErrorMessage from "../components/common/ErrorMessage";

export default function TransactionDetail() {
  const { analysisId } = useParams();
  const navigate = useNavigate();

  const [record, setRecord] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchDetail() {
      if (!analysisId) return;
      setLoading(true);
      setError(null);
      try {
        const data = await transactionService.getTransactionById(analysisId);
        setRecord(data);
      } catch (err) {
        setError(err.message || `Failed to fetch analysis record for ID: ${analysisId}`);
      } finally {
        setLoading(false);
      }
    }
    fetchDetail();
  }, [analysisId]);

  if (loading) return <LoadingSpinner message="Fetching analysis detail record..." />;
  if (error) return <ErrorMessage title="Record Error" message={error} />;
  if (!record) return null;

  // Prepare 20 IEEE-CIS features for visual presentation
  const ieeeFeatures = [
    { key: "card1", label: "Card1 (Issuer ID)", val: record.card1, category: "Core" },
    { key: "transaction_amt", label: "Transaction Amount ($)", val: record.transaction_amt ? `$${record.transaction_amt.toFixed(2)}` : "null", category: "Core" },
    { key: "card2", label: "Card2 Code", val: record.card2, category: "Card/Purchaser" },
    { key: "card5", label: "Card5 Code", val: record.card5, category: "Card/Purchaser" },
    { key: "addr1", label: "Addr1 Region Code", val: record.addr1, category: "Card/Purchaser" },
    { key: "purchaser_email_domain", label: "Purchaser Email Domain", val: record.purchaser_email_domain || "null", category: "Card/Purchaser" },
    { key: "log_amt", label: "Log Amount (Derived)", val: record.log_amt, category: "Core" },

    { key: "C1", label: "C1 — Anonymized Feature", val: record.C1, category: "Count (C*)" },
    { key: "C2", label: "C2 — Anonymized Feature", val: record.C2, category: "Count (C*)" },
    { key: "C5", label: "C5 — Anonymized Feature", val: record.C5, category: "Count (C*)" },
    { key: "C6", label: "C6 — Anonymized Feature", val: record.C6, category: "Count (C*)" },
    { key: "C13", label: "C13 — Anonymized Feature", val: record.C13, category: "Count (C*)" },
    { key: "C14", label: "C14 — Anonymized Feature", val: record.C14, category: "Count (C*)" },

    { key: "D1", label: "D1 — Anonymized Time Delta", val: record.D1, category: "Time-Delta (D*)" },
    { key: "D2", label: "D2 — Anonymized Time Delta", val: record.D2, category: "Time-Delta (D*)" },
    { key: "D10", label: "D10 — Anonymized Time Delta", val: record.D10, category: "Time-Delta (D*)" },
    { key: "D15", label: "D15 — Anonymized Time Delta", val: record.D15, category: "Time-Delta (D*)" },

    { key: "M4_enc", label: "M4_enc — Encoded Match Feature", val: record.M4_enc, category: "Match (M*_enc)" },
    { key: "M5_enc", label: "M5_enc — Encoded Match Feature", val: record.M5_enc, category: "Match (M*_enc)" },
    { key: "M6_enc", label: "M6_enc — Encoded Match Feature", val: record.M6_enc, category: "Match (M*_enc)" },
  ];

  const handleLaunchAgent = () => {
    // Construct transaction form object from record
    const formObj = {
      card1: record.card1,
      transaction_amt: record.transaction_amt,
      card2: record.card2 ?? "",
      addr1: record.addr1 ?? "",
      card5: record.card5 ?? "",
      purchaser_email_domain: record.purchaser_email_domain ?? "",
      log_amt: record.log_amt ?? "",
      C1: record.C1 ?? "",
      C2: record.C2 ?? "",
      C5: record.C5 ?? "",
      C6: record.C6 ?? "",
      C13: record.C13 ?? "",
      C14: record.C14 ?? "",
      D1: record.D1 ?? "",
      D2: record.D2 ?? "",
      D10: record.D10 ?? "",
      D15: record.D15 ?? "",
      M4_enc: record.M4_enc ?? "",
      M5_enc: record.M5_enc ?? "",
      M6_enc: record.M6_enc ?? "",
    };

    navigate("/agent", { state: { transactionForm: formObj, analysisId: record.analysis_id } });
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Top Bar */}
      <div className="flex items-center justify-between">
        <Link to="/history" className="text-xs text-gray-400 hover:text-white flex items-center space-x-2 font-medium">
          <ArrowLeft size={16} />
          <span>Back to Audit History</span>
        </Link>

        <button
          onClick={handleLaunchAgent}
          className="btn-secondary text-xs px-3.5 py-1.5 text-purple-300 border-purple-900/60 hover:bg-purple-950/40"
        >
          <Bot size={16} />
          <span>Investigate with AI Agent</span>
        </button>
      </div>

      {/* Overview Card */}
      <div className="card-glass p-6 space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-gray-800 pb-4">
          <div>
            <div className="flex items-center space-x-2 text-xs text-gray-400 font-mono mb-1">
              <Key size={14} className="text-cyan-400" />
              <span>Analysis ID:</span>
            </div>
            <h1 className="text-xl md:text-2xl font-bold font-mono text-white tracking-tight">
              {record.analysis_id}
            </h1>
            <p className="text-xs text-gray-400 font-mono mt-1 flex items-center space-x-1">
              <Calendar size={12} />
              <span>Created At: {new Date(record.created_at).toLocaleString()}</span>
            </p>
          </div>

          <RiskBadge risk={record.risk} size="md" />
        </div>

        {/* Fraud Probability Visual Bar */}
        <ProbabilityBar
          probability={record.fraud_probability}
          threshold={record.threshold}
          risk={record.risk}
        />

        {/* Prediction Properties Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 rounded-xl bg-gray-900/60 border border-gray-800 text-xs font-mono">
          <div>
            <span className="text-gray-500 block text-[10px] uppercase">Model</span>
            <span className="text-white font-semibold">{record.model}</span>
          </div>
          <div>
            <span className="text-gray-500 block text-[10px] uppercase">Decision Source</span>
            <span className="text-cyan-400 font-semibold">{record.decision_source}</span>
          </div>
          <div>
            <span className="text-gray-500 block text-[10px] uppercase">Feature Count</span>
            <span className="text-white">{record.feature_count} IEEE-CIS</span>
          </div>
          <div>
            <span className="text-gray-500 block text-[10px] uppercase">Valid Output</span>
            <span className={record.valid_output ? "text-emerald-400 font-semibold" : "text-rose-400"}>
              {record.valid_output ? "True" : "False"}
            </span>
          </div>
        </div>
      </div>

      {/* Submitted 20 Transaction Inputs Grid */}
      <div className="card-glass p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-gray-800 pb-3">
          <div className="flex items-center space-x-2">
            <Database size={18} className="text-cyan-400" />
            <h2 className="font-bold text-white text-base">Submitted Transaction Inputs (20 Features)</h2>
          </div>
          <div className="flex items-center space-x-1 text-xs text-gray-400">
            <HelpCircle size={14} />
            <span>C*, D*, M* features are anonymized IEEE-CIS attributes</span>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {ieeeFeatures.map((item) => (
            <div
              key={item.key}
              className="p-3 rounded-lg bg-gray-900/70 border border-gray-800 flex flex-col justify-between"
            >
              <div className="flex items-center justify-between text-[11px] text-gray-400 mb-1">
                <span>{item.label}</span>
                <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-gray-800 text-cyan-300">
                  {item.category}
                </span>
              </div>
              <div className="font-mono text-sm font-semibold text-white">
                {item.val === null || item.val === undefined ? (
                  <span className="text-gray-500 italic text-xs">null</span>
                ) : (
                  String(item.val)
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Safety Disclaimer Card */}
      <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-900/40 text-amber-200 text-xs flex items-start space-x-3">
        <AlertCircle size={18} className="text-amber-400 shrink-0 mt-0.5" />
        <div className="space-y-1 leading-relaxed">
          <p className="font-semibold text-amber-300">Important Fraud Screening Notice</p>
          <p className="text-gray-300">
            AI-assisted fraud screening — not a final fraud determination. High risk classifications indicate statistical anomaly relative to the IEEE-CIS baseline and require human investigator verification.
          </p>
        </div>
      </div>
    </div>
  );
}
