import React, { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import {
  Bot,
  Sparkles,
  Send,
  HelpCircle,
  CheckCircle2,
  AlertTriangle,
  Cpu,
  RotateCcw,
  ListChecks,
  MessageSquareText,
  Zap
} from "lucide-react";

import { agentService } from "../services/agentService";
import RiskBadge from "../components/common/RiskBadge";
import ProbabilityBar from "../components/common/ProbabilityBar";
import LoadingSpinner from "../components/common/LoadingSpinner";
import ErrorMessage from "../components/common/ErrorMessage";

const DEFAULT_TRANSACTION = {
  card1: 12695,
  transaction_amt: 850.00,
  card2: 490,
  addr1: 325,
  card5: 226,
  purchaser_email_domain: "gmail.com",
  log_amt: "",
  C1: 12,
  C2: 10,
  C5: 0,
  C6: 6,
  C13: 30,
  C14: 8,
  D1: 120,
  D2: 120,
  D10: null,
  D15: 180,
  M4_enc: null,
  M5_enc: null,
  M6_enc: null,
};

export default function FraudAgent() {
  const location = useLocation();

  // Pre-fill transaction if state passed from Navigate/Detail page
  const initialTransaction = location.state?.transactionForm || DEFAULT_TRANSACTION;

  const [agentStatus, setAgentStatus] = useState(null);
  const [statusLoading, setStatusLoading] = useState(true);

  const [txForm, setTxForm] = useState(initialTransaction);
  const [question, setQuestion] = useState("");

  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [agentResult, setAgentResult] = useState(null);

  // Fetch Agent Runtime Status on mount
  useEffect(() => {
    async function fetchStatus() {
      setStatusLoading(true);
      try {
        const res = await agentService.getAgentStatus();
        setAgentStatus(res);
      } catch (err) {
        console.warn("Could not fetch agent status:", err);
      } finally {
        setStatusLoading(false);
      }
    }
    fetchStatus();
  }, []);

  const handleTxChange = (e) => {
    const { name, value } = e.target;
    setTxForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    setAnalyzing(true);
    setError(null);
    setAgentResult(null);

    try {
      const res = await agentService.analyzeWithAgent(txForm, question);
      setAgentResult(res);
    } catch (err) {
      setError(err.message || "Fraud agent investigation failed.");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-gray-800 pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <Bot size={24} className="text-purple-400" />
            <h1 className="text-2xl font-bold text-white tracking-tight">Fraud Investigation Agent</h1>
          </div>
          <p className="text-sm text-gray-400">AI-assisted forensic investigation grounded in CatBoost fraud risk scoring</p>
        </div>

        {/* Agent Runtime Status Badge */}
        {!statusLoading && agentStatus && (
          <div className="flex items-center space-x-3 text-xs font-mono bg-gray-900/80 p-2.5 rounded-xl border border-gray-800">
            <div className="flex items-center space-x-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  agentStatus.groq_configured ? "bg-purple-400 animate-pulse" : "bg-amber-400"
                }`}
              />
              <span className="text-gray-300">
                LLM Mode: <strong className="text-white">{agentStatus.groq_configured ? "Groq LLM Active" : "Deterministic Logic"}</strong>
              </span>
            </div>
            <span className="text-gray-600">|</span>
            <span className="text-gray-400">{agentStatus.model}</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Transaction Input Form + Question */}
        <form onSubmit={handleAnalyze} className="lg:col-span-6 space-y-6">
          <div className="card-glass p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-gray-800 pb-2">
              <h3 className="font-semibold text-white text-sm">Transaction Inputs</h3>
              <span className="text-xs text-gray-400 font-mono">20 IEEE-CIS Features</span>
            </div>

            {/* Core Required Inputs */}
            <div className="grid grid-cols-2 gap-3">
              <div className="form-group">
                <label className="form-label" htmlFor="ag_card1">Card1 Identifier *</label>
                <input id="ag_card1" name="card1" type="number" required value={txForm.card1} onChange={handleTxChange} className="form-input font-mono text-xs" />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="ag_transaction_amt">Amount ($) *</label>
                <input id="ag_transaction_amt" name="transaction_amt" type="number" step="0.01" required value={txForm.transaction_amt} onChange={handleTxChange} className="form-input font-mono text-xs" />
              </div>
            </div>

            {/* Card & Domain Info */}
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="form-group">
                <label className="text-[11px] text-gray-400" htmlFor="ag_card2">Card2</label>
                <input id="ag_card2" name="card2" type="number" value={txForm.card2} onChange={handleTxChange} className="form-input text-xs font-mono" />
              </div>
              <div className="form-group">
                <label className="text-[11px] text-gray-400" htmlFor="ag_card5">Card5</label>
                <input id="ag_card5" name="card5" type="number" value={txForm.card5} onChange={handleTxChange} className="form-input text-xs font-mono" />
              </div>
              <div className="form-group">
                <label className="text-[11px] text-gray-400" htmlFor="ag_addr1">Addr1</label>
                <input id="ag_addr1" name="addr1" type="number" value={txForm.addr1} onChange={handleTxChange} className="form-input text-xs font-mono" />
              </div>
            </div>

            <div className="form-group">
              <label className="text-[11px] text-gray-400" htmlFor="ag_purchaser_email_domain">Purchaser Email Domain</label>
              <input id="ag_purchaser_email_domain" name="purchaser_email_domain" type="text" value={txForm.purchaser_email_domain} onChange={handleTxChange} placeholder="e.g. gmail.com" className="form-input text-xs" />
            </div>

            {/* Compact C/D/M Anonymized inputs */}
            <div className="space-y-2 pt-2 border-t border-gray-800">
              <span className="text-[11px] font-mono text-cyan-400 font-semibold block">Anonymized IEEE-CIS Signals (C*, D*, M*)</span>
              <div className="grid grid-cols-4 sm:grid-cols-6 gap-2 text-xs">
                {["C1", "C2", "C5", "C6", "C13", "C14"].map((feat) => (
                  <div key={feat} className="form-group">
                    <label className="text-[10px] font-mono text-gray-500" htmlFor={`ag_${feat}`}>{feat}</label>
                    <input id={`ag_${feat}`} name={feat} type="number" value={txForm[feat]} onChange={handleTxChange} className="form-input text-xs font-mono p-1" />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Optional Investigator Question Textarea */}
          <div className="card-glass p-5 space-y-3">
            <div className="flex items-center space-x-2 text-purple-300">
              <MessageSquareText size={18} />
              <h3 className="font-semibold text-white text-sm">Investigator Question (Optional)</h3>
            </div>
            <textarea
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g., Explain why this transaction flagged high risk despite normal amount..."
              className="form-input text-xs w-full resize-none"
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={analyzing}
            className="btn-primary w-full py-3 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 shadow-purple-500/20"
          >
            <Bot size={18} />
            <span>{analyzing ? "Generating AI Investigation..." : "Run AI Investigation"}</span>
          </button>
        </form>

        {/* Right Column: AI Investigation Results */}
        <div className="lg:col-span-6 space-y-6">
          {analyzing && <LoadingSpinner message="Consulting Fraud Investigation Agent & CatBoost classifier..." />}

          {error && <ErrorMessage title="Investigation Error" message={error} />}

          {!analyzing && !agentResult && !error && (
            <div className="card-glass p-8 text-center space-y-4 border-dashed border-gray-800">
              <div className="p-4 rounded-full bg-purple-950/40 text-purple-400 w-fit mx-auto border border-purple-800">
                <Bot size={36} />
              </div>
              <h4 className="font-semibold text-gray-300">Ready for Investigation</h4>
              <p className="text-xs text-gray-400 leading-relaxed">
                Submit transaction features to generate automated risk explanations and actionable investigator recommendations.
              </p>
            </div>
          )}

          {agentResult && (
            <div className="space-y-6">
              {/* CatBoost Classifier Prediction Box */}
              <div className="card-glass p-6 space-y-4 border-purple-500/30">
                <div className="flex items-center justify-between border-b border-gray-800 pb-3">
                  <div>
                    <h3 className="font-bold text-white text-base">CatBoost Fraud Score</h3>
                    <p className="text-xs text-gray-400 font-mono">Analysis ID: {agentResult.analysis_id}</p>
                  </div>
                  <RiskBadge risk={agentResult.prediction.risk} size="md" />
                </div>

                <ProbabilityBar
                  probability={agentResult.prediction.fraud_probability}
                  threshold={agentResult.prediction.threshold}
                  risk={agentResult.prediction.risk}
                />
              </div>

              {/* AI Investigation Explanation */}
              <div className="card-glass p-6 space-y-4 border-indigo-500/30">
                <div className="flex items-center justify-between border-b border-gray-800 pb-3">
                  <div className="flex items-center space-x-2">
                    <Sparkles size={18} className="text-purple-400" />
                    <h3 className="font-bold text-white text-base">AI Investigation Explanation</h3>
                  </div>

                  {/* LLM vs Deterministic Fallback Mode Indicator */}
                  <span
                    className={`px-2.5 py-1 rounded-full text-[10px] font-mono font-semibold border ${
                      agentResult.llm_used
                        ? "bg-purple-950 text-purple-300 border-purple-800"
                        : "bg-amber-950 text-amber-300 border-amber-800"
                    }`}
                  >
                    {agentResult.llm_used ? "Groq LLM Powered" : "Deterministic Logic Fallback"}
                  </span>
                </div>

                <div className="text-xs text-gray-200 leading-relaxed font-sans bg-gray-900/60 p-4 rounded-xl border border-gray-800 whitespace-pre-line">
                  {agentResult.explanation}
                </div>
              </div>

              {/* Recommended Actions */}
              <div className="card-glass p-6 space-y-4 border-cyan-500/30">
                <div className="flex items-center space-x-2 border-b border-gray-800 pb-3">
                  <ListChecks size={18} className="text-cyan-400" />
                  <h3 className="font-bold text-white text-base">Recommended Investigator Actions</h3>
                </div>

                <ul className="space-y-2 text-xs">
                  {agentResult.recommendations && agentResult.recommendations.length > 0 ? (
                    agentResult.recommendations.map((rec, idx) => (
                      <li key={idx} className="flex items-start space-x-3 p-3 rounded-lg bg-gray-900/70 border border-gray-800">
                        <CheckCircle2 size={16} className="text-cyan-400 shrink-0 mt-0.5" />
                        <span className="text-gray-200 leading-relaxed">{rec}</span>
                      </li>
                    ))
                  ) : (
                    <li className="text-gray-500 italic">No specific recommendations provided.</li>
                  )}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
