import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ShieldCheck,
  ShieldAlert,
  Send,
  RotateCcw,
  Sparkles,
  HelpCircle,
  Bot,
  AlertCircle,
  CheckCircle2,
  FileText
} from "lucide-react";

import { transactionService } from "../services/transactionService";
import RiskBadge from "../components/common/RiskBadge";
import ProbabilityBar from "../components/common/ProbabilityBar";
import LoadingSpinner from "../components/common/LoadingSpinner";
import ErrorMessage from "../components/common/ErrorMessage";

const DEFAULT_FORM = {
  card1: 10000,
  transaction_amt: 150.00,
  card2: "",
  addr1: "",
  card5: "",
  purchaser_email_domain: "",
  log_amt: "",
  C1: "",
  C2: "",
  C5: "",
  C6: "",
  C13: "",
  C14: "",
  D1: "",
  D2: "",
  D10: "",
  D15: "",
  M4_enc: "",
  M5_enc: "",
  M6_enc: "",
};

const LOW_RISK_PRESET = {
  card1: 13979,
  transaction_amt: 49.00,
  card2: 321,
  addr1: 315,
  card5: 226,
  purchaser_email_domain: "gmail.com",
  log_amt: "",
  C1: 1,
  C2: 1,
  C5: 0,
  C6: 1,
  C13: 1,
  C14: 1,
  D1: 0,
  D2: 0,
  D10: 12,
  D15: 0,
  M4_enc: 0,
  M5_enc: 0,
  M6_enc: 0,
};

const HIGH_RISK_PRESET = {
  card1: 12695,
  transaction_amt: 150.00,
  card2: 490,
  addr1: 325,
  card5: 226,
  purchaser_email_domain: "gmail.com",
  log_amt: "",
  C1: 9,
  C2: 4,
  C5: 0,
  C6: 2,
  C13: 1,
  C14: 1,
  D1: 0,
  D2: null,
  D10: null,
  D15: null,
  M4_enc: null,
  M5_enc: null,
  M6_enc: null,
};

export default function AnalyzeTransaction() {
  const navigate = useNavigate();
  const [form, setForm] = useState(DEFAULT_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handlePreset = (preset) => {
    setForm(preset);
    setResult(null);
    setError(null);
  };

  const handleReset = () => {
    setForm(DEFAULT_FORM);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await transactionService.analyzeTransaction(form);
      setResult(response);
    } catch (err) {
      setError(err.message || "Failed to complete transaction analysis.");
    } finally {
      setLoading(false);
    }
  };

  const handleInvestigateWithAgent = () => {
    if (result) {
      navigate("/agent", { state: { transactionForm: form, analysisResult: result } });
    }
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-gray-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Analyze Transaction</h1>
          <p className="text-sm text-gray-400">Evaluate transaction risk against the 20-feature CatBoost production model</p>
        </div>

        {/* Quick Sample Presets */}
        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={() => handlePreset(LOW_RISK_PRESET)}
            className="btn-secondary text-xs px-3 py-1.5 text-emerald-400 border-emerald-900/60 hover:bg-emerald-950/40"
          >
            <Sparkles size={14} />
            <span>Load Low Risk Sample</span>
          </button>
          <button
            type="button"
            onClick={() => handlePreset(HIGH_RISK_PRESET)}
            className="btn-secondary text-xs px-3 py-1.5 text-rose-400 border-rose-900/60 hover:bg-rose-950/40"
          >
            <Sparkles size={14} />
            <span>Load High Risk Sample</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Transaction Form (Left Column) */}
        <form onSubmit={handleSubmit} className="lg:col-span-7 space-y-6">
          {/* Required Fields Section */}
          <div className="card-glass p-5 space-y-4 border-cyan-900/50">
            <div className="flex items-center space-x-2 text-cyan-400 border-b border-gray-800 pb-2">
              <ShieldCheck size={18} />
              <h3 className="font-semibold text-white text-sm">Required Transaction Data</h3>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="form-group">
                <label className="form-label" htmlFor="card1">
                  <span>Card1 Identifier *</span>
                </label>
                <input
                  id="card1"
                  name="card1"
                  type="number"
                  required
                  min="0"
                  step="1"
                  value={form.card1}
                  onChange={handleChange}
                  placeholder="e.g. 13979"
                  className="form-input font-mono"
                />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="transaction_amt">
                  <span>Transaction Amount ($) *</span>
                </label>
                <input
                  id="transaction_amt"
                  name="transaction_amt"
                  type="number"
                  required
                  min="0"
                  step="0.01"
                  value={form.transaction_amt}
                  onChange={handleChange}
                  placeholder="e.g. 49.00"
                  className="form-input font-mono"
                />
              </div>
            </div>
          </div>

          {/* Optional Card & Address Details */}
          <div className="card-glass p-5 space-y-4">
            <h3 className="font-semibold text-white text-sm border-b border-gray-800 pb-2">
              Card & Purchaser Metadata (Optional)
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="form-group">
                <label className="form-label" htmlFor="card2">Card2 Code</label>
                <input id="card2" name="card2" type="number" value={form.card2} onChange={handleChange} placeholder="null" className="form-input font-mono" />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="card5">Card5 Code</label>
                <input id="card5" name="card5" type="number" value={form.card5} onChange={handleChange} placeholder="null" className="form-input font-mono" />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="addr1">Addr1 Region</label>
                <input id="addr1" name="addr1" type="number" value={form.addr1} onChange={handleChange} placeholder="null" className="form-input font-mono" />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="form-group">
                <label className="form-label" htmlFor="purchaser_email_domain">Purchaser Email Domain</label>
                <input id="purchaser_email_domain" name="purchaser_email_domain" type="text" value={form.purchaser_email_domain} onChange={handleChange} placeholder="e.g. gmail.com" className="form-input" />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="log_amt">
                  <span>Log Amount</span>
                  <span className="text-[10px] text-gray-500 font-mono">(auto-derived if blank)</span>
                </label>
                <input id="log_amt" name="log_amt" type="number" step="0.0001" value={form.log_amt} onChange={handleChange} placeholder="Derived automatically" className="form-input font-mono" />
              </div>
            </div>
          </div>

          {/* Anonymized IEEE-CIS Features (C*, D*, M*) */}
          <div className="card-glass p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-gray-800 pb-2">
              <h3 className="font-semibold text-white text-sm">Anonymized IEEE-CIS Features</h3>
              <div className="flex items-center space-x-1 text-xs text-gray-400">
                <HelpCircle size={14} />
                <span>Masked dataset attributes</span>
              </div>
            </div>

            {/* C Features */}
            <div className="space-y-2">
              <span className="text-xs font-mono font-semibold text-cyan-400">Count (C*) Features</span>
              <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
                {["C1", "C2", "C5", "C6", "C13", "C14"].map((feat) => (
                  <div key={feat} className="form-group">
                    <label className="text-[11px] font-mono text-gray-400" htmlFor={feat}>{feat}</label>
                    <input id={feat} name={feat} type="number" value={form[feat]} onChange={handleChange} placeholder="null" className="form-input text-xs font-mono p-1.5" />
                  </div>
                ))}
              </div>
            </div>

            {/* D Features */}
            <div className="space-y-2 pt-2">
              <span className="text-xs font-mono font-semibold text-cyan-400">Time-Delta (D*) Features</span>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {["D1", "D2", "D10", "D15"].map((feat) => (
                  <div key={feat} className="form-group">
                    <label className="text-[11px] font-mono text-gray-400" htmlFor={feat}>{feat}</label>
                    <input id={feat} name={feat} type="number" value={form[feat]} onChange={handleChange} placeholder="null" className="form-input text-xs font-mono p-1.5" />
                  </div>
                ))}
              </div>
            </div>

            {/* M Features */}
            <div className="space-y-2 pt-2">
              <span className="text-xs font-mono font-semibold text-cyan-400">Encoded Match (M*_enc) Features</span>
              <div className="grid grid-cols-3 gap-2">
                {["M4_enc", "M5_enc", "M6_enc"].map((feat) => (
                  <div key={feat} className="form-group">
                    <label className="text-[11px] font-mono text-gray-400" htmlFor={feat}>{feat}</label>
                    <input id={feat} name={feat} type="number" value={form[feat]} onChange={handleChange} placeholder="null" className="form-input text-xs font-mono p-1.5" />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Buttons */}
          <div className="flex items-center space-x-3 pt-2">
            <button type="submit" disabled={loading} className="btn-primary flex-1 py-3">
              <Send size={18} />
              <span>{loading ? "Analyzing Transaction..." : "Submit for Fraud Screening"}</span>
            </button>
            <button type="button" onClick={handleReset} className="btn-secondary py-3 px-4">
              <RotateCcw size={18} />
            </button>
          </div>
        </form>

        {/* Prediction Results Display (Right Column) */}
        <div className="lg:col-span-5 space-y-6">
          {loading && <LoadingSpinner message="Running CatBoost 20-feature fraud detection model..." />}

          {error && <ErrorMessage title="Analysis Error" message={error} />}

          {!loading && !result && !error && (
            <div className="card-glass p-8 text-center space-y-4 border-dashed border-gray-800">
              <div className="p-4 rounded-full bg-gray-900 w-fit mx-auto text-gray-500">
                <ShieldCheck size={36} />
              </div>
              <h4 className="font-semibold text-gray-300">Ready for Analysis</h4>
              <p className="text-xs text-gray-400 leading-relaxed">
                Fill in transaction details or load a sample preset to compute probability against the active CatBoost classifier.
              </p>
            </div>
          )}

          {result && (
            <div className="card-glass p-6 space-y-6 border-cyan-500/30 bg-gradient-to-b from-[#111827] to-[#0f172a]">
              <div className="flex items-center justify-between border-b border-gray-800 pb-4">
                <div>
                  <h3 className="font-bold text-white text-lg">Analysis Result</h3>
                  <p className="text-xs text-gray-400 font-mono">ID: {result.analysis_id}</p>
                </div>
                <RiskBadge risk={result.prediction.risk} size="md" />
              </div>

              {/* Fraud Probability Visual Bar */}
              <ProbabilityBar
                probability={result.prediction.fraud_probability}
                threshold={result.prediction.threshold}
                risk={result.prediction.risk}
              />

              {/* Model Metadata Grid */}
              <div className="grid grid-cols-2 gap-3 text-xs font-mono pt-2 bg-gray-900/60 p-4 rounded-xl border border-gray-800">
                <div>
                  <span className="text-gray-500 block text-[10px] uppercase">Decision Model</span>
                  <span className="text-white font-semibold">{result.prediction.model}</span>
                </div>
                <div>
                  <span className="text-gray-500 block text-[10px] uppercase">Decision Source</span>
                  <span className="text-cyan-400 font-semibold">{result.prediction.decision_source}</span>
                </div>
                <div>
                  <span className="text-gray-500 block text-[10px] uppercase">Feature Count</span>
                  <span className="text-white">{result.prediction.feature_count} IEEE-CIS</span>
                </div>
                <div>
                  <span className="text-gray-500 block text-[10px] uppercase">Valid Output</span>
                  <span className={result.prediction.valid_output ? "text-emerald-400 font-semibold" : "text-rose-400"}>
                    {result.prediction.valid_output ? "True" : "False"}
                  </span>
                </div>
              </div>

              {/* Disclaimer */}
              <div className="p-3 rounded-lg bg-amber-950/30 border border-amber-900/40 text-amber-200 text-xs flex items-start space-x-2">
                <AlertCircle size={16} className="text-amber-400 shrink-0 mt-0.5" />
                <span>
                  <strong>Disclaimer:</strong> AI-assisted fraud screening — not a final fraud determination.
                </span>
              </div>

              {/* Action Button to AI Agent */}
              <button
                type="button"
                onClick={handleInvestigateWithAgent}
                className="btn-secondary w-full justify-center py-2.5 text-xs text-purple-300 border-purple-900/60 hover:bg-purple-950/40"
              >
                <Bot size={16} />
                <span>Investigate with AI Agent →</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
