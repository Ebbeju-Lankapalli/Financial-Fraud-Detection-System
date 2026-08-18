import React, { useEffect, useState } from "react";
import {
  BarChart3,
  Cpu,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Grid,
  FileCheck,
  Layers,
  Sparkles
} from "lucide-react";

import { evaluationService } from "../services/evaluationService";
import MetricCard from "../components/common/MetricCard";
import LoadingSpinner from "../components/common/LoadingSpinner";
import ErrorMessage from "../components/common/ErrorMessage";

export default function ModelEvaluation() {
  const [activeTab, setActiveTab] = useState("production"); // 'production', 'threshold', 'full_reference'

  const [prodMetrics, setProdMetrics] = useState(null);
  const [thresholdData, setThresholdData] = useState(null);
  const [fullRefData, setFullRefData] = useState(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchEvaluations = async () => {
    setLoading(true);
    setError(null);
    try {
      const [prod, thresh, ref] = await Promise.all([
        evaluationService.getProductionEvaluation(),
        evaluationService.getProductionThreshold(),
        evaluationService.getFullReferenceEvaluation(),
      ]);
      setProdMetrics(prod);
      setThresholdData(thresh);
      setFullRefData(ref);
    } catch (err) {
      setError(err.message || "Failed to load evaluation metrics.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvaluations();
  }, []);

  if (loading) return <LoadingSpinner message="Loading production evaluation benchmarks..." />;
  if (error) return <ErrorMessage title="Evaluation Data Error" message={error} onRetry={fetchEvaluations} />;
  if (!prodMetrics) return null;

  const m = prodMetrics.metrics || {};
  const cm = prodMetrics.confusion_matrix || {};

  const formatPct = (val) => (typeof val === "number" ? `${(val * 100).toFixed(2)}%` : "N/A");
  const formatDec = (val) => (typeof val === "number" ? val.toFixed(4) : "N/A");

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-1 rounded font-mono text-xs bg-cyan-950 text-cyan-300 border border-cyan-800">
              Active Production Classifier
            </span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight mt-1">Production Model Evaluation</h1>
          <p className="text-sm text-gray-400">Strict chronological test set evaluation for CatBoost 20-feature classifier</p>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center space-x-1 p-1 bg-gray-900 rounded-xl border border-gray-800 text-xs">
          <button
            onClick={() => setActiveTab("production")}
            className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
              activeTab === "production"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "text-gray-400 hover:text-white"
            }`}
          >
            Production Model
          </button>
          <button
            onClick={() => setActiveTab("threshold")}
            className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
              activeTab === "threshold"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "text-gray-400 hover:text-white"
            }`}
          >
            Threshold Selection
          </button>
          <button
            onClick={() => setActiveTab("full_reference")}
            className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
              activeTab === "full_reference"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "text-gray-400 hover:text-white"
            }`}
          >
            Full Reference Model
          </button>
        </div>
      </div>

      {/* TAB 1: PRODUCTION MODEL EVALUATION */}
      {activeTab === "production" && (
        <div className="space-y-8">
          {/* Metadata Bar */}
          <div className="card-glass p-5 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-4 text-xs font-mono">
            <div>
              <span className="text-gray-500 block uppercase text-[10px]">Model Family</span>
              <span className="text-white font-semibold">{prodMetrics.model_family}</span>
            </div>
            <div>
              <span className="text-gray-500 block uppercase text-[10px]">Dataset</span>
              <span className="text-cyan-400 font-semibold">{prodMetrics.dataset}</span>
            </div>
            <div>
              <span className="text-gray-500 block uppercase text-[10px]">Features</span>
              <span className="text-white">{prodMetrics.feature_count} features</span>
            </div>
            <div>
              <span className="text-gray-500 block uppercase text-[10px]">Decision Threshold</span>
              <span className="text-amber-400 font-bold">{prodMetrics.threshold}</span>
            </div>
            <div>
              <span className="text-gray-500 block uppercase text-[10px]">Test Split Rows</span>
              <span className="text-white">{prodMetrics.test_rows?.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-gray-500 block uppercase text-[10px]">Fraud Test Cases</span>
              <span className="text-rose-400 font-bold">{prodMetrics.fraud_rows?.toLocaleString()}</span>
            </div>
          </div>

          {/* Held-out Test Set Disclaimer */}
          {prodMetrics.test_used_for_model_selection === false && (
            <div className="p-4 rounded-xl bg-emerald-950/20 border border-emerald-900/40 text-emerald-300 text-xs flex items-start space-x-3">
              <CheckCircle2 size={18} className="text-emerald-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <p className="font-semibold">Unbiased Chronological Test Evaluation</p>
                <p className="text-gray-300 leading-relaxed">
                  The chronological test set was held out from model and threshold selection. All evaluation metrics below reflect untouched generalization performance.
                </p>
              </div>
            </div>
          )}

          {/* Primary Model Metrics Grid */}
          <div className="space-y-4">
            <h3 className="font-bold text-white text-lg tracking-tight">Performance Metrics</h3>
            <p className="text-xs text-gray-400">
              Emphasizing ROC-AUC, PR-AUC, Precision, Recall, and F1 score for imbalanced fraud screening.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <MetricCard
                title="ROC-AUC Score"
                value={formatDec(m.roc_auc)}
                subtext="Area under ROC Curve"
                icon={BarChart3}
                color="cyan"
              />
              <MetricCard
                title="PR-AUC Score"
                value={formatDec(m.pr_auc)}
                subtext="Precision-Recall Area"
                icon={Sparkles}
                color="purple"
              />
              <MetricCard
                title="F1 Score"
                value={formatDec(m.f1)}
                subtext="Harmonic mean of P & R"
                icon={FileCheck}
                color="amber"
              />
              <MetricCard
                title="Precision"
                value={formatPct(m.precision)}
                subtext="True Positives / Predicted Positive"
                icon={CheckCircle2}
                color="emerald"
              />
              <MetricCard
                title="Recall (Sensitivity)"
                value={formatPct(m.recall)}
                subtext="True Positives / Actual Fraud"
                icon={CheckCircle2}
                color="rose"
              />
              <MetricCard
                title="Accuracy"
                value={formatPct(m.accuracy)}
                subtext="Overall correct predictions"
                icon={Grid}
                color="cyan"
              />
            </div>
          </div>

          {/* Confusion Matrix Section */}
          <div className="space-y-4">
            <h3 className="font-bold text-white text-lg tracking-tight">Test Confusion Matrix</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Matrix Layout */}
              <div className="card-glass p-6 space-y-4">
                <div className="grid grid-cols-2 gap-3 text-center">
                  <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-900/60 space-y-1">
                    <span className="text-[10px] uppercase font-mono text-emerald-400 font-semibold block">True Negatives</span>
                    <span className="text-2xl font-bold font-mono text-white">{cm.true_negatives?.toLocaleString()}</span>
                    <span className="text-[11px] text-gray-400 block">Legitimate correctly identified</span>
                  </div>

                  <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-900/60 space-y-1">
                    <span className="text-[10px] uppercase font-mono text-amber-400 font-semibold block">False Positives</span>
                    <span className="text-2xl font-bold font-mono text-white">{cm.false_positives?.toLocaleString()}</span>
                    <span className="text-[11px] text-gray-400 block">Legitimate flagged as high risk</span>
                  </div>

                  <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-900/60 space-y-1">
                    <span className="text-[10px] uppercase font-mono text-rose-400 font-semibold block">False Negatives</span>
                    <span className="text-2xl font-bold font-mono text-white">{cm.false_negatives?.toLocaleString()}</span>
                    <span className="text-[11px] text-gray-400 block">Fraud missed by threshold</span>
                  </div>

                  <div className="p-4 rounded-xl bg-cyan-950/40 border border-cyan-900/60 space-y-1">
                    <span className="text-[10px] uppercase font-mono text-cyan-400 font-semibold block">True Positives</span>
                    <span className="text-2xl font-bold font-mono text-white">{cm.true_positives?.toLocaleString()}</span>
                    <span className="text-[11px] text-gray-400 block">Fraud correctly detected</span>
                  </div>
                </div>
              </div>

              {/* Explanatory Context */}
              <div className="card-glass p-6 space-y-4 flex flex-col justify-between">
                <div className="space-y-3">
                  <h4 className="font-semibold text-white text-sm">Evaluating High Threshold (0.83)</h4>
                  <p className="text-xs text-gray-300 leading-relaxed">
                    Fraud detection models operating in enterprise financial pipelines require high precision to avoid overwhelming investigator queues with false alarms.
                  </p>
                  <p className="text-xs text-gray-400 leading-relaxed">
                    The 0.83 decision threshold was explicitly calibrated on validation data to optimize high-confidence risk screening.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-gray-900 text-xs font-mono border border-gray-800 text-gray-400">
                  <span>Evaluation Split: <strong>{prodMetrics.evaluation_split}</strong></span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: THRESHOLD SELECTION INFORMATION */}
      {activeTab === "threshold" && thresholdData && (
        <div className="space-y-6">
          <div className="card-glass p-6 space-y-6">
            <div className="flex items-center space-x-3 border-b border-gray-800 pb-4">
              <Sliders className="text-amber-400" size={22} />
              <div>
                <h3 className="font-bold text-white text-lg">Threshold Selection Strategy</h3>
                <p className="text-xs text-gray-400 font-mono">Validation-driven decision boundary optimization</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
              <div className="p-4 rounded-xl bg-gray-900 border border-gray-800 space-y-1">
                <span className="text-gray-500 uppercase text-[10px]">Selected Threshold</span>
                <span className="text-2xl font-bold text-amber-400 block">{thresholdData.threshold || 0.83}</span>
              </div>
              <div className="p-4 rounded-xl bg-gray-900 border border-gray-800 space-y-1">
                <span className="text-gray-500 uppercase text-[10px]">Selection Dataset</span>
                <span className="text-lg font-semibold text-white block">{thresholdData.selection_dataset || "Validation Split"}</span>
              </div>
              <div className="p-4 rounded-xl bg-gray-900 border border-gray-800 space-y-1">
                <span className="text-gray-500 uppercase text-[10px]">Optimization Metric</span>
                <span className="text-lg font-semibold text-cyan-400 block">{thresholdData.selection_metric || "F1 / Precision-Recall"}</span>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800 text-xs text-gray-300 space-y-2 leading-relaxed">
              <p className="font-semibold text-white">Why 0.83 threshold?</p>
              <p>
                Default 0.50 thresholds often produce unmanageable false-positive rates on skewed financial datasets. By optimizing on validation data, 0.83 filters out marginal noise while preserving strong true-positive signals for high-risk triage.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: FULL REFERENCE MODEL */}
      {activeTab === "full_reference" && fullRefData && (
        <div className="space-y-6">
          <div className="card-glass p-6 space-y-6">
            <div className="flex items-center space-x-3 border-b border-gray-800 pb-4">
              <Layers className="text-purple-400" size={22} />
              <div>
                <h3 className="font-bold text-white text-lg">Full Feature Reference Model (61 Features)</h3>
                <p className="text-xs text-gray-400 font-mono">Baseline CatBoost model trained on full feature set</p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-purple-950/20 border border-purple-900/40 text-purple-300 text-xs leading-relaxed space-y-2">
              <p className="font-semibold">Architectural Tradeoff Rationale</p>
              <p>
                The production deployment selected the reduced 20-feature model over the full 61-feature reference model to minimize feature extraction latency, payload sizes, and upstream missingness while maintaining equivalent predictive power.
              </p>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono bg-gray-900/60 p-4 rounded-xl border border-gray-800">
              <div>
                <span className="text-gray-500 block text-[10px] uppercase">Reference Model</span>
                <span className="text-white font-semibold">{fullRefData.model || "CatBoost 61-Feature"}</span>
              </div>
              <div>
                <span className="text-gray-500 block text-[10px] uppercase">Feature Count</span>
                <span className="text-purple-400 font-bold">61 Features</span>
              </div>
              <div>
                <span className="text-gray-500 block text-[10px] uppercase">ROC-AUC</span>
                <span className="text-white font-bold">{formatDec(fullRefData.metrics?.roc_auc)}</span>
              </div>
              <div>
                <span className="text-gray-500 block text-[10px] uppercase">PR-AUC</span>
                <span className="text-white font-bold">{formatDec(fullRefData.metrics?.pr_auc)}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
