import React, { useEffect, useState } from "react";
import {
  FlaskConical,
  AlertTriangle,
  GitCompare,
  Cpu,
  Sparkles,
  BookOpen,
  ArrowRight,
  Info
} from "lucide-react";

import { evaluationService } from "../services/evaluationService";
import LoadingSpinner from "../components/common/LoadingSpinner";
import ErrorMessage from "../components/common/ErrorMessage";

export default function Research() {
  const [researchData, setResearchData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchResearch = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await evaluationService.getResearchEvaluation();
      setResearchData(res);
    } catch (err) {
      setError(err.message || "Failed to load QLoRA research evaluation.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResearch();
  }, []);

  if (loading) return <LoadingSpinner message="Loading LLM QLoRA research experiment benchmarks..." />;
  if (error) return <ErrorMessage title="Research Data Error" message={error} onRetry={fetchResearch} />;
  if (!researchData) return null;

  const base = researchData.base_model || {};
  const finetuned = researchData.finetuned_model || {};
  const comp = researchData.comparison || {};

  const formatDec = (val) => (typeof val === "number" ? val.toFixed(4) : "N/A");
  const formatPct = (val) => (typeof val === "number" ? `${(val * 100).toFixed(2)}%` : "N/A");

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="badge-research">
              <FlaskConical size={14} />
              Research & Fine-Tuning Experiment
            </span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight mt-1">
            Qwen2.5 + QLoRA Fine-Tuning Research
          </h1>
          <p className="text-sm text-gray-400">Parameter-efficient fine-tuning (PEFT) experiment on 1.5B instruction LLM</p>
        </div>
      </div>

      {/* Prominent Disclaimer Banner */}
      <div className="p-5 rounded-xl bg-purple-950/30 border border-purple-800/60 text-purple-200 space-y-2">
        <div className="flex items-center space-x-2 text-purple-300 font-bold text-sm">
          <AlertTriangle size={18} className="text-purple-400 shrink-0" />
          <span>Research Experiment Notice — NOT Active Production Model</span>
        </div>
        <p className="text-xs text-purple-200/90 leading-relaxed font-sans">
          This model is preserved strictly as a fine-tuning research experiment and is <strong>NOT</strong> the active production fraud decision source. The production classifier remains <code className="font-mono text-cyan-300 bg-gray-900 px-1.5 py-0.5 rounded">catboost_ieee_cis</code>.
        </p>
      </div>

      {/* Comparison Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Base Qwen Model Card */}
        <div className="card-glass p-6 space-y-4 border-gray-800">
          <div className="flex items-center space-x-3 border-b border-gray-800 pb-3">
            <div className="p-2 rounded-lg bg-gray-800 text-gray-400">
              <Cpu size={20} />
            </div>
            <div>
              <h3 className="font-bold text-white text-base">Zero-Shot Base LLM</h3>
              <p className="text-xs text-gray-400 font-mono">{base.model_name || "Qwen2.5-1.5B-Instruct"}</p>
            </div>
          </div>

          <p className="text-xs text-gray-400 leading-relaxed">
            Baseline evaluation of un-tuned Qwen2.5-1.5B instructed zero-shot on financial transaction prompt templates.
          </p>

          <div className="grid grid-cols-2 gap-3 text-xs font-mono bg-gray-900/60 p-4 rounded-xl border border-gray-800">
            <div>
              <span className="text-gray-500 block text-[10px] uppercase">Accuracy</span>
              <span className="text-white font-semibold">{formatPct(base.metrics?.accuracy)}</span>
            </div>
            <div>
              <span className="text-gray-500 block text-[10px] uppercase">Precision</span>
              <span className="text-white font-semibold">{formatPct(base.metrics?.precision)}</span>
            </div>
            <div>
              <span className="text-gray-500 block text-[10px] uppercase">Recall</span>
              <span className="text-white font-semibold">{formatPct(base.metrics?.recall)}</span>
            </div>
            <div>
              <span className="text-gray-500 block text-[10px] uppercase">F1 Score</span>
              <span className="text-white font-semibold">{formatDec(base.metrics?.f1)}</span>
            </div>
          </div>
        </div>

        {/* Fine-Tuned QLoRA Model Card */}
        <div className="card-glass p-6 space-y-4 border-purple-800/60">
          <div className="flex items-center space-x-3 border-b border-gray-800 pb-3">
            <div className="p-2 rounded-lg bg-purple-950 text-purple-400 border border-purple-800">
              <Sparkles size={20} />
            </div>
            <div>
              <h3 className="font-bold text-white text-base">Fine-Tuned QLoRA Adapter</h3>
              <p className="text-xs text-purple-400 font-mono">{finetuned.model_name || "Qwen2.5-1.5B-QLoRA"}</p>
            </div>
          </div>

          <p className="text-xs text-gray-400 leading-relaxed">
            Parameter-efficient 4-bit quantized LoRA fine-tuning trained on structured instruction prompt datasets.
          </p>

          <div className="grid grid-cols-2 gap-3 text-xs font-mono bg-purple-950/20 p-4 rounded-xl border border-purple-900/50">
            <div>
              <span className="text-gray-500 block text-[10px] uppercase">Accuracy</span>
              <span className="text-purple-300 font-bold">{formatPct(finetuned.metrics?.accuracy)}</span>
            </div>
            <div>
              <span className="text-gray-500 block text-[10px] uppercase">Precision</span>
              <span className="text-purple-300 font-bold">{formatPct(finetuned.metrics?.precision)}</span>
            </div>
            <div>
              <span className="text-gray-500 block text-[10px] uppercase">Recall</span>
              <span className="text-purple-300 font-bold">{formatPct(finetuned.metrics?.recall)}</span>
            </div>
            <div>
              <span className="text-gray-500 block text-[10px] uppercase">F1 Score</span>
              <span className="text-purple-300 font-bold">{formatDec(finetuned.metrics?.f1)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Benchmarking Insights & Technical Synthesis */}
      <div className="card-glass p-6 space-y-4">
        <div className="flex items-center space-x-2 border-b border-gray-800 pb-3">
          <GitCompare size={18} className="text-purple-400" />
          <h3 className="font-bold text-white text-base">Research Synthesis & Tradeoff Analysis</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-gray-300 leading-relaxed">
          <div className="p-4 rounded-xl bg-gray-900 border border-gray-800 space-y-2">
            <h4 className="font-semibold text-white">1. Parameter Efficiency</h4>
            <p>
              QLoRA enables adapting large language models using 4-bit NF4 quantization and low-rank adapter matrices (r=16, alpha=32) without full backpropagation across all 1.5B parameters.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-gray-900 border border-gray-800 space-y-2">
            <h4 className="font-semibold text-white">2. Tabular vs Generative LLMs</h4>
            <p>
              While fine-tuning significantly improves base zero-shot classification, specialized tabular GBDTs (CatBoost) outperform generative LLMs on structured numeric tabular datasets in both latency and ROC-AUC metrics.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-gray-900 border border-gray-800 space-y-2">
            <h4 className="font-semibold text-white">3. Role in Architecture</h4>
            <p>
              This experiment demonstrates why CatBoost is assigned as the primary decision classifier while Groq-powered LLMs are leveraged strictly in the investigation agent layer for reasoning and explanation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
