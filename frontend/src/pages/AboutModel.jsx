import React from "react";
import {
  ShieldCheck,
  Cpu,
  Bot,
  FlaskConical,
  AlertOctagon,
  CheckCircle2,
  Lock,
  Layers,
  Database
} from "lucide-react";

export default function AboutModel() {
  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="border-b border-gray-800 pb-4 space-y-1">
        <h1 className="text-2xl font-bold text-white tracking-tight">About System & Architecture</h1>
        <p className="text-sm text-gray-400">Detailed technical documentation of model pipelines, AI agent grounding, and safety boundaries</p>
      </div>

      {/* Section 1: Production Fraud Detection */}
      <section className="card-glass p-6 space-y-4 border-cyan-800/60">
        <div className="flex items-center space-x-3 text-cyan-400 border-b border-gray-800 pb-3">
          <Cpu size={22} />
          <div>
            <h2 className="font-bold text-white text-lg">1. Production Fraud Classifier</h2>
            <span className="text-xs font-mono text-cyan-400">catboost_ieee_cis</span>
          </div>
        </div>

        <div className="space-y-3 text-xs text-gray-300 leading-relaxed">
          <p>
            The core production decision engine uses a reduced <strong>20-feature CatBoostClassifier</strong> trained on the benchmark <strong>IEEE-CIS Fraud Detection dataset</strong>.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 font-mono">
            <div className="p-3 rounded-lg bg-gray-900 border border-gray-800 space-y-1">
              <span className="text-gray-500 text-[10px] uppercase block">Primary Features</span>
              <span className="text-white font-semibold">20 IEEE-CIS Attributes</span>
            </div>
            <div className="p-3 rounded-lg bg-gray-900 border border-gray-800 space-y-1">
              <span className="text-gray-500 text-[10px] uppercase block">Decision Threshold</span>
              <span className="text-amber-400 font-bold">0.83 (Validation Optimized)</span>
            </div>
            <div className="p-3 rounded-lg bg-gray-900 border border-gray-800 space-y-1">
              <span className="text-gray-500 text-[10px] uppercase block">Evaluation Split</span>
              <span className="text-white">Chronological Holdout Test</span>
            </div>
            <div className="p-3 rounded-lg bg-gray-900 border border-gray-800 space-y-1">
              <span className="text-gray-500 text-[10px] uppercase block">Audit Persistence</span>
              <span className="text-emerald-400 font-semibold">SQLite Audit Log</span>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: AI Investigation Agent */}
      <section className="card-glass p-6 space-y-4 border-purple-800/60">
        <div className="flex items-center space-x-3 text-purple-400 border-b border-gray-800 pb-3">
          <Bot size={22} />
          <div>
            <h2 className="font-bold text-white text-lg">2. AI Investigation Agent Layer</h2>
            <span className="text-xs font-mono text-purple-400">Grounded LLM Reasoning</span>
          </div>
        </div>

        <div className="space-y-3 text-xs text-gray-300 leading-relaxed">
          <p>
            The fraud agent layer uses Groq-accelerated LLMs to generate natural language explanations and actionable forensic recommendations for fraud analysts.
          </p>

          <div className="p-4 rounded-xl bg-purple-950/20 border border-purple-900/40 text-purple-200 font-sans space-y-2">
            <div className="flex items-center space-x-2 text-purple-300 font-semibold">
              <Lock size={16} />
              <span>Strict Non-Overriding Policy</span>
            </div>
            <p className="text-xs text-gray-300 leading-relaxed">
              The AI Agent is strictly grounded on the output of the CatBoost classifier. It <strong>does NOT</strong> override or alter the risk score or fraud probability calculated by the production CatBoost model. If external LLM APIs are unreachable, a deterministic logic fallback automatically handles the explanation without breaking.
            </p>
          </div>
        </div>
      </section>

      {/* Section 3: LLM Fine-Tuning Research Track */}
      <section className="card-glass p-6 space-y-4 border-purple-900/40">
        <div className="flex items-center space-x-3 text-purple-300 border-b border-gray-800 pb-3">
          <FlaskConical size={22} />
          <div>
            <h2 className="font-bold text-white text-lg">3. LLM Fine-Tuning Research Track</h2>
            <span className="text-xs font-mono text-purple-400">Qwen2.5-1.5B + QLoRA</span>
          </div>
        </div>

        <div className="space-y-3 text-xs text-gray-300 leading-relaxed">
          <p>
            The project preserves an experimental track investigating 4-bit QLoRA fine-tuning on Qwen2.5-1.5B-Instruct using PaySim/synthetic transaction datasets.
          </p>
          <p className="text-gray-400">
            This track demonstrates parameter-efficient adapter tuning (PEFT), instruction prompt formatting, and comparative evaluation against traditional GBDT models.
          </p>
        </div>
      </section>

      {/* Section 4: Safety & Limitations */}
      <section className="card-glass p-6 space-y-4 border-amber-900/50">
        <div className="flex items-center space-x-3 text-amber-400 border-b border-gray-800 pb-3">
          <AlertOctagon size={22} />
          <div>
            <h2 className="font-bold text-white text-lg">4. Safety & System Limitations</h2>
            <span className="text-xs font-mono text-amber-400">Responsible AI Guidelines</span>
          </div>
        </div>

        <div className="space-y-3 text-xs text-gray-300 leading-relaxed">
          <ul className="space-y-2.5">
            <li className="flex items-start space-x-2.5">
              <CheckCircle2 size={16} className="text-amber-400 shrink-0 mt-0.5" />
              <span>
                <strong>Screening Signal:</strong> Fraud probability is a statistical screening signal derived from historical IEEE-CIS data patterns. It is not proof of fraud or criminal intent.
              </span>
            </li>
            <li className="flex items-start space-x-2.5">
              <CheckCircle2 size={16} className="text-amber-400 shrink-0 mt-0.5" />
              <span>
                <strong>Human-in-the-loop:</strong> All consequential decisions (e.g., account suspension or transaction reversal) require review by qualified fraud analysts.
              </span>
            </li>
            <li className="flex items-start space-x-2.5">
              <CheckCircle2 size={16} className="text-amber-400 shrink-0 mt-0.5" />
              <span>
                <strong>Dataset Anonymization:</strong> Feature names (C1-C14, D1-D15, M1-M9) are anonymized in accordance with IEEE-CIS dataset specifications to protect private financial attributes.
              </span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}
