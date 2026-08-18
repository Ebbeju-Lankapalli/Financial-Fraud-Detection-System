import React from "react";
import { Link } from "react-router-dom";
import {
  ShieldCheck,
  LayoutDashboard,
  ArrowRight,
  Cpu,
  Database,
  Bot,
  FlaskConical,
  Zap,
  CheckCircle2,
  AlertTriangle
} from "lucide-react";

export default function Home() {
  return (
    <div className="space-y-12 py-4">
      {/* Hero Section */}
      <section className="relative overflow-hidden card-glass p-8 md:p-12 border-cyan-900/40 bg-gradient-to-br from-[#0d1527] via-[#0f172a] to-[#131d35]">
        <div className="absolute top-0 right-0 -mt-12 -mr-12 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/3 -mb-12 w-64 h-64 bg-blue-600/10 rounded-full blur-2xl pointer-events-none" />

        <div className="max-w-3xl space-y-6 relative z-10">
          <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-cyan-950/80 text-cyan-300 border border-cyan-800/80 text-xs font-mono">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span>Active Model: CatBoost (IEEE-CIS 20 Features)</span>
          </div>

          <h1 className="text-3xl md:text-5xl font-extrabold text-white tracking-tight leading-tight">
            Financial Fraud Detection System
          </h1>

          <p className="text-lg text-gray-300 leading-relaxed">
            Production fraud screening powered by CatBoost with AI-assisted investigation. Evaluates 20 IEEE-CIS transactional features against a strict 0.83 validation decision threshold with complete audit persistence.
          </p>

          <div className="flex flex-wrap gap-4 pt-2">
            <Link to="/analyze" className="btn-primary text-sm px-6 py-3">
              <ShieldCheck size={18} />
              <span>Analyze Transaction</span>
              <ArrowRight size={16} />
            </Link>

            <Link to="/dashboard" className="btn-secondary text-sm px-6 py-3">
              <LayoutDashboard size={18} />
              <span>View Dashboard</span>
            </Link>
          </div>
        </div>
      </section>

      {/* Production Pipeline Architecture */}
      <section className="space-y-6">
        <div className="space-y-1">
          <h2 className="text-xl font-bold text-white tracking-tight">System Architecture</h2>
          <p className="text-sm text-gray-400">End-to-end production flow from transaction submission to AI explanation</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="card-glass p-5 space-y-3 relative">
            <div className="p-2.5 rounded-lg bg-blue-950 text-blue-400 border border-blue-800 w-fit">
              <Database size={20} />
            </div>
            <h3 className="font-semibold text-white text-sm">1. Transaction Input</h3>
            <p className="text-xs text-gray-400 leading-relaxed">
              20 IEEE-CIS features including card descriptors, transaction amounts, and anonymized signals.
            </p>
          </div>

          <div className="card-glass p-5 space-y-3 relative">
            <div className="p-2.5 rounded-lg bg-cyan-950 text-cyan-400 border border-cyan-800 w-fit">
              <Cpu size={20} />
            </div>
            <h3 className="font-semibold text-white text-sm">2. CatBoost Classifier</h3>
            <p className="text-xs text-gray-400 leading-relaxed">
              Active production model yielding calibrated fraud probability against 0.83 decision threshold.
            </p>
          </div>

          <div className="card-glass p-5 space-y-3 relative">
            <div className="p-2.5 rounded-lg bg-amber-950 text-amber-400 border border-amber-800 w-fit">
              <Zap size={20} />
            </div>
            <h3 className="font-semibold text-white text-sm">3. Risk Classification</h3>
            <p className="text-xs text-gray-400 leading-relaxed">
              Categorizes transactions into High Risk (≥ 0.83) or Low Risk with exact percentage probabilities.
            </p>
          </div>

          <div className="card-glass p-5 space-y-3 relative">
            <div className="p-2.5 rounded-lg bg-emerald-950 text-emerald-400 border border-emerald-800 w-fit">
              <CheckCircle2 size={20} />
            </div>
            <h3 className="font-semibold text-white text-sm">4. Audit Persistence</h3>
            <p className="text-xs text-gray-400 leading-relaxed">
              Stores transaction inputs and prediction results immutably in SQLite database with analysis IDs.
            </p>
          </div>

          <div className="card-glass p-5 space-y-3 relative">
            <div className="p-2.5 rounded-lg bg-purple-950 text-purple-400 border border-purple-800 w-fit">
              <Bot size={20} />
            </div>
            <h3 className="font-semibold text-white text-sm">5. AI Agent Layer</h3>
            <p className="text-xs text-gray-400 leading-relaxed">
              Grounded Groq LLM agent generates human-readable explanations and action recommendations.
            </p>
          </div>
        </div>
      </section>

      {/* Model Distinction Section: Production vs Research */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card-glass p-6 border-cyan-800/60 space-y-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-cyan-950 text-cyan-400 border border-cyan-800">
              <Cpu size={20} />
            </div>
            <div>
              <h3 className="font-bold text-white">Active Production Decision Source</h3>
              <p className="text-xs text-cyan-400 font-mono">catboost_ieee_cis</p>
            </div>
          </div>

          <p className="text-xs text-gray-300 leading-relaxed">
            The active production classifier is a reduced 20-feature CatBoost model trained on the IEEE-CIS Fraud Detection dataset. It handles all real-time fraud scoring and API analyses.
          </p>

          <ul className="text-xs text-gray-400 space-y-2">
            <li className="flex items-center space-x-2">
              <CheckCircle2 size={14} className="text-cyan-400 shrink-0" />
              <span>20 IEEE-CIS features for fast execution</span>
            </li>
            <li className="flex items-center space-x-2">
              <CheckCircle2 size={14} className="text-cyan-400 shrink-0" />
              <span>Decision threshold selected on validation set (0.83)</span>
            </li>
            <li className="flex items-center space-x-2">
              <CheckCircle2 size={14} className="text-cyan-400 shrink-0" />
              <span>Chronological test set strictly held out</span>
            </li>
          </ul>

          <div className="pt-2">
            <Link to="/evaluation" className="text-xs text-cyan-400 hover:text-cyan-300 font-semibold inline-flex items-center space-x-1">
              <span>View Production Evaluation Metrics</span>
              <ArrowRight size={14} />
            </Link>
          </div>
        </div>

        <div className="card-glass p-6 border-purple-800/60 space-y-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-purple-950 text-purple-400 border border-purple-800">
              <FlaskConical size={20} />
            </div>
            <div>
              <h3 className="font-bold text-white">LLM Fine-Tuning Experiment</h3>
              <p className="text-xs text-purple-400 font-mono">Qwen2.5-1.5B + QLoRA</p>
            </div>
          </div>

          <p className="text-xs text-gray-300 leading-relaxed">
            Preserved for research transparency. Explores parameter-efficient fine-tuning (QLoRA) of LLMs for financial fraud classification on synthetic/PaySim data.
          </p>

          <ul className="text-xs text-gray-400 space-y-2">
            <li className="flex items-center space-x-2">
              <AlertTriangle size={14} className="text-purple-400 shrink-0" />
              <span>Research track — NOT the production classifier</span>
            </li>
            <li className="flex items-center space-x-2">
              <CheckCircle2 size={14} className="text-purple-400 shrink-0" />
              <span>Demonstrates PEFT, LoRA adapters & instruction tuning</span>
            </li>
            <li className="flex items-center space-x-2">
              <CheckCircle2 size={14} className="text-purple-400 shrink-0" />
              <span>Base vs Fine-tuned performance benchmarking</span>
            </li>
          </ul>

          <div className="pt-2">
            <Link to="/research" className="text-xs text-purple-400 hover:text-purple-300 font-semibold inline-flex items-center space-x-1">
              <span>Explore Research Experiment</span>
              <ArrowRight size={14} />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
