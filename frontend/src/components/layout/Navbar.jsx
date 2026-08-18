import React from "react";
import { Menu, ShieldAlert, Cpu } from "lucide-react";

export default function Navbar({ setMobileOpen }) {
  return (
    <header className="sticky top-0 z-30 bg-[#0d1322]/90 backdrop-blur-md border-b border-[#1f293d] px-4 py-3 lg:px-8">
      <div className="flex items-center justify-between">
        {/* Left: Mobile Toggle & Title */}
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setMobileOpen(true)}
            className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 lg:hidden focus:outline-none"
            aria-label="Open navigation menu"
          >
            <Menu size={22} />
          </button>
          <div className="lg:hidden flex items-center space-x-2">
            <span className="font-bold text-white text-sm">Financial Fraud System</span>
          </div>
        </div>

        {/* Right: Live Status Pills */}
        <div className="flex items-center space-x-3 text-xs font-mono">
          <div className="hidden sm:flex items-center space-x-2 px-3 py-1.5 rounded-full bg-emerald-950/60 text-emerald-400 border border-emerald-800/60">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>API Online</span>
          </div>

          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-cyan-950/60 text-cyan-300 border border-cyan-800/60">
            <Cpu size={14} className="text-cyan-400" />
            <span>catboost_ieee_cis</span>
          </div>
        </div>
      </div>
    </header>
  );
}
