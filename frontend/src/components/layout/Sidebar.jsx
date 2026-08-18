import React from "react";
import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  ShieldCheck,
  History,
  Bot,
  BarChart3,
  FlaskConical,
  Info,
  Home,
  SlidersHorizontal
} from "lucide-react";

export default function Sidebar({ mobileOpen = false, setMobileOpen = () => {} }) {
  const navItems = [
    { label: "Overview", path: "/", icon: Home },
    { label: "Dashboard", path: "/dashboard", icon: LayoutDashboard },
    { label: "Analyze Transaction", path: "/analyze", icon: ShieldCheck },
    { label: "History", path: "/history", icon: History },
    { label: "Fraud Agent", path: "/agent", icon: Bot },
    { label: "Model Evaluation", path: "/evaluation", icon: BarChart3 },
    { label: "LLM Research", path: "/research", icon: FlaskConical },
    { label: "About Model", path: "/about", icon: Info },
  ];

  return (
    <>
      {/* Mobile Drawer Overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/70 z-40 lg:hidden backdrop-blur-sm"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 w-64 bg-[#0d1322] border-r border-[#1f293d] flex flex-col justify-between transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:z-auto ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="p-5 space-y-6">
          {/* Brand Header */}
          <div className="flex items-center space-x-3 px-2">
            <div className="p-2.5 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-600 text-white shadow-lg shadow-cyan-500/20">
              <ShieldCheck size={22} />
            </div>
            <div>
              <h1 className="font-bold text-white text-base tracking-tight leading-tight">
                Fraud Detection
              </h1>
              <span className="text-[10px] font-mono uppercase tracking-widest text-cyan-400 font-semibold">
                Production System
              </span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  end={item.path === "/"}
                  onClick={() => setMobileOpen(false)}
                  className={({ isActive }) =>
                    `flex items-center space-x-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                      isActive
                        ? "bg-gradient-to-r from-cyan-500/20 to-blue-500/10 text-cyan-300 border-l-2 border-cyan-400 font-semibold"
                        : "text-gray-400 hover:text-gray-100 hover:bg-gray-800/50"
                    }`
                  }
                >
                  <Icon size={18} />
                  <span>{item.label}</span>
                </NavLink>
              );
            })}
          </nav>
        </div>

        {/* Sidebar Footer — Production Model Status Card */}
        <div className="p-4 m-3 rounded-xl bg-[#111827] border border-[#1f293d] text-xs space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-gray-400 font-medium">Production Model</span>
            <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-950 text-cyan-300 border border-cyan-800">
              Active
            </span>
          </div>
          <p className="font-semibold text-white font-mono">CatBoostClassifier</p>
          <div className="flex items-center justify-between text-[11px] text-gray-400 font-mono">
            <span>IEEE-CIS</span>
            <span>20 Features</span>
          </div>
        </div>
      </aside>
    </>
  );
}
