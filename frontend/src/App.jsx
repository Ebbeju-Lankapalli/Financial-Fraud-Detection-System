import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import Layout from "./components/layout/Layout";
import Home from "./pages/Home";
import Dashboard from "./pages/Dashboard";
import AnalyzeTransaction from "./pages/AnalyzeTransaction";
import History from "./pages/History";
import TransactionDetail from "./pages/TransactionDetail";
import FraudAgent from "./pages/FraudAgent";
import ModelEvaluation from "./pages/ModelEvaluation";
import Research from "./pages/Research";
import AboutModel from "./pages/AboutModel";
import NotFound from "./pages/NotFound";

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/analyze" element={<AnalyzeTransaction />} />
          <Route path="/history" element={<History />} />
          <Route path="/history/:analysisId" element={<TransactionDetail />} />
          <Route path="/agent" element={<FraudAgent />} />
          <Route path="/evaluation" element={<ModelEvaluation />} />
          <Route path="/research" element={<Research />} />
          <Route path="/about" element={<AboutModel />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
