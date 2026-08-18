import React from "react";
import { Loader2 } from "lucide-react";

export default function LoadingSpinner({ message = "Loading data from backend..." }) {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center space-y-4">
      <Loader2 size={36} className="text-cyan-400 animate-spin" />
      <p className="text-sm font-medium text-gray-400">{message}</p>
    </div>
  );
}
