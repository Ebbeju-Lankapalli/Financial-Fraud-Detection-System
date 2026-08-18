import React from "react";
import { Inbox } from "lucide-react";

export default function EmptyState({ title = "No Data Available", message = "No records found matching your request.", action }) {
  return (
    <div className="card-glass p-12 text-center flex flex-col items-center justify-center space-y-4">
      <div className="p-4 rounded-full bg-gray-800/80 text-gray-400 border border-gray-700">
        <Inbox size={32} />
      </div>
      <div className="space-y-1 max-w-sm">
        <h4 className="font-semibold text-gray-200">{title}</h4>
        <p className="text-sm text-gray-400">{message}</p>
      </div>
      {action && <div className="pt-2">{action}</div>}
    </div>
  );
}
