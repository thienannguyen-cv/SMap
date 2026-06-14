import React from "react";
import { createRoot } from "react-dom/client";
import SMapSimulator from "./SMapSimulator.jsx";

createRoot(document.getElementById("root")).render(
  React.createElement(React.StrictMode, null, React.createElement(SMapSimulator))
);
