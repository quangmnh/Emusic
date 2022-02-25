import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import Fer from "./pages/Fer";
import Dashboard from "./pages/Dashboard";

import { useState } from "react";
import logo from "./logo.svg";

function App() {
  return (
    <>
      <Router>
        <div className="container">
          <Routes>
            <Route path="/fer" element={<Fer />} />
            <Route path="/" element={<Dashboard />} />
          </Routes>
        </div>
      </Router>
      <ToastContainer />
    </>
  );
}

export default App;
