import React from 'react';
import { Routes, Route } from 'react-router-dom';
import MainPage from './pages/MainPage';
import ClassificationViz from './pages/ClassificationViz';
import AttentionViz from './pages/AttentionViz';
import GcnViz from './pages/GcnViz';
import VizLayout from './components/VizLayout';
import './App.css';

const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<MainPage />} />
      <Route element={<VizLayout />}>
        <Route path="/classification" element={<ClassificationViz />} />
        <Route path="/attention" element={<AttentionViz />} />
        <Route path="/gcn" element={<GcnViz />} />
      </Route>
    </Routes>
  );
};

export default App;
