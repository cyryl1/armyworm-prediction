import React from 'react';
import { Routes, Route, NavLink, useLocation } from 'react-router-dom';

import DiagnosticHub from './pages/DiagnosticHub';
import LiveStream from './pages/LiveStream';

export default function App() {
  const location = useLocation();

  return (
    <div className="font-body-md text-on-surface bg-surface min-h-screen flex flex-col">
      {/* Top Bar */}
      <header className="fixed top-0 left-0 right-0 h-16 bg-white/5 backdrop-blur-xl border-b border-white/10 z-50 flex items-center justify-between px-lg">
        <div className="flex items-center gap-md">
          <span className="material-symbols-outlined text-primary text-3xl">eco</span>
          <h1 className="font-headline-md text-headline-md font-bold text-primary tracking-tight">Maize Guard</h1>
        </div>
        <div className="flex items-center gap-md">
          <span className="flex items-center gap-xs font-label-sm text-label-sm text-primary">
            <span className="w-2 h-2 rounded-full bg-primary status-pulse"></span>
            Online
          </span>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 pt-16 pb-20 md:pb-0 md:pt-16 min-h-screen">
        <Routes>
          <Route path="/" element={<DiagnosticHub />} />
          <Route path="/stream" element={<LiveStream />} />
        </Routes>
      </main>

      {/* Bottom Tab Bar (mobile) / Side tabs (desktop) */}
      <nav className="fixed bottom-0 left-0 right-0 md:top-16 md:bottom-auto md:left-0 md:right-auto md:w-sidebar-width bg-white/5 backdrop-blur-xl border-t md:border-t-0 md:border-r border-white/10 z-40 flex md:flex-col md:py-xl md:gap-sm md:h-[calc(100vh-64px)]">
        <TabItem
          to="/"
          icon="photo_camera"
          label="Capture"
          isActive={location.pathname === '/'}
        />
        <TabItem
          to="/stream"
          icon="videocam"
          label="Live View"
          isActive={location.pathname === '/stream'}
        />
      </nav>
    </div>
  );
}

function TabItem({ to, icon, label, isActive }) {
  return (
    <NavLink
      to={to}
      className={`flex-1 md:flex-none flex flex-col md:flex-row items-center md:items-center justify-center md:justify-start gap-xs md:gap-md py-sm md:py-md md:px-lg transition-all duration-300 ease-out relative group ${
        isActive
          ? 'text-primary'
          : 'text-on-surface-variant hover:text-on-surface'
      }`}
    >
      {/* Active indicator line */}
      {isActive && (
        <span className="absolute top-0 left-1/2 -translate-x-1/2 md:top-1/2 md:-translate-y-1/2 md:left-0 md:translate-x-0 w-12 h-1 md:w-1 md:h-8 bg-primary rounded-full transition-all" />
      )}
      <span className={`material-symbols-outlined text-2xl transition-transform duration-300 ${isActive ? 'scale-110' : 'group-hover:scale-105'}`}>{icon}</span>
      <span className={`text-xs md:text-sm font-medium ${isActive ? 'font-bold' : ''}`}>{label}</span>
    </NavLink>
  );
}
