"use client"

import { useState } from "react"
import Header from "@/components/header"
import HeroSection from "@/components/hero-section"
import LiveFeedSection from "@/components/live-feed-section"
import VitalsPanel from "@/components/vitals-panel"
import MeshMetricsPanel from "@/components/mesh-metrics-panel"
import AlertsPanel from "@/components/alerts-panel"
import ReportsSection from "@/components/reports-section"
import AboutSection from "@/components/about-section"
import Footer from "@/components/footer"

export default function Home() {
  const [isDark, setIsDark] = useState(false)
  const [isMonitoring, setIsMonitoring] = useState(false)
  const [meshData, setMeshData] = useState(null)

  const toggleDarkMode = () => {
    setIsDark(!isDark)
    document.documentElement.classList.toggle("dark", !isDark)
  }

  return (
    <div className={isDark ? "dark" : ""}>
      <div className="min-h-screen bg-background text-foreground">
        <Header onToggleDarkMode={toggleDarkMode} isDark={isDark} />
        <main>
          <HeroSection />
          <LiveFeedSection onMonitoringChange={setIsMonitoring} onMeshDataChange={setMeshData} />
          <MeshMetricsPanel meshData={meshData} isMonitoring={isMonitoring} />
          <VitalsPanel />
          <AlertsPanel />
          <ReportsSection />
          <AboutSection />
        </main>
        <Footer />
      </div>
    </div>
  )
}
