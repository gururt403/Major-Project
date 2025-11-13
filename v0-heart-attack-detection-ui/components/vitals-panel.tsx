"use client"

import { Card, CardContent } from "@/components/ui/card"
import { useState, useEffect } from "react"
import { wsService, type VitalData } from "@/lib/websocket-service"

interface Vital {
  label: string
  value: number | string
  unit: string
  status: "healthy" | "warning" | "critical"
  icon: string
}

export default function VitalsPanel() {
  const [vitals, setVitals] = useState<Vital[]>([
    { label: "Heart Rate", value: "--", unit: "BPM", status: "healthy", icon: "❤️" },
    { label: "Posture", value: "Unknown", unit: "", status: "healthy", icon: "🧍" },
    { label: "Risk Level", value: "Unknown", unit: "", status: "healthy", icon: "⚠️" },
  ])

  useEffect(() => {
    const unsubscribe = wsService.onMessage((data: VitalData) => {
      const newVitals = [...vitals]

      // Update Heart Rate
      if (data.heart_rate !== null) {
        const hrRisk = data.hr_risk
        newVitals[0] = {
          ...newVitals[0],
          value: Math.round(data.heart_rate),
          status: hrRisk === "normal" ? "healthy" : hrRisk === "warning" ? "warning" : "critical",
        }
      }

      // Update Posture
      if (data.posture && data.posture.name) {
        newVitals[1] = {
          ...newVitals[1],
          value: data.posture.name.replace(/_/g, " "),
          status: data.posture.class === 0 ? "healthy" : "warning",
        }
      }

      // Update Risk Level
      const riskLevel = data.hr_risk === "critical" ? "critical" : data.hr_risk === "warning" ? "warning" : "healthy"
      newVitals[2] = {
        ...newVitals[2],
        value: riskLevel === "healthy" ? "Low" : riskLevel === "warning" ? "Medium" : "High",
        status: riskLevel,
      }

      setVitals(newVitals)
    })

    return () => {
      unsubscribe()
    }
  }, [vitals])

  const getStatusColor = (status: string) => {
    switch (status) {
      case "healthy":
        return "border-emerald-200 dark:border-emerald-900 bg-emerald-50 dark:bg-emerald-950"
      case "warning":
        return "border-amber-200 dark:border-amber-900 bg-amber-50 dark:bg-amber-950"
      case "critical":
        return "border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950"
      default:
        return "border-border bg-card"
    }
  }

  const getStatusPercentage = (vital: Vital, index: number) => {
    if (index === 0 && typeof vital.value === "number") {
      // Heart rate: scale from 40-180 BPM
      return Math.min(100, Math.max(0, ((vital.value - 40) / 140) * 100))
    }
    if (index === 1) {
      // Posture: always 50% for visual balance
      return 50
    }
    if (index === 2) {
      // Risk level
      return vital.status === "healthy" ? 20 : vital.status === "warning" ? 50 : 100
    }
    return 50
  }

  return (
    <section className="py-16 md:py-24 bg-muted/30">
      <div className="max-w-6xl mx-auto px-4 md:px-6">
        <div className="space-y-8">
          <div>
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">Live Vitals</h2>
            <p className="text-muted-foreground mt-2">Real-time health metrics and status indicators</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {vitals.map((vital, index) => (
              <Card
                key={vital.label}
                className={`border-2 ${getStatusColor(vital.status)} overflow-hidden transition-all duration-300 hover:shadow-lg`}
              >
                <CardContent className="pt-6">
                  <div className="flex items-start justify-between mb-4">
                    <span className="text-3xl">{vital.icon}</span>
                    <div
                      className={`w-3 h-3 rounded-full ${
                        vital.status === "healthy"
                          ? "bg-emerald-500"
                          : vital.status === "warning"
                            ? "bg-amber-500"
                            : "bg-red-500"
                      } animate-pulse`}
                    ></div>
                  </div>

                  <p className="text-sm font-medium text-muted-foreground">{vital.label}</p>
                  <div className="mt-2 flex items-baseline gap-1">
                    <span className="text-4xl font-bold text-foreground">
                      {vital.value}
                    </span>
                    {vital.unit && <span className="text-lg text-muted-foreground">{vital.unit}</span>}
                  </div>

                  <div className="mt-4 w-full h-1 bg-background rounded-full overflow-hidden">
                    <div
                      className={`h-full ${
                        vital.status === "healthy"
                          ? "bg-emerald-500"
                          : vital.status === "warning"
                            ? "bg-amber-500"
                            : "bg-red-500"
                      } rounded-full transition-all duration-500`}
                      style={{ width: `${getStatusPercentage(vital, index)}%` }}
                    ></div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
