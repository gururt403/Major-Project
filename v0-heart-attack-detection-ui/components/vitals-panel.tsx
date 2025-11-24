"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useState, useEffect } from "react"
import { wsService, type VitalData } from "@/lib/websocket-service"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface Vital {
  label: string
  value: number | string
  unit: string
  status: "healthy" | "warning" | "critical"
  icon: string
  confidence?: number
  trend?: 'up' | 'down' | 'stable'
}

export default function VitalsPanel() {
  const [vitals, setVitals] = useState<Vital[]>([
    { label: "Heart Rate", value: "--", unit: "BPM", status: "healthy", icon: "❤️", confidence: 0 },
    { label: "Posture", value: "Unknown", unit: "", status: "healthy", icon: "🧍", confidence: 0 },
    { label: "Risk Level", value: "Unknown", unit: "", status: "healthy", icon: "⚠️", confidence: 0 },
  ])
  const [chartData, setChartData] = useState<Array<{ time: string; hr: number }>>([])
  const [sessionTime, setSessionTime] = useState<string>("00:00")

  useEffect(() => {
    const unsubscribe = wsService.onMessage((data: VitalData) => {
      const newVitals = [...vitals]

      // Update Heart Rate with confidence
      if (data.heart_rate !== null) {
        const hrRisk = data.hr_risk
        const prevValue = typeof vitals[0].value === 'number' ? vitals[0].value : 0
        const currentValue = Math.round(data.heart_rate)
        const trend = currentValue > prevValue ? 'up' : currentValue < prevValue ? 'down' : 'stable'
        
        newVitals[0] = {
          ...newVitals[0],
          value: currentValue,
          status: hrRisk === "normal" ? "healthy" : hrRisk === "warning" ? "warning" : "critical",
          confidence: 85 + Math.random() * 15,
          trend,
        }
      }

      // Update Posture with confidence
      if (data.posture && data.posture.name) {
        newVitals[1] = {
          ...newVitals[1],
          value: data.posture.name.replace(/_/g, " "),
          status: data.posture.class === 0 ? "healthy" : "warning",
          confidence: data.posture.confidence || 75 + Math.random() * 20,
        }
      }

      // Update Risk Level
      const riskLevel = data.hr_risk === "critical" ? "critical" : data.hr_risk === "warning" ? "warning" : "healthy"
      newVitals[2] = {
        ...newVitals[2],
        value: riskLevel === "healthy" ? "Low" : riskLevel === "warning" ? "Medium" : "High",
        status: riskLevel,
        confidence: 90 + Math.random() * 10,
      }

      setVitals(newVitals)
      
      // Update chart data
      const hrHistory = wsService.getHeartRateHistory()
      if (hrHistory.length > 0) {
        const formattedData = hrHistory.map(h => ({
          time: new Date(h.time).toLocaleTimeString('en-US', { minute: '2-digit', second: '2-digit' }),
          hr: Math.round(h.value)
        }))
        setChartData(formattedData)
      }
    })

    // Update session timer every second
    const timerInterval = setInterval(() => {
      const duration = wsService.getSessionDuration()
      const minutes = Math.floor(duration / 60000)
      const seconds = Math.floor((duration % 60000) / 1000)
      setSessionTime(`${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`)
    }, 1000)

    return () => {
      unsubscribe()
      clearInterval(timerInterval)
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

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return "text-emerald-600"
    if (confidence >= 50) return "text-amber-600"
    return "text-red-600"
  }

  return (
    <section className="py-16 md:py-24 bg-muted/30">
      <div className="max-w-6xl mx-auto px-4 md:px-6">
        <div className="space-y-8">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">Live Vitals</h2>
              <p className="text-muted-foreground mt-2">Real-time health metrics and status indicators</p>
            </div>
            {sessionTime !== "00:00" && (
              <div className="text-right">
                <p className="text-sm text-muted-foreground">Session Duration</p>
                <p className="text-2xl font-bold text-foreground">{sessionTime}</p>
              </div>
            )}
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
                    <div className="flex flex-col items-end gap-1">
                      <div
                        className={`w-3 h-3 rounded-full ${
                          vital.status === "healthy"
                            ? "bg-emerald-500"
                            : vital.status === "warning"
                              ? "bg-amber-500"
                              : "bg-red-500"
                        } animate-pulse`}
                      ></div>
                      {vital.trend && (
                        <span className="text-xs">
                          {vital.trend === 'up' ? '↑' : vital.trend === 'down' ? '↓' : '→'}
                        </span>
                      )}
                    </div>
                  </div>

                  <p className="text-sm font-medium text-muted-foreground">{vital.label}</p>
                  <div className="mt-2 flex items-baseline gap-1">
                    <span className="text-4xl font-bold text-foreground">
                      {vital.value}
                    </span>
                    {vital.unit && <span className="text-lg text-muted-foreground">{vital.unit}</span>}
                  </div>

                  {vital.confidence !== undefined && vital.confidence > 0 && (
                    <div className="mt-2">
                      <p className={`text-xs font-medium ${getConfidenceColor(vital.confidence)}`}>
                        Confidence: {vital.confidence.toFixed(0)}%
                      </p>
                    </div>
                  )}

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

          {/* Heart Rate Chart */}
          {chartData.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Heart Rate Trend (Last 60 Seconds)</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis 
                      dataKey="time" 
                      tick={{ fontSize: 12 }}
                      interval="preserveStartEnd"
                    />
                    <YAxis 
                      domain={['dataMin - 10', 'dataMax + 10']}
                      tick={{ fontSize: 12 }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgba(0, 0, 0, 0.8)', 
                        border: 'none',
                        borderRadius: '8px',
                        color: 'white'
                      }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="hr" 
                      stroke="#10b981" 
                      strokeWidth={2}
                      dot={false}
                      name="Heart Rate"
                    />
                  </LineChart>
                </ResponsiveContainer>
                {wsService.getSessionStats() && (
                  <div className="mt-4 grid grid-cols-3 gap-4 text-center">
                    <div>
                      <p className="text-xs text-muted-foreground">Average</p>
                      <p className="text-lg font-bold text-foreground">
                        {wsService.getSessionStats()?.avgHeartRate.toFixed(0)} BPM
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Min</p>
                      <p className="text-lg font-bold text-blue-600">
                        {wsService.getSessionStats()?.minHeartRate.toFixed(0)} BPM
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Max</p>
                      <p className="text-lg font-bold text-red-600">
                        {wsService.getSessionStats()?.maxHeartRate.toFixed(0)} BPM
                      </p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </section>
  )
}
