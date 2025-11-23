"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"

interface MeshData {
  detected: boolean
  status: string
  current_mesh: number
  baseline_mesh: number | null
  difference: number
  metrics: {
    cheek_dist: number
    jaw_dist: number
    eye_ratio: number
    mouth_ratio: number
  }
  danger_frames: number
  is_danger: boolean
  is_final_alert: boolean
}

interface MeshMetricsPanelProps {
  meshData: MeshData | null
  isMonitoring: boolean
}

export default function MeshMetricsPanel({ meshData, isMonitoring }: MeshMetricsPanelProps) {
  if (!isMonitoring || !meshData) {
    return null
  }

  // Determine status color based on mesh status
  const getStatusBadgeColor = (status: string) => {
    if (meshData.is_final_alert) {
      return "bg-red-500/10 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800"
    }
    if (meshData.is_danger) {
      return "bg-orange-500/10 text-orange-700 dark:text-orange-400 border-orange-200 dark:border-orange-800"
    }
    switch (status.toLowerCase()) {
      case "stable":
        return "bg-green-500/10 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800"
      case "calibrating...":
        return "bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-800"
      case "effort (normal gym)":
        return "bg-cyan-500/10 text-cyan-700 dark:text-cyan-400 border-cyan-200 dark:border-cyan-800"
      case "risk (warning)":
        return "bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-800"
      default:
        return "bg-gray-500/10 text-gray-700 dark:text-gray-400 border-gray-200 dark:border-gray-800"
    }
  }

  // Prepare chart data
  const chartData = [
    {
      name: "Cheek Distance",
      value: Math.round(meshData.metrics.cheek_dist),
    },
    {
      name: "Jaw Distance",
      value: Math.round(meshData.metrics.jaw_dist),
    },
    {
      name: "Eye Ratio",
      value: Math.round(meshData.metrics.eye_ratio),
    },
    {
      name: "Mouth Ratio",
      value: Math.round(meshData.metrics.mouth_ratio),
    },
  ]

  // Calculate danger duration
  const dangerDuration = meshData.danger_frames > 0 ? (meshData.danger_frames / 30).toFixed(1) : "0"

  return (
    <section className="py-8 md:py-12 bg-background">
      <div className="max-w-6xl mx-auto px-4 md:px-6">
        <div className="space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl md:text-3xl font-bold text-foreground">Facial Mesh Analysis</h2>
              <p className="text-muted-foreground mt-1">Real-time facial strain & stress detection</p>
            </div>
          </div>

          {/* Status Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Current Mesh Score */}
            <Card className="border border-border shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">Current Mesh Score</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-foreground">{meshData.current_mesh.toFixed(2)}</div>
                <p className="text-xs text-muted-foreground mt-2">
                  Combined facial metric
                </p>
              </CardContent>
            </Card>

            {/* Baseline Mesh Score */}
            <Card className="border border-border shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">Baseline Score</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-foreground">
                  {meshData.baseline_mesh !== null ? meshData.baseline_mesh.toFixed(2) : "--"}
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Reference baseline
                </p>
              </CardContent>
            </Card>

            {/* Difference */}
            <Card className="border border-border shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">Difference</CardTitle>
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-bold ${Math.abs(meshData.difference) < 4 ? "text-green-600" : "text-red-600"}`}>
                  {meshData.difference.toFixed(2)}
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Deviation from baseline
                </p>
              </CardContent>
            </Card>

            {/* Facial Status */}
            <Card className="border border-border shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">Facial Status</CardTitle>
              </CardHeader>
              <CardContent>
                <Badge className={getStatusBadgeColor(meshData.status)}>
                  {meshData.status}
                </Badge>
                {meshData.danger_frames > 0 && (
                  <p className="text-xs text-red-600 dark:text-red-400 mt-2">
                    Danger: {dangerDuration}s
                  </p>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Metrics Visualization */}
          <Card className="border border-border shadow-sm">
            <CardHeader>
              <CardTitle>Facial Metrics Breakdown</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: "rgba(0, 0, 0, 0.8)",
                      border: "1px solid rgba(255, 255, 255, 0.2)",
                      borderRadius: "8px",
                      color: "#fff"
                    }}
                  />
                  <Bar dataKey="value" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Detailed Metrics */}
          <Card className="border border-border shadow-sm">
            <CardHeader>
              <CardTitle>Detailed Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Cheek Distance */}
                <div className="p-4 bg-muted rounded-lg">
                  <p className="text-sm font-medium text-muted-foreground">Cheek-to-Cheek Distance</p>
                  <p className="text-xl font-bold mt-2">{meshData.metrics.cheek_dist.toFixed(2)} px</p>
                  <p className="text-xs text-muted-foreground mt-1">Indicates facial tension</p>
                </div>

                {/* Jaw Distance */}
                <div className="p-4 bg-muted rounded-lg">
                  <p className="text-sm font-medium text-muted-foreground">Jaw Width</p>
                  <p className="text-xl font-bold mt-2">{meshData.metrics.jaw_dist.toFixed(2)} px</p>
                  <p className="text-xs text-muted-foreground mt-1">Jaw clenching indicator</p>
                </div>

                {/* Eye Ratio */}
                <div className="p-4 bg-muted rounded-lg">
                  <p className="text-sm font-medium text-muted-foreground">Eye Opening Ratio</p>
                  <p className="text-xl font-bold mt-2">{meshData.metrics.eye_ratio.toFixed(2)}%</p>
                  <p className="text-xs text-muted-foreground mt-1">High = eyes open, Low = fatigue</p>
                </div>

                {/* Mouth Ratio */}
                <div className="p-4 bg-muted rounded-lg">
                  <p className="text-sm font-medium text-muted-foreground">Mouth Opening Ratio</p>
                  <p className="text-xl font-bold mt-2">{meshData.metrics.mouth_ratio.toFixed(2)}%</p>
                  <p className="text-xs text-muted-foreground mt-1">Stress/strain indicator</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Risk Thresholds Information */}
          <Card className="border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-950/30 shadow-sm">
            <CardHeader>
              <CardTitle className="text-blue-900 dark:text-blue-100">Risk Assessment Levels</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-2 bg-green-50 dark:bg-green-950/30 rounded">
                  <span className="text-sm text-green-900 dark:text-green-100">Stable (Difference &lt; 4)</span>
                  <Badge className="bg-green-500/10 text-green-700 dark:text-green-400">✓</Badge>
                </div>
                <div className="flex items-center justify-between p-2 bg-cyan-50 dark:bg-cyan-950/30 rounded">
                  <span className="text-sm text-cyan-900 dark:text-cyan-100">Effort - Normal Gym (4-10)</span>
                  <Badge className="bg-cyan-500/10 text-cyan-700 dark:text-cyan-400">●</Badge>
                </div>
                <div className="flex items-center justify-between p-2 bg-amber-50 dark:bg-amber-950/30 rounded">
                  <span className="text-sm text-amber-900 dark:text-amber-100">Risk - Warning (10-15)</span>
                  <Badge className="bg-amber-500/10 text-amber-700 dark:text-amber-400">⚠</Badge>
                </div>
                <div className="flex items-center justify-between p-2 bg-red-50 dark:bg-red-950/30 rounded">
                  <span className="text-sm text-red-900 dark:text-red-100">Danger (≥ 15)</span>
                  <Badge className="bg-red-500/10 text-red-700 dark:text-red-400">⚠⚠</Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Critical Alert */}
          {meshData.is_final_alert && (
            <div className="p-6 bg-red-50 dark:bg-red-950/30 border-2 border-red-500 rounded-lg">
              <p className="text-lg font-bold text-red-700 dark:text-red-400 animate-pulse">
                🚨 FINAL ALERT: Critical facial strain detected! Please seek medical attention!
              </p>
            </div>
          )}

          {meshData.is_danger && !meshData.is_final_alert && (
            <div className="p-4 bg-orange-50 dark:bg-orange-950/30 border border-orange-500 rounded-lg">
              <p className="text-base font-semibold text-orange-700 dark:text-orange-400">
                ⚠️ WARNING: Dangerous facial strain detected ({dangerDuration}s)
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
