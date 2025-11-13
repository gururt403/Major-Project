"use client"

import { Card, CardContent } from "@/components/ui/card"
import { AlertTriangle, CheckCircle, AlertCircle, Zap } from "lucide-react"
import { useState, useEffect } from "react"
import { wsService, type VitalData } from "@/lib/websocket-service"

interface Alert {
  id: string
  type: "warning" | "normal" | "critical" | "emergency"
  title: string
  message: string
  time: string
}

export default function AlertsPanel() {
  const [alerts, setAlerts] = useState<Alert[]>([
    {
      id: "initial",
      type: "normal",
      title: "✅ System Ready",
      message: "Awaiting connection to backend service",
      time: "Just now",
    },
  ])
  const [lastAlertMessage, setLastAlertMessage] = useState<string>("")
  const [alertCooldown, setAlertCooldown] = useState(false)

  useEffect(() => {
    const unsubscribe = wsService.onMessage((data: VitalData) => {
      // Only add alert if it's different from the last one and not "none"
      if (data.alert && data.alert.level !== "none" && data.alert.message !== lastAlertMessage) {
        const newAlert: Alert = {
          id: `${Date.now()}-${Math.random()}`,
          type: data.alert.level as any,
          title: data.alert.message.split(":")[0] || "Alert",
          message: data.alert.message,
          time: "Just now",
        }

        setAlerts((prev) => [newAlert, ...prev].slice(0, 10))
        setLastAlertMessage(data.alert.message)
        
        // Set cooldown to prevent duplicate alerts within 2 seconds
        setAlertCooldown(true)
        setTimeout(() => setAlertCooldown(false), 2000)
      }
    })

    return () => {
      unsubscribe()
    }
  }, [lastAlertMessage])

  const getAlertIcon = (type: string) => {
    switch (type) {
      case "emergency":
        return <Zap className="w-6 h-6 text-destructive animate-pulse" />
      case "critical":
        return <AlertTriangle className="w-6 h-6 text-destructive" />
      case "warning":
        return <AlertCircle className="w-6 h-6 text-amber-500" />
      case "normal":
      default:
        return <CheckCircle className="w-6 h-6 text-emerald-500" />
    }
  }

  const getAlertBorderColor = (type: string) => {
    switch (type) {
      case "emergency":
      case "critical":
        return "border-l-destructive bg-destructive/5"
      case "warning":
        return "border-l-amber-500 bg-amber-500/5"
      case "normal":
      default:
        return "border-l-emerald-500 bg-emerald-50 dark:bg-emerald-950/30"
    }
  }

  return (
    <section className="py-16 md:py-24 bg-background">
      <div className="max-w-6xl mx-auto px-4 md:px-6">
        <div className="space-y-8">
          <div>
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">System Alerts</h2>
            <p className="text-muted-foreground mt-2">Real-time notifications and anomaly detection</p>
          </div>

          <div className="space-y-4">
            {alerts.map((alert) => (
              <Card
                key={alert.id}
                className={`border-l-4 ${getAlertBorderColor(
                  alert.type
                )} transition-all duration-300 hover:shadow-md animate-in fade-in slide-in-from-top-2`}
              >
                <CardContent className="pt-6 flex items-start gap-4">
                  <div className="mt-1">{getAlertIcon(alert.type)}</div>
                  <div className="flex-1">
                    <h3 className="font-semibold text-foreground">{alert.title}</h3>
                    <p className="text-sm text-muted-foreground mt-1">{alert.message}</p>
                    <p className="text-xs text-muted-foreground mt-2">{alert.time}</p>
                  </div>
                  {(alert.type === "critical" || alert.type === "emergency") && (
                    <div className="ml-4">
                      <div className="relative inline-block">
                        <div className="absolute inset-0 bg-red-500 rounded-full opacity-25 animate-ping"></div>
                        <div className="relative w-4 h-4 bg-red-500 rounded-full"></div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}

            {alerts.length === 0 && (
              <div className="text-center py-12">
                <CheckCircle className="w-12 h-12 text-emerald-500 mx-auto mb-4" />
                <p className="text-muted-foreground">No alerts. System is running normally.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
