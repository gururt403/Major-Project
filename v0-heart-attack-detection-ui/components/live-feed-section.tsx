"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { wsService, type VitalData } from "@/lib/websocket-service"

export default function LiveFeedSection() {
  const [frameData, setFrameData] = useState<string | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [faceDetected, setFaceDetected] = useState(false)
  const [heartRate, setHeartRate] = useState<number | null>(null)
  const [posture, setPosture] = useState<string>("Unknown")
  const [error, setError] = useState<string | null>(null)
  const [isMonitoring, setIsMonitoring] = useState(false)
  const [lastAlertLevel, setLastAlertLevel] = useState<string | null>(null)
  const imageRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    let unsubscribeMessage: (() => void) | null = null
    let unsubscribeConnect: (() => void) | null = null
    let unsubscribeError: (() => void) | null = null

    const initializeWebSocket = async () => {
      try {
        // Always subscribe to connection and error events
        unsubscribeConnect = wsService.onConnectionChange((connected) => {
          setIsConnected(connected)
          if (!connected) {
            setError("Disconnected from server. Attempting to reconnect...")
          } else {
            setError(null)
          }
        })

        unsubscribeError = wsService.onError((err) => {
          setError(err.message)
          console.error("WebSocket error:", err)
        })

        if (isMonitoring) {
          // Subscribe to vital data updates ONLY when monitoring
          unsubscribeMessage = wsService.onMessage((data: VitalData) => {
            // Only update if frame actually changed (avoid unnecessary re-renders)
            if (data.frame) {
              setFrameData(`data:image/jpeg;base64,${data.frame}`)
            }
            // Update other data less frequently to reduce re-renders
            if (data.face_detected !== faceDetected) {
              setFaceDetected(data.face_detected)
            }
            if (data.heart_rate !== null && data.heart_rate !== heartRate) {
              setHeartRate(Math.round(data.heart_rate))
            }
            if (data.posture && data.posture.name) {
              const newPosture = data.posture.name.replace(/_/g, " ")
              if (newPosture !== posture) {
                setPosture(newPosture)
              }
            }
          })

          // Connect to WebSocket
          await wsService.connect()
        } else {
          // Explicitly unsubscribe from messages when NOT monitoring
          if (unsubscribeMessage) {
            unsubscribeMessage()
            unsubscribeMessage = null
          }
          
          // Disconnect from WebSocket
          await wsService.disconnect()
          
          // Clear all data
          setFrameData(null)
          setFaceDetected(false)
          setHeartRate(null)
          setPosture("Unknown")
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Connection failed"
        setError(errorMessage)
        console.error("Failed to connect to WebSocket:", err)
      }
    }

    initializeWebSocket()

    return () => {
      // Proper cleanup on component unmount or dependency change
      console.log(`[LiveFeed] Cleaning up. isMonitoring: ${isMonitoring}`)
      
      // Unsubscribe from all handlers
      if (unsubscribeMessage) {
        unsubscribeMessage()
      }
      if (unsubscribeConnect) {
        unsubscribeConnect()
      }
      if (unsubscribeError) {
        unsubscribeError()
      }
      
      // Always disconnect when cleaning up
      wsService.disconnect()
      
      // Clear remaining handlers
      wsService.clearAllHandlers()
    }
  }, [isMonitoring])

  return (
    <section id="monitoring" className="py-16 md:py-24 bg-background">
      <div className="max-w-6xl mx-auto px-4 md:px-6">
        <div className="space-y-8">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">Live Monitoring Feed</h2>
              <p className="text-muted-foreground mt-2">Real-time facial and posture tracking</p>
            </div>
            <div className="flex gap-3">
              <Button
                onClick={() => {
                  console.log("[LiveFeed] Start monitoring clicked")
                  setIsMonitoring(true)
                }}
                disabled={isMonitoring}
                className="bg-emerald-600 hover:bg-emerald-700 text-white"
              >
                ▶️ Start Monitoring
              </Button>
              <Button
                onClick={() => {
                  console.log("[LiveFeed] Stop monitoring clicked")
                  setIsMonitoring(false)
                  setLastAlertLevel(null)
                  // Clear display immediately
                  setFrameData(null)
                  setHeartRate(null)
                  setPosture("Unknown")
                  setFaceDetected(false)
                  // Send stop signal to backend
                  wsService.stop()
                  wsService.disconnect()
                }}
                disabled={!isMonitoring}
                className="bg-red-600 hover:bg-red-700 text-white"
              >
                ⏹️ Stop Monitoring
              </Button>
            </div>
          </div>

          {error && (
            <div className="bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-lg p-4">
              <p className="text-red-800 dark:text-red-200 text-sm">{error}</p>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Video Feed Card */}
            <Card className="overflow-hidden border border-border shadow-sm">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">Face Detection</CardTitle>
                  <Badge className={`${
                    faceDetected
                      ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800"
                      : "bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-800"
                  }`}>
                    {faceDetected ? "Face Detected" : "No Face"}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="aspect-square bg-muted rounded-lg flex items-center justify-center relative overflow-hidden">
                  {frameData ? (
                    <img
                      ref={imageRef}
                      src={frameData}
                      alt="Live feed"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <>
                      <svg className="w-full h-full opacity-20" viewBox="0 0 200 200">
                        <circle cx="100" cy="80" r="30" fill="currentColor" />
                        <ellipse cx="100" cy="140" rx="35" ry="40" fill="currentColor" />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className={`w-24 h-24 border-2 border-primary rounded-lg ${
                          isConnected ? "animate-pulse" : ""
                        }`}></div>
                      </div>
                      {!isConnected && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                          <span className="text-white text-sm">Connecting...</span>
                        </div>
                      )}
                    </>
                  )}
                </div>
                <p className="text-xs text-muted-foreground mt-4">
                  rPPG Heart Rate Detection • {heartRate ? `${heartRate} BPM` : "No data"}
                </p>
                <div className="mt-2 flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${isConnected ? "bg-emerald-500" : "bg-red-500"} animate-pulse`}></div>
                  <span className="text-xs text-muted-foreground">
                    {isConnected ? "Connected" : "Disconnected"}
                  </span>
                </div>
              </CardContent>
            </Card>

            {/* Posture Analysis Card */}
            <Card className="overflow-hidden border border-border shadow-sm">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">Posture Analysis</CardTitle>
                  <Badge className="bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-800">
                    Tracking
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="aspect-square bg-muted rounded-lg flex items-center justify-center relative overflow-hidden">
                  {frameData ? (
                    <img
                      src={frameData}
                      alt="Posture analysis"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <>
                      <svg className="w-32 h-32 text-primary opacity-30" viewBox="0 0 100 200" fill="none">
                        <circle cx="50" cy="20" r="8" stroke="currentColor" strokeWidth="2" />
                        <line x1="50" y1="28" x2="50" y2="60" stroke="currentColor" strokeWidth="2" />
                        <line x1="50" y1="35" x2="35" y2="50" stroke="currentColor" strokeWidth="2" />
                        <line x1="50" y1="35" x2="65" y2="50" stroke="currentColor" strokeWidth="2" />
                        <line x1="50" y1="60" x2="35" y2="100" stroke="currentColor" strokeWidth="2" />
                        <line x1="50" y1="60" x2="65" y2="100" stroke="currentColor" strokeWidth="2" />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className={`w-24 h-24 border-2 border-primary rounded-lg ${
                          isConnected ? "animate-pulse" : ""
                        }`}></div>
                      </div>
                      {!isConnected && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                          <span className="text-white text-sm">Connecting...</span>
                        </div>
                      )}
                    </>
                  )}
                </div>
                <p className="text-xs text-muted-foreground mt-4">
                  MHAVH Posture Model • {posture}
                </p>
                <div className="mt-4 text-center">
                  <div className="inline-block bg-primary/10 rounded-full px-4 py-2">
                    <p className="text-sm font-semibold text-primary capitalize">{posture}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </section>
  )
}
