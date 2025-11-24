export interface VitalData {
  frame?: string
  face_detected?: boolean
  heart_rate?: number | null
  hr_risk?: 'normal' | 'warning' | 'critical'
  posture?: {
    name: string
    class: number
    confidence?: number
  }
  mesh?: any
  alert?: {
    level: 'none' | 'normal' | 'warning' | 'critical' | 'emergency'
    message: string
  }
  timestamp?: number
  status?: 'analyzing' | 'no_face' | 'processing'
}

export interface SessionStats {
  startTime: number
  duration: number
  hrHistory: Array<{ time: number; value: number }>
  avgHeartRate: number
  minHeartRate: number
  maxHeartRate: number
}

type VitalCallback = (data: VitalData) => void
type ConnectionCallback = (connected: boolean) => void
type ErrorCallback = (error: Error) => void

class WebSocketService {
  private ws: WebSocket | null = null
  private messageListeners: Set<VitalCallback> = new Set()
  private connectionListeners: Set<ConnectionCallback> = new Set()
  private errorListeners: Set<ErrorCallback> = new Set()
  private reconnectTimer: NodeJS.Timeout | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private isConnecting = false
  private url: string = 'ws://localhost:8000/ws/stream'
  private sessionStartTime: number | null = null
  private hrHistory: Array<{ time: number; value: number }> = []
  private readonly maxHistoryLength = 60

  async connect(url: string = 'ws://localhost:8000/ws/stream') {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      return
    }

    this.url = url
    this.isConnecting = true

    try {
      this.ws = new WebSocket(url)

      this.ws.onopen = () => {
        console.log('WebSocket connected')
        this.isConnecting = false
        this.reconnectAttempts = 0
        this.sessionStartTime = Date.now()
        this.hrHistory = []
        this.notifyConnectionListeners(true)
      }

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as VitalData
          
          if (data.heart_rate !== null && data.heart_rate !== undefined) {
            const now = Date.now()
            this.hrHistory.push({ time: now, value: data.heart_rate })
            const cutoff = now - 60000
            this.hrHistory = this.hrHistory.filter(h => h.time > cutoff)
          }
          
          this.notifyMessageListeners(data)
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        this.isConnecting = false
        this.notifyErrorListeners(new Error('WebSocket connection error'))
      }

      this.ws.onclose = () => {
        console.log('WebSocket disconnected')
        this.isConnecting = false
        this.notifyConnectionListeners(false)
        this.scheduleReconnect()
      }
    } catch (error) {
      console.error('Error creating WebSocket:', error)
      this.isConnecting = false
      this.notifyErrorListeners(error instanceof Error ? error : new Error('Unknown error'))
      this.scheduleReconnect()
    }
  }

  private scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached')
      return
    }

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
    }

    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000)
    this.reconnectAttempts++

    this.reconnectTimer = setTimeout(() => {
      console.log(`Reconnecting... (attempt ${this.reconnectAttempts})`)
      this.connect(this.url)
    }, delay)
  }

  async disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }

    if (this.ws) {
      this.ws.close()
      this.ws = null
    }

    this.reconnectAttempts = 0
    this.isConnecting = false
    this.notifyConnectionListeners(false)
  }

  onMessage(callback: VitalCallback) {
    this.messageListeners.add(callback)
    return () => this.messageListeners.delete(callback)
  }

  onConnectionChange(callback: ConnectionCallback) {
    this.connectionListeners.add(callback)
    return () => this.connectionListeners.delete(callback)
  }

  onError(callback: ErrorCallback) {
    this.errorListeners.add(callback)
    return () => this.errorListeners.delete(callback)
  }

  private notifyMessageListeners(data: VitalData) {
    this.messageListeners.forEach((callback) => callback(data))
  }

  private notifyConnectionListeners(connected: boolean) {
    this.connectionListeners.forEach((callback) => callback(connected))
  }

  private notifyErrorListeners(error: Error) {
    this.errorListeners.forEach((callback) => callback(error))
  }

  clearAllHandlers() {
    this.messageListeners.clear()
    this.connectionListeners.clear()
    this.errorListeners.clear()
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }

  send(data: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    } else {
      console.warn('WebSocket is not connected')
    }
  }

  stop() {
    this.send({ action: 'stop' })
  }

  getSessionStats(): SessionStats | null {
    if (!this.sessionStartTime) return null
    const duration = Date.now() - this.sessionStartTime
    const hrValues = this.hrHistory.map(h => h.value)
    return {
      startTime: this.sessionStartTime,
      duration,
      hrHistory: this.hrHistory,
      avgHeartRate: hrValues.length > 0 ? hrValues.reduce((a, b) => a + b, 0) / hrValues.length : 0,
      minHeartRate: hrValues.length > 0 ? Math.min(...hrValues) : 0,
      maxHeartRate: hrValues.length > 0 ? Math.max(...hrValues) : 0,
    }
  }

  getHeartRateHistory() {
    return this.hrHistory
  }

  getSessionDuration(): number {
    return this.sessionStartTime ? Date.now() - this.sessionStartTime : 0
  }
}

export const wsService = new WebSocketService()
