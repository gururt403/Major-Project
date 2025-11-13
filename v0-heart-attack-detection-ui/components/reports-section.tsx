import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface SessionData {
  id: string
  date: string
  duration: string
  avgHeartRate: number
  status: "normal" | "warning" | "risk"
}

const sessions: SessionData[] = [
  { id: "1", date: "Today", duration: "45 min", avgHeartRate: 72, status: "normal" },
  { id: "2", date: "Yesterday", duration: "52 min", avgHeartRate: 75, status: "normal" },
  { id: "3", date: "2 days ago", duration: "38 min", avgHeartRate: 78, status: "warning" },
  { id: "4", date: "3 days ago", duration: "41 min", avgHeartRate: 71, status: "normal" },
]

export default function ReportsSection() {
  const getStatusBadge = (status: string) => {
    switch (status) {
      case "normal":
        return (
          <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200 border-emerald-300 dark:border-emerald-700">
            Normal
          </Badge>
        )
      case "warning":
        return (
          <Badge className="bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-200 border-amber-300 dark:border-amber-700">
            Caution
          </Badge>
        )
      case "risk":
        return (
          <Badge className="bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-200 border-red-300 dark:border-red-700">
            Risk
          </Badge>
        )
      default:
        return null
    }
  }

  return (
    <section id="reports" className="py-16 md:py-24 bg-muted/30">
      <div className="max-w-6xl mx-auto px-4 md:px-6">
        <div className="space-y-8">
          <div>
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">Monitoring Reports</h2>
            <p className="text-muted-foreground mt-2">Session history and heart rate trends</p>
          </div>

          <div className="overflow-x-auto">
            <div className="space-y-3">
              {sessions.map((session) => (
                <Card key={session.id} className="border border-border hover:shadow-md transition-all duration-300">
                  <CardContent className="pt-6">
                    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                      <div className="flex-1">
                        <h3 className="font-semibold text-foreground">{session.date}</h3>
                        <p className="text-sm text-muted-foreground">Session duration: {session.duration}</p>
                      </div>
                      <div className="flex items-center gap-6">
                        <div className="text-center">
                          <p className="text-xs text-muted-foreground">Avg Heart Rate</p>
                          <p className="text-2xl font-bold text-primary">{session.avgHeartRate}</p>
                          <p className="text-xs text-muted-foreground">BPM</p>
                        </div>
                        {getStatusBadge(session.status)}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
